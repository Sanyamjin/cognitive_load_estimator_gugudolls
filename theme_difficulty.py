import os
import json
import numpy as np
import torch
import spacy
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
from wordfreq import zipf_frequency
from scipy.spatial.distance import cosine

# --- Initialize heavy models once ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.load("en_core_web_sm")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
sent_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)


@torch.no_grad()
def compute_surprisal(text):
    """Compute average token-level negative log-probability (bits/token)
    using GPT-2 as a proxy for surprisal."""
    enc = gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc.input_ids.to(device)
    target_ids = input_ids.clone()
    outputs = gpt2_model(input_ids, labels=target_ids)
    neg_log_likelihood = outputs.loss.item()  # mean over tokens
    surprisal_norm = np.clip(neg_log_likelihood / 10.0, 0, 1)
    return surprisal_norm


def compute_chunk_features(chunks, chunk_embeddings):
    """Compute per-chunk feature dictionary with GPT-2 surprisal & cohesion."""
    features = []
    for i, text in enumerate(chunks):
        doc = nlp(text)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        token_count = len(tokens)
        sentence_count = max(1, len(list(doc.sents)))

        # --- A. Linguistic complexity ---
        mean_sentence_length = token_count / sentence_count
        parse_depth = np.mean([len(list(sent.root.ancestors)) for sent in doc.sents]) if sentence_count else 0
        dep_lengths = [abs(tok.i - tok.head.i) for tok in doc if tok.dep_ != "ROOT"]
        avg_dep_length = np.mean(dep_lengths) if dep_lengths else 0
        clause_count = sum(1 for tok in doc if tok.dep_ in ["advcl", "ccomp", "xcomp", "acl"])

        # --- B. Lexical difficulty ---
        freqs = [zipf_frequency(tok.text.lower(), "en") for tok in tokens]
        avg_word_freq = np.mean(freqs) if freqs else 0
        type_token_ratio = len(set([t.lemma_ for t in tokens])) / token_count if token_count else 0
        jargon_density = 0  # placeholder

        # --- C. Semantic surprisal & coherence ---
        surprisal = compute_surprisal(text)

        # Cohesion: mean cosine similarity between adjacent sentence embeddings
        sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 2]
        if len(sents) > 1:
            sent_embs = sent_encoder.encode(sents, normalize_embeddings=True)
            sims = [1 - cosine(sent_embs[j], sent_embs[j + 1]) for j in range(len(sent_embs) - 1)]
            cohesion = np.mean(sims)
        else:
            cohesion = 1.0

        # Topic shift: between chunk embeddings
        topic_shift = 1 - cosine(chunk_embeddings[i], chunk_embeddings[i - 1]) if i > 0 else 1.0

        # --- D. Discourse/coref proxies ---
        pronouns = [t for t in doc if t.pos_ == "PRON"]
        anaphora_rate = len(pronouns) / token_count if token_count else 0
        coref_chain_length = 0  # could integrate a coref model later

        # --- E. Formatting cues ---
        has_equation = int(any(sym in text for sym in ["=", "+", "-", "×", "÷"]))
        has_table = int("table" in text.lower())
        has_figure = int("figure" in text.lower())

        features.append({
            "mean_sentence_length": mean_sentence_length,
            "parse_depth": parse_depth,
            "avg_dep_length": avg_dep_length,
            "clause_count": clause_count,
            "avg_word_freq": avg_word_freq,
            "type_token_ratio": type_token_ratio,
            "jargon_density": jargon_density,
            "surprisal": surprisal,
            "cohesion": cohesion,
            "topic_shift": topic_shift,
            "coref_chain_length": coref_chain_length,
            "anaphora_rate": anaphora_rate,
            "has_equation": has_equation,
            "has_table": has_table,
            "has_figure": has_figure,
            "tokens": token_count
        })
    return features


def normalize_features(features):
    """Min–max normalize each feature across document."""
    if not features:
        raise ValueError("❌ No features provided for normalization.")
    keys = [k for k in features[0].keys() if k != "tokens"]
    for k in keys:
        vals = np.array([f[k] for f in features])
        vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)
        for f in features:
            f[k] = float(np.clip((f[k] - vmin) / (vmax - vmin + 1e-9), 0, 1))
    return features


def compute_theme_difficulty(themes, features, chunk_theme_sims):
    """Weighted difficulty computation per theme."""
    alpha = {
        "linguistic": 0.30,
        "lexical": 0.25,
        "semantic": 0.25,
        "discourse": 0.10,
        "format": 0.10
    }
    theme_scores = {}

    for t in themes:
        sims = np.array(chunk_theme_sims[t])
        token_weights = np.array([f["tokens"] for f in features])
        w = sims * token_weights
        if w.sum() == 0:
            theme_scores[t] = 0
            continue

        def wavg(vals): return np.average(vals, weights=w)

        linguistic = wavg([np.mean([f["mean_sentence_length"], f["parse_depth"],
                                    f["avg_dep_length"], f["clause_count"]]) for f in features])
        lexical = wavg([np.mean([1 - f["avg_word_freq"], 1 - f["type_token_ratio"],
                                 f["jargon_density"]]) for f in features])
        semantic = wavg([np.mean([f["surprisal"], 1 - f["cohesion"],
                                  1 - f["topic_shift"]]) for f in features])
        discourse = wavg([np.mean([f["coref_chain_length"], f["anaphora_rate"]])
                          for f in features])
        fmt = wavg([np.mean([f["has_equation"], f["has_table"], f["has_figure"]])
                    for f in features])

        score = (alpha["linguistic"] * linguistic +
                 alpha["lexical"] * lexical +
                 alpha["semantic"] * semantic +
                 alpha["discourse"] * discourse +
                 alpha["format"] * fmt)

        theme_scores[t] = float(np.clip(score, 0, 1))
    return theme_scores


if __name__ == "__main__":
    # --- Step 1: Load JSON from previous module ---
    input_path = "theme_extraction_result2.json"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input file: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Step 2: Extract themes ---
    if isinstance(data["themes"], dict):
        themes = list(data["themes"].keys())
    else:
        themes = [t.get("name", f"theme_{t.get('theme_id', 'unknown')}") for t in data["themes"]]

    # --- Step 3: Prepare chunk/theme similarity map ---
    chunk_theme_sims = {t: [] for t in themes}
    for chunk in data.get("chunk_assignments", []):
        sims = {f"theme_{aid}": sim for aid, sim in chunk.get("assignments", [])}
        for t in themes:
            chunk_theme_sims[t].append(sims.get(t, 0.0))

    # --- Step 4: Compute features and difficulty ---
    chunks = [c["text"] for c in data.get("chunk_assignments", [])]
    print(f"Loaded {len(chunks)} chunks from JSON.")
    if not chunks:
        raise ValueError("❌ No chunks found in data['chunk_assignments']. Check JSON structure.")

    chunk_embeddings = sent_encoder.encode(chunks, normalize_embeddings=True)
    chunk_features = compute_chunk_features(chunks, chunk_embeddings)
    chunk_features = normalize_features(chunk_features)
    theme_scores = compute_theme_difficulty(themes, chunk_features, chunk_theme_sims)

    # --- Step 5: Save to output JSON ---
    output = {
        "themes": {t: {"difficulty_score": float(theme_scores[t])} for t in themes},
        "chunks": chunk_features
    }

    output_path = "theme_difficulty_result2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[✔] Theme difficulty scores written to {output_path}")
