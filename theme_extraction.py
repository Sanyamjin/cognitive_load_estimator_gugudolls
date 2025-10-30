# We'll implement the theme-extraction + sentence-tagging pipeline.
# The code tries to use sentence-transformers + hdbscan + umap if available.
# If not available in this environment, it falls back to TF-IDF + TruncatedSVD embeddings and DBSCAN clustering.
# Then it computes centroids, cosine similarities, assigns soft theme memberships, returns representative spans and keywords per theme.
# We'll run the pipeline on a small sample text to demonstrate outputs.

from typing import List, Dict, Tuple, Any
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import math
import textwrap

# Try to import optional libraries; if missing, we'll use fallbacks
HAS_SENTENCE_TRANSFORMERS = False
HAS_UMAP = False
HAS_HDBSCAN = False
HAS_SPACY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception as e:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

try:
    import spacy
    HAS_SPACY = True
    # try to load English model if available
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        try:
            nlp = spacy.load("en_core_web_trf")
        except Exception:
            HAS_SPACY = False
            nlp = None
except Exception:
    HAS_SPACY = False
    nlp = None

print("Environment: sentence-transformers:", HAS_SENTENCE_TRANSFORMERS, 
      "| umap:", HAS_UMAP, "| hdbscan:", HAS_HDBSCAN, "| spacy:", HAS_SPACY)


# --- Utilities ---

def simple_sentence_tokenize(text: str) -> List[str]:
    # A conservative sentence splitter using regex; keeps abbreviations attached.
    # This is intentionally simple and works reasonably for demo texts.
    text = text.strip().replace("\n", " ")
    # split on . ? ! but keep decimals and common abbreviations naive check
    sentence_end = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9"])')
    sentences = [s.strip() for s in sentence_end.split(text) if s.strip()]
    # fallback: split on newlines if nothing found
    if not sentences:
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
    return sentences

def tokenize_simple(s: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", s, re.UNICODE)

def merge_short_sentences(sentences: List[str], min_tokens=6) -> List[str]:
    out = []
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        toks = tokenize_simple(sent)
        if len(toks) < min_tokens and i < len(sentences) - 1:
            # merge with next
            merged = sent + " " + sentences[i+1]
            out.append(merged)
            i += 2
        else:
            out.append(sent)
            i += 1
    return out

# Embedding function with fallback
def embed_sentences(sentences: List[str], model_name="all-mpnet-base-v2", dim=384):
    if HAS_SENTENCE_TRANSFORMERS:
        model = SentenceTransformer(model_name)
        emb = model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
        return emb, "sbert"
    else:
        # fallback: tf-idf + SVD to dense vectors
        tf = TfidfVectorizer(stop_words='english', max_features=2000)
        X = tf.fit_transform(sentences)
        svd = TruncatedSVD(n_components=min(64, X.shape[1]-1 if X.shape[1]>1 else 1), random_state=42)
        Xd = svd.fit_transform(X)
        # normalize
        Xd_norm = Xd / (np.linalg.norm(Xd, axis=1, keepdims=True) + 1e-9)
        return Xd_norm, "tfidf_svd"

def reduce_dimensionality(embeddings, n_components=64):
    import umap
    n_samples = embeddings.shape[0]

    # Handle edge cases automatically
    if n_samples < 10:
        print(f"[UMAP skipped: only {n_samples} samples]")
        return embeddings, "none"

    # Clamp neighbors and components safely
    n_neighbors = min(15, max(2, n_samples // 2))
    n_components = min(n_components, max(2, n_samples - 2))

    print(f"[UMAP] n_samples={n_samples}, n_neighbors={n_neighbors}, n_components={n_components}")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        random_state=42,
        metric="cosine"
    )
    return reducer.fit_transform(embeddings), "umap"
'''
def reduce_dimensionality(embeddings, n_components=64):
    import umap
    n_samples = embeddings.shape[0]

    # For small datasets, disable UMAP and return embeddings directly
    if n_samples < 15:
        print(f"[UMAP skipped: only {n_samples} samples]")
        return embeddings, "none"

    reducer = umap.UMAP(
        n_neighbors=min(15, n_samples - 1),
        n_components=min(n_components, n_samples - 1),
        random_state=42,
    )
    return reducer.fit_transform(embeddings), "umap"


def reduce_dimensionality(embeddings: np.ndarray, n_components=32):
    if HAS_UMAP and embeddings.shape[1] > n_components:
        import umap.umap_ as umaplib
        reducer = umaplib.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(embeddings), "umap"
    else:
        return embeddings, "identity"
'''

def cluster_embeddings(embeddings: np.ndarray, method_prefer_hdbscan=True):
    if HAS_HDBSCAN and method_prefer_hdbscan:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        labels = clusterer.fit_predict(embeddings)
        return labels, "hdbscan"
    else:
        # fallback: DBSCAN with cosine metric (requires sklearn >=0.22)
        # compute cosine distances
        # sklearn DBSCAN doesn't support cosine directly for dense vectors prior to certain versions; so transform via 1-cosine
        sim = cosine_similarity(embeddings)
        # transform to distance
        dist = 1.0 - sim
        # use DBSCAN with precomputed distances
        try:
            db = DBSCAN(metric='precomputed', eps=0.6, min_samples=1)
            labels = db.fit_predict(dist)
            return labels, "dbscan_precomputed"
        except Exception:
            # fallback: Agglomerative with n_clusters = 3 heuristically
            agg = AgglomerativeClustering(n_clusters=3)
            labels = agg.fit_predict(embeddings)
            return labels, "agglomerative_k3"

def compute_centroids(embeddings: np.ndarray, labels: np.ndarray):
    centroids = {}
    for lab in np.unique(labels):
        if lab == -1:
            continue
        idx = np.where(labels == lab)[0]
        centroids[int(lab)] = embeddings[idx].mean(axis=0)
    return centroids

def top_k_representative_sentences(sentences: List[str], embeddings: np.ndarray, centroids: Dict[int, np.ndarray], k=3):
    reps = {}
    for lab, cent in centroids.items():
        sims = cosine_similarity(embeddings, cent.reshape(1, -1)).reshape(-1)
        # pick top k among sentences that have label==lab
        reps[lab] = [sentences[i] for i in sims.argsort()[::-1][:k]]
    return reps

def cluster_keywords_by_tfidf(sentences: List[str], labels: np.ndarray, topn=8):
    # For each cluster, compute top tf-idf terms from member sentences
    keywords = {}
    for lab in sorted(set(labels.tolist())):
        if lab == -1:
            continue
        member_texts = [sentences[i] for i in range(len(sentences)) if labels[i]==lab]
        if not member_texts:
            keywords[lab] = []
            continue
        tf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
        X = tf.fit_transform(member_texts)
        # get top terms by average tf-idf across cluster
        scores = np.asarray(X.mean(axis=0)).ravel()
        terms = np.array(tf.get_feature_names_out())
        top_idx = scores.argsort()[::-1][:topn]
        keywords[lab] = terms[top_idx].tolist()
    return keywords

def mmr_rerank(candidates: List[str], query_emb: np.ndarray, emb_map: Dict[str, np.ndarray], top_n=5, diversity=0.7):
    # Simple MMR: selects items balancing relevance (cosine to query) and diversity (cosine to selected)
    selected = []
    candidate_set = candidates.copy()
    sims = {c: float(cosine_similarity(emb_map[c].reshape(1,-1), query_emb.reshape(1,-1))[0,0]) for c in candidates}
    while len(selected) < min(top_n, len(candidate_set)):
        if not selected:
            # pick best relevance
            best = max(candidate_set, key=lambda c: sims[c])
            selected.append(best)
            candidate_set.remove(best)
            continue
        # for each candidate compute: lambda*relevance - (1-lambda)*max_sim_to_selected
        scores = {}
        for c in candidate_set:
            sim_to_selected = max([float(cosine_similarity(emb_map[c].reshape(1,-1), emb_map[s].reshape(1,-1))[0,0]) for s in selected])
            scores[c] = diversity * sims[c] - (1-diversity) * sim_to_selected
        best = max(scores.items(), key=lambda x: x[1])[0]
        selected.append(best)
        candidate_set.remove(best)
    return selected

# --- Main pipeline function ---

def extract_themes_and_tag_sentences(text: str, 
                                    min_tokens_merge=6, 
                                    sim_threshold=0.30, 
                                    reduce_dim=True, 
                                    umap_dim=32,
                                    debug=False) -> Dict[str, Any]:
    # 1. chunking
    sentences = simple_sentence_tokenize(text)
    sentences = merge_short_sentences(sentences, min_tokens=min_tokens_merge)
    if debug:
        print("Chunks (sentences):", sentences)
    # 2. embeddings
    embeddings, emb_method = embed_sentences(sentences)
    if debug:
        print("Embedding method:", emb_method, "shape:", embeddings.shape)
    # 3. optional dim reduction
    if reduce_dim:
        emb_reduced, red_method = reduce_dimensionality(embeddings, n_components=umap_dim)
    else:
        emb_reduced, red_method = embeddings, "identity"
    if debug:
        print("Reduced by:", red_method, "shape:", emb_reduced.shape)
    # 4. clustering
    labels, cluster_method = cluster_embeddings(emb_reduced)
    if debug:
        print("Clustering method:", cluster_method, "labels:", labels)
    # 5. centroids computed on original embedding space (not reduced)
    centroids = compute_centroids(embeddings, labels)
    # 6. compute soft similarities s_{c,t}
    sims = {}
    for i, emb in enumerate(embeddings):
        sims[i] = {}
        for t, cent in centroids.items():
            s = float(cosine_similarity(emb.reshape(1,-1), cent.reshape(1,-1))[0,0])
            if s > sim_threshold:
                sims[i][t] = s
    # 7. occupancy_by_count and by_weight (later if needed)
    # 8. representative spans and keywords
    reps = top_k_representative_sentences(sentences, embeddings, centroids, k=3)
    keywords = cluster_keywords_by_tfidf(sentences, labels, topn=8)
    
    # build outputs
    themes = []
    for t in sorted(centroids.keys()):
        theme = {
            "theme_id": int(t),
            "centroid_norm": float(np.linalg.norm(centroids[t])),
            "representative_spans": reps.get(t, []),
            "keywords": keywords.get(t, [])
        }
        themes.append(theme)
    
    # per-chunk assignments
    chunk_assignments = []
    for i, sent in enumerate(sentences):
        assigned = [(int(t), float(s)) for t,s in sims[i].items()]
        # also include primary label if nothing above threshold
        primary_label = int(labels[i])
        if not assigned and primary_label != -1:
            # compute similarity to primary centroid and include it
            cent = centroids.get(primary_label)
            if cent is not None:
                s = float(cosine_similarity(embeddings[i].reshape(1,-1), cent.reshape(1,-1))[0,0])
                assigned = [(primary_label, s)]
        chunk_assignments.append({
            "chunk_index": i,
            "text": sent,
            "tokens": len(tokenize_simple(sent)),
            "assignments": sorted(assigned, key=lambda x: x[1], reverse=True)
        })
    
    output = {
        "sentences": sentences,
        "emb_method": emb_method,
        "reduce_method": red_method,
        "cluster_method": cluster_method,
        "themes": themes,
        "chunk_assignments": chunk_assignments,
        "labels": labels.tolist()
    }
    return output

# --- Demo run on sample text ---
sample_text = """
Human cannibalism is the act or practice of humans eating the flesh or internal organs of other human beings. A person who practices cannibalism is called a cannibal. The meaning of "cannibalism" has been extended into zoology to describe animals consuming parts of individuals of the same species as food.

Anatomically modern humans, Neanderthals, and Homo antecessor are known to have practised cannibalism to some extent in the Pleistocene.[1][2][3][4][5] Cannibalism was occasionally practised in Egypt during ancient and Roman times, as well as later during severe famines.[6][7] The Island Caribs of the Lesser Antilles, whose name is the origin of the word cannibal, acquired a long-standing reputation as eaters of human flesh, reconfirmed when their legends were recorded in the 17th century.[8] Some controversy exists over the accuracy of these legends and the prevalence of actual cannibalism in the culture.

Reports describing cannibal practices were most often recorded by outsiders and were especially during the colonialist epoch commonly used to justify the subjugation and exploitation of non-European peoples. Therefore, such sources need to be particularly critically examined before being accepted. A few scholars argue that no firm evidence exists that cannibalism has ever been a socially acceptable practice anywhere in the world,[9] but such views have been largely rejected as irreconcilable with the actual evidence.[10][11]

Cannibalism has been well documented in much of the world, including Fiji (once nicknamed the "Cannibal Isles"),[12] the Amazon Basin, the Congo, and the Māori people of New Zealand.[13] Cannibalism was also practised in New Guinea and in parts of the Solomon Islands, and human flesh was sold at markets in some parts of Melanesia[14] and the Congo Basin.[15][16] A form of cannibalism popular in early modern Europe was the consumption of body parts or blood for medical purposes. Reaching its height during the 17th century, this practice continued in some cases into the second half of the 19th century.[17]

Cannibalism has occasionally been practised as a last resort by people suffering from famine. Well-known examples include the ill-fated Donner Party (1846–1847), the Holodomor (1932–1933), and the crash of Uruguayan Air Force Flight 571 (1972), after which the survivors ate the bodies of the dead. Additionally, there are cases of people engaging in cannibalism for sexual pleasure, such as Albert Fish, Issei Sagawa, Jeffrey Dahmer, and Armin Meiwes. Cannibalism has been both practised and fiercely condemned in several recent wars, especially in Liberia[18] and the Democratic Republic of the Congo.[19] It was still practised in Papua New Guinea as of 2012, for cultural reasons.[20][21]

Cannibalism has been said to test the bounds of cultural relativism because it challenges anthropologists "to define what is or is not beyond the pale of acceptable human behavior".[22]

Etymology
The word "cannibal" is derived from Spanish caníbal or caríbal, originally used as a name variant for the Kalinago (Island Caribs), a people from the West Indies said to have eaten human flesh.[23] The older term anthropophagy, meaning "eating humans", is also used for human cannibalism.[24]

Reasons and types
Cannibalism has been practised under a variety of circumstances and for various motives. To adequately express this diversity, Shirley Lindenbaum suggests that "it might be better to talk about 'cannibalisms'" in the plural.[25]

Institutionalized, survival, and pathological cannibalism
Cannibalism during Russian famine (1921)
Survival cannibalism during the Russian famine of 1921–1922
One major distinction is whether cannibal acts are

accepted by the culture in which they occur ("institutionalized cannibalism"),
practised under starvation conditions to ensure one's immediate survival ("survival cannibalism"), or
committed by isolated individuals considered criminal and often pathological by society at large ("cannibalism as psychopathology" or as "aberrant behavior").[26]
Institutionalized cannibalism, sometimes also called "learned cannibalism", is the consumption of human body parts as "an institutionalized practice" generally accepted in the culture where it occurs.[27]


Sketch of the Mignonette by Tom Dudley. In English common law, the R v Dudley and Stephens (1884) case banned survival cannibalism after maritime disasters, which had been a widely accepted custom of the sea.
By contrast, survival cannibalism means "the consumption of others under conditions of starvation such as shipwreck, military siege, and famine, in which persons normally averse to the idea are driven [to it] by the will to live".[28] Also known as famine cannibalism,[29][30] such forms of cannibalism resorted to only in situations of extreme necessity have occurred in many cultures where cannibalism is otherwise clearly rejected. The survivors of the shipwrecks of the Essex and Méduse in the 19th century are said to have engaged in cannibalism, as did the members of Franklin's lost expedition and the Donner Party.

Such cases often involve only necro-cannibalism (eating the corpse of someone already dead) as opposed to homicidal cannibalism (killing someone for food). In modern English law, the latter is always considered a crime, even in the most trying circumstances. The case of R v Dudley and Stephens, in which two men were found guilty of murder for killing and eating a cabin boy while adrift at sea in a lifeboat, set the precedent that necessity is no defence to a charge of murder. This decision outlawed and effectively ended the practice of shipwrecked sailors drawing lots in order to determine who would be killed and eaten to prevent the others from starving, a time-honoured practice formerly known as a "custom of the sea".[31]

In other cases, cannibalism is an expression of a psychopathology or mental disorder, condemned by the society in which it occurs and "considered to be an indicator of [a] severe personality disorder or psychosis".[28] Well-known cases include Albert Fish, Issei Sagawa, and Armin Meiwes. Fantasies of cannibalism, whether acted out or not, are not specifically mentioned in manuals of mental disorders such as the DSM, presumably because at least serious cases (that lead to murder) are very rare.[32]

Exo-, endo-, and autocannibalism
Within institutionalized cannibalism, exocannibalism is often distinguished from endocannibalism. Endocannibalism refers to the consumption of a person from the same community. Often it is a part of a funerary ceremony, similar to burial or cremation in other cultures. The consumption of the recently deceased in such rites can be considered "an act of affection"[33] and a major part of the grieving process.[34] It has also been explained as a way of guiding the souls of the dead into the bodies of living descendants.[35]

In contrast, exocannibalism is the consumption of a person from outside the community. It is frequently "an act of aggression, often in the context of warfare",[33] where the flesh of killed or captured enemies may be eaten to celebrate one's victory over them.[35]

Some scholars explain both types of cannibalism as due to a belief that eating a person's flesh or internal organs will endow the cannibal with some of the positive characteristics of the deceased.[36] However, several authors investigating exocannibalism in New Zealand, New Guinea, and the Congo Basin observe that such beliefs were absent in these regions.[37][38][39][40]

A further type, different from both exo- and endocannibalism, is autocannibalism (also called autophagy or self-cannibalism), "the act of eating parts of oneself".[41] It does not ever seem to have been an institutionalized practice, but it occasionally occurs as pathological behaviour or due to other reasons such as curiosity. Also on record are instances of forced autocannibalism committed as acts of aggression, where individuals are forced to eat parts of their own bodies as a form of torture.[41]

Exocannibalism is thus often associated with the consumption of enemies as an act of aggression, a practice also known as war cannibalism.[42][43] Endocannibalism is often associated with the consumption of deceased relatives in funerary rites driven by affection — a practice known as funerary[42][44] or mortuary cannibalism.[45]

Additional motives

An 18th-century albarello used for storing mummia. Medicinal cannibalism was widespread in many countries of early modern Europe.
Medicinal cannibalism (also called medical cannibalism) means "the ingestion of human tissue ... as a supposed medicine or tonic". In contrast to other forms of cannibalism, which Europeans generally frowned upon, the "medicinal ingestion" of various "human body parts was widely practiced throughout Europe from the sixteenth to the eighteenth centuries", with early records of the practice going back to the first century CE.[33] It was also frequently practised in China.[46]

Sacrificial cannibalism refers the consumption of the flesh of victims of human sacrifice, for example among the Aztecs.[41] Human and animal remains excavated in Knossos, Crete, have been interpreted as evidence of a ritual in which children and sheep were sacrificed and eaten together during the Bronze Age.[47] According to Ancient Roman reports, the Celts in Britain practised sacrificial cannibalism,[48] and archaeological evidence backing these claims has been found.[49]

Infanticidal cannibalism or cannibalistic infanticide refers to cases where newborns or infants are killed because they are "considered unwanted or unfit to live" and then "consumed by the mother, father, both parents or close relatives".[44][50] Infanticide followed by cannibalism was practised in various regions, but is particularly well documented among Aboriginal Australians.[50][51] Among animals, such behaviour is called filial cannibalism, and it is common in many species, especially among fish.[52][53]

Human predation is the hunting of people from unrelated and possibly hostile groups in order to eat them. In parts of the Southern New Guinea lowland rain forests, hunting people "was an opportunistic extension of seasonal foraging or pillaging strategies", with human bodies just as welcome as those of animals as sources of protein, according to the anthropologist Bruce M. Knauft. As populations living near coasts and rivers were usually better nourished and hence often physically larger and stronger than those living inland, they "raided inland 'bush' peoples with impunity and often with little fear of retaliation".[54] Cases of human predation are also on record for the neighbouring Bismarck Archipelago[55] and for Australia.[56] In the Congo Basin, there lived groups such as the Bankutu who hunted humans for food even when game was plentiful.[57][58][59]

The term innocent cannibalism has been used for cases of people eating human flesh without knowing what they are eating. It is a subject of myths, such as the myth of Thyestes who unknowingly ate the flesh of his own sons.[41] There are also actual cases on record, for example from the Congo Basin, where cannibalism had been quite widespread and where even in the 1950s travellers were sometimes served a meat dish, learning only afterwards that the meat had been of human origin.[15][60]
"""

result = extract_themes_and_tag_sentences(sample_text, debug=True)

# Pretty-print result
print("\n--- THEMES FOUND ---")
for th in result["themes"]:
    print(f"\nTheme {th['theme_id']}: centroid_norm={th['centroid_norm']:.3f}")
    print("Representative spans:")
    for s in th["representative_spans"]:
        print(" -", textwrap.shorten(s, width=120))
    print("Keywords:", ", ".join(th["keywords"][:8]))

print("\n--- CHUNK ASSIGNMENTS ---")
for ca in result["chunk_assignments"]:
    print(f"\nChunk {ca['chunk_index']} (tokens={ca['tokens']}): {textwrap.shorten(ca['text'], width=120)}")
    if ca["assignments"]:
        for (t,s) in ca["assignments"]:
            print(f"   -> Theme {t} (sim={s:.3f})")
    else:
        print("   -> No theme above sim threshold; primary label:", result["labels"][ca["chunk_index"]])

# Save a small JSON-like summary to /mnt/data for download if needed
import json, os
outpath = "C:/Users/IdeaPad/Desktop/cogload/again/theme_extraction_result2.json"
with open(outpath, "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved result to", outpath)
