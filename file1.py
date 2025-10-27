import numpy as np
import nltk
import hdbscan
import umap
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


class ThemeExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2', min_cluster_size=2, min_sentence_len=30):
        self.model = SentenceTransformer(model_name)
        self.min_cluster_size = min_cluster_size
        self.min_sentence_len = min_sentence_len

        # small summarization model for naming themes
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            tokenizer="sshleifer/distilbart-cnn-12-6"
        )

    def preprocess_text(self, text):
        """Split text into longer, meaning-rich segments."""
        sentences = nltk.sent_tokenize(text)
        merged = []
        buffer = ""
        for sent in sentences:
            if len(buffer) + len(sent) < self.min_sentence_len:
                buffer += " " + sent
            else:
                if buffer:
                    merged.append(buffer.strip())
                buffer = sent
        if buffer:
            merged.append(buffer.strip())
        return merged

    def _generate_theme_name(self, sentences):
        """Use transformer summarization for a short, descriptive theme name."""
        if not sentences:
            return "General Theme"

        joined_text = " ".join(sentences)
        try:
            summary = self.summarizer(
                joined_text,
                max_length=10,
                min_length=3,
                do_sample=False
            )[0]["summary_text"]
            theme_name = summary.strip().rstrip(".")
            return theme_name
        except Exception:
            # fallback to TF-IDF keywords if summarizer fails
            vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
            X = vectorizer.fit_transform([joined_text])
            keywords = vectorizer.get_feature_names_out()
            return " ".join(keywords[:3]).title() if len(keywords) > 0 else "General Theme"

    def extract_themes(self, text):
        sentences = self.preprocess_text(text)
        if not sentences:
            return []

        embeddings = self.model.encode(sentences, show_progress_bar=False)
        n_sent = len(sentences)

        # --- Dynamically adjust UMAP parameters
        n_neighbors = min(15, max(2, n_sent - 1))
        n_components = min(10, max(2, n_sent - 1))

        # --- Dimensionality reduction
        if n_sent > 5:
            try:
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0.0,
                    n_components=n_components,
                    random_state=42
                )
                reduced_embeddings = reducer.fit_transform(embeddings)
            except Exception:
                reduced_embeddings = embeddings
        else:
            reduced_embeddings = embeddings

        # --- Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, metric='euclidean')
        labels = clusterer.fit_predict(reduced_embeddings)

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(i)

        results = []
        total_sentences = len(sentences)
        for label, indices in clusters.items():
            cluster_embeds = embeddings[indices]
            centroid = np.mean(cluster_embeds, axis=0)
            sims = cosine_similarity([centroid], cluster_embeds)[0]

            topk = np.argsort(sims)[::-1][:3]
            reps = [sentences[indices[i]] for i in topk]

            occupancy = len(indices) / total_sentences
            weight = float(np.mean(sims))
            theme_name = self._generate_theme_name(reps)

            results.append({
                "theme_id": int(label),
                "theme_name": theme_name,
                "occupancy": round(occupancy, 3),
                "weight": round(weight, 3),
                "representative_sentences": reps
            })

        return sorted(results, key=lambda x: x["weight"], reverse=True)


def get_themes_from_text(text: str):
    extractor = ThemeExtractor()
    return extractor.extract_themes(text)
