import re
import pandas as pd
import spacy
import textstat as tx
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk

# Download stopwords once
nltk.download('stopwords')

import pytextrank

def extract_distinctive_topics(text, n_topics=10):
    """
    Extracts a list of distinctive topics from the input text using PyTextRank.
    Returns the top N keyphrases as topics.
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")  # Add PyTextRank pipeline

    doc = nlp(text)
    topics = []
    for phrase in doc._.phrases[:n_topics]:
        topics.append(phrase.text)

    return topics

def load_spacy_model():
    """Load spaCy model safely, download if missing."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def preprocess_text(text, nlp_model):
    """Full preprocessing: lowercase, remove punctuation/numbers, lemmatize, remove stopwords."""
    text_clean = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    doc = nlp_model(text_clean)
    stop_words = set(stopwords.words('english'))
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words]
    return " ".join(tokens)

def analyze_text(text):
    """
    Analyzes text for readability, word cloud, TF-IDF, LDA topics, BERT clusters,
    and distinctive topics. Emotion frequency and TextRank are removed.
    """

    # -------- 1. Split text into sections --------
    sections = [s.strip() for s in re.split(r"(Canto\s+\d+)", text) if s.strip()]
    if not sections:
        sections = [text]
    corpus = sections

    metrics = []
    all_words = []

    for sec in corpus:
        blob = TextBlob(sec)
        words = re.findall(r'\b\w+\b', sec.lower())

        metrics.append({
            "FleschEase": tx.flesch_reading_ease(sec),
            "FKGrade": tx.flesch_kincaid_grade(sec),
            "SMOG": tx.smog_index(sec),
            "DifficultWords": tx.difficult_words(sec),
            "LexicalDiversity": round(len(set(words)) / len(words), 3) if words else 0,
            "AvgSentenceLength": tx.avg_sentence_length(sec),
            "Polarity": blob.sentiment.polarity,
            "Subjectivity": blob.sentiment.subjectivity
        })
        all_words.extend(words)

    metrics_df = pd.DataFrame(metrics)

    # -------- 2. Word Cloud --------
    wordcloud = WordCloud(width=1000, height=500, background_color="white").generate(" ".join(all_words))
    fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')

    # -------- 3. Keyword Extraction (TF-IDF) --------
    nlp = load_spacy_model()
    clean_corpus = [preprocess_text(sec, nlp_model=nlp) for sec in corpus]

    if len(clean_corpus) >= 1:
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(clean_corpus)
        tfidf_top = pd.DataFrame(
            X.toarray(), columns=vectorizer.get_feature_names_out()
        ).sum().sort_values(ascending=False).head(10)
    else:
        tfidf_top = pd.Series([])

    # -------- 4. Topic Modeling (LDA) --------
    lda_topics = []
    if len(clean_corpus) > 1:
        lda = LatentDirichletAllocation(n_components=min(3, len(clean_corpus)), random_state=42)
        lda.fit(X)
        for idx, topic in enumerate(lda.components_):
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-8:]]
            lda_topics.append((f"Topic {idx}", top_words))

    # -------- 5. BERT + Clustering --------
    if len(clean_corpus) > 1:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(clean_corpus, show_progress_bar=False)
        km = KMeans(n_clusters=min(3, len(clean_corpus)), random_state=42)
        labels = km.fit_predict(embeddings)
        bert_clusters = pd.DataFrame({"Section": corpus, "Cluster": labels})
    else:
        bert_clusters = pd.DataFrame({"Section": corpus, "Cluster": [0]})
    # -------- 6. Distinctive Topics (PyTextRank) --------
    distinctive_topics = extract_distinctive_topics(text)


    # Return everything
    return metrics_df, fig_wc, tfidf_top, lda_topics, bert_clusters, distinctive_topics
