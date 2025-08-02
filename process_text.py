import pandas as pd
import textstat as tx
import re
from collections import Counter
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def analyze_text(text, profile):
    # Split text into cantos
    sections = [s.strip() for s in text.split("Canto") if s.strip()]
    cantos = [f"Canto {s}" for s in sections]

    metrics = []
    all_words = []

    for canto in cantos:
        blob = TextBlob(canto)
        words = re.findall(r'\b\w+\b', canto.lower())

        metrics.append({
            "FleschEase": tx.flesch_reading_ease(canto),
            "FKGrade": tx.flesch_kincaid_grade(canto),
            "SMOG": tx.smog_index(canto),
            "DifficultWords": tx.difficult_words(canto),
            "LexicalDiversity": round(len(set(words)) / len(words), 3),
            "AvgSentenceLength": tx.avg_sentence_length(canto),
            "Polarity": blob.sentiment.polarity,
            "Subjectivity": blob.sentiment.subjectivity
        })
        all_words.extend(words)

    df = pd.DataFrame(metrics)

    # Generate word cloud figure
    wordcloud = WordCloud(width=1000, height=500, background_color="white").generate(" ".join(all_words))
    fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')

    # Emotion frequency (simplified: use example emotions)
    example_emotions = ["joy", "fear", "anger", "sadness", "trust"]
    word_emotions = [word for word in all_words if word in example_emotions]
    emotion_counts = Counter(word_emotions)

    fig_emotion, ax_emotion = plt.subplots(figsize=(10, 5))
    ax_emotion.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
    ax_emotion.set_title("Emotion Frequency (Sample Words)")
    ax_emotion.set_ylabel("Frequency")
    ax_emotion.set_xlabel("Emotion")
    ax_emotion.grid(axis='y', linestyle='--', alpha=0.6)

    return df, fig_wc, fig_emotion
