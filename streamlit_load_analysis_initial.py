import streamlit as st
import pandas as pd
from process_text import analyze_text

# Configure page
st.set_page_config(page_title="Project1 App", layout="wide")
st.title("📘 Text Complexity & Topic Analysis")

# Step 1: User enters text
st.subheader("Enter Text for Analysis")
user_text = st.text_area(
    "Paste your text here (e.g., a Canto from Dante's Inferno):",
    height=300
)

# Step 2: Initialize session state for analysis results
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "metrics_df" not in st.session_state:
    st.session_state.metrics_df = None
if "wordcloud_fig" not in st.session_state:
    st.session_state.wordcloud_fig = None
if "tfidf_top" not in st.session_state:
    st.session_state.tfidf_top = None
if "lda_topics" not in st.session_state:
    st.session_state.lda_topics = None
if "bert_clusters" not in st.session_state:
    st.session_state.bert_clusters = None
if "distinctive_topics" not in st.session_state:
    st.session_state.distinctive_topics = None

# Step 3: Analysis Trigger
if st.button("🔍 Analyze") and user_text.strip():
    with st.spinner("Analyzing text..."):
        (st.session_state.metrics_df, st.session_state.wordcloud_fig,
         st.session_state.tfidf_top, st.session_state.lda_topics,
         st.session_state.bert_clusters, st.session_state.distinctive_topics) = analyze_text(user_text)
        st.session_state.analysis_done = True

# Step 4: Display results if analysis is done
if st.session_state.analysis_done:

    # ---- Cognitive Load & Metrics ----
    st.subheader("📊 Cognitive Load Metrics")
    st.dataframe(st.session_state.metrics_df)

    st.download_button(
        "📥 Download Metrics CSV",
        data=st.session_state.metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="text_metrics.csv",
        mime="text/csv"
    )

    st.subheader("☁️ Word Cloud")
    st.pyplot(st.session_state.wordcloud_fig)

    st.subheader("📝 Top Keywords (TF-IDF)")
    st.write(st.session_state.tfidf_top)

    st.subheader("📂 LDA Topics")
    for topic, words in st.session_state.lda_topics:
        st.write(f"**{topic}**: {', '.join(words)}")

    st.subheader("🤖 BERT-based Clusters")
    st.dataframe(st.session_state.bert_clusters)

    st.subheader("📌 Distinctive Topics (PyTextRank)")
    for idx, topic in enumerate(st.session_state.distinctive_topics, 1):
        st.write(f"**Topic {idx}**: {topic}")

    # ---- Cognitive Load Assessment ----
    st.subheader("🧠 Assess Your Understanding of Each Topic")
    user_topic_scores = {}
    cognitive_scores = []

    for topic in st.session_state.distinctive_topics:
        # Store slider state per topic
        score = st.slider(
            label=f"How well do you understand '{topic}'?",
            min_value=1, max_value=5, value=3, key=f"slider_{topic}"
        )
        user_topic_scores[topic] = score

        # Determine Cognitive Load
        if score <= 2:
            load = "High"
        elif score == 3:
            load = "Medium"
        else:
            load = "Low"

        cognitive_scores.append({
            "Topic": topic,
            "UserUnderstanding": score,
            "CognitiveLoad": load
        })

    st.subheader("📊 Your Cognitive Load Analysis")
    st.table(pd.DataFrame(cognitive_scores))

    # Overall score
    overall_score = round(sum(user_topic_scores.values()) / len(user_topic_scores), 2)
    if overall_score <= 2:
        overall_load = "High"
    elif overall_score <= 3.5:
        overall_load = "Medium"
    else:
        overall_load = "Low"

    st.markdown(f"**Overall Understanding Score:** {overall_score} / 5")
    st.markdown(f"**Overall Cognitive Load:** {overall_load}")
