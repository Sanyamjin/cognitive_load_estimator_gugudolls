import streamlit as st
from user_profile import get_user_profile
from process_text import analyze_text

# Configure page
st.set_page_config(page_title="Project1 App", layout="wide")
st.title("📘 Text Complexity & Emotion Analysis")

# Step 1: User enters text
st.subheader("Enter Text for Analysis")
user_text = st.text_area("Paste your text here (e.g., a Canto from Dante's Inferno):", height=300)

# Step 2: User profile input
st.sidebar.header("📋 User Profile")
level = st.sidebar.selectbox("Reading Level", ["Beginner", "Intermediate", "Advanced"])
domain = st.sidebar.text_input("Familiar Domain (e.g., History, Religion, Mythology)")

# Step 3: Analysis Trigger
if st.button("🔍 Analyze"):

    if not user_text.strip():
        st.warning("⚠️ Please enter text to analyze.")
    else:
        # Build user profile
        profile = get_user_profile(level, domain)

        # Analyze text using backend logic
        with st.spinner("Analyzing text..."):
            metrics_df, wordcloud_fig, emotion_fig = analyze_text(user_text, profile)

        # Show results
        st.subheader("📊 Cognitive Load Metrics")
        st.dataframe(metrics_df)

        st.download_button("📥 Download Metrics CSV", data=metrics_df.to_csv(index=False).encode("utf-8"),
                           file_name="text_metrics.csv", mime="text/csv")

        st.subheader("☁️ Word Cloud")
        st.pyplot(wordcloud_fig)

        st.subheader("📈 Emotion Frequency")
        st.pyplot(emotion_fig)
