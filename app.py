import streamlit as st
from project1_backend.user_profile import get_user_profile
from project1_backend.process_text import analyze_text

st.set_page_config(page_title="Basic App")
st.title("📘 Cognitive Load Estimator (Basic Demo)")
#step1 text input
user_text = st.text_area("Enter your text here:")

#step 2 profile form
with st.form("profile_form"):
    level = st.selectbox("Reading Level", ["Beginner", "Intermediate", "Advanced"])
    domain = st.text_input("Familiar Domain (e.g., Biology, History)")
    submitted = st.form_submit_button("Analyze")

#pass data to backend and display output
if submitted and user_text:
    profile = get_user_profile(level, domain)
    score, label = analyze_text(user_text, profile)
    st.success(f"Cognitive Load: {label} ({score})")
else:
    st.info("Please enter text and submit profile.")

