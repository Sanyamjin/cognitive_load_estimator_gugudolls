import streamlit as st
import pandas as pd
import altair as alt
from file1 import get_themes_from_text

st.set_page_config(page_title="Theme Extraction Demo", layout="wide")

st.title("🧠 Cognitive Load Estimator — Theme Extraction Prototype")
st.write("""
Analyze any text to discover its **main semantic themes**, 
their **importance**, and **representative sentences**.
""")

# --- Text input box ---
text_input = st.text_area(
    "Paste your text below:",
    placeholder="Enter a paragraph, article, or research abstract...",
    height=300
)

# --- Button triggers analysis ---
if st.button("🔍 Analyze Themes"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Extracting themes... please wait..."):
            try:
                themes = get_themes_from_text(text_input)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()

        if not themes:
            st.warning("No clear themes detected. Try longer text.")
        else:
            st.success(f"✅ Extracted {len(themes)} themes successfully!")

            # Convert to DataFrame for visualization
            df = pd.DataFrame([
                {"Theme": f"Theme {t['theme_id']}", "Weight": t["weight"]}
                for t in themes
            ])

            # --- Chart ---
            st.subheader("📊 Theme Importance Overview")
            chart = (
                alt.Chart(df)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("Theme", sort=None),
                    y=alt.Y("Weight", title="Relative Importance"),
                    color="Theme"
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

            # --- Detailed theme display ---
            st.subheader("🧩 Theme Details")
            for idx, t in enumerate(themes, 1):
                with st.expander(f"Theme {idx}", expanded=False):
                    st.metric("Occupancy", t["occupancy"])
                    st.metric("Weight", t["weight"])
                    st.write("**Representative Sentences:**")
                    for s in t["representative_sentences"]:
                        st.markdown(f"- {s}")
else:
    st.info("👈 Paste text and click **Analyze Themes** to get started.")
