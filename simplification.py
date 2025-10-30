import os
import json
import time
from tqdm import tqdm
import google.generativeai as genai

# ----------------------------------------------------------
# ✅ CONFIG
# ----------------------------------------------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError("❌ GOOGLE_API_KEY not found. Please run: setx GOOGLE_API_KEY \"your_api_key_here\"")

genai.configure(api_key=API_KEY)

# Valid Gemini models: "models/gemini-2.0-pro", "models/gemini-1.5-pro", "models/gemini-1.5-flash"
MODEL_NAME = "models/gemini-2.5-flash-lite"

model = genai.GenerativeModel(MODEL_NAME)

INPUT_FILE = "simplification_strategy.json"  # <-- your input JSON
OUTPUT_FILE = "simplified_sentences.json"

# ----------------------------------------------------------
# ✅ LOAD INPUT DATA
# ----------------------------------------------------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file '{INPUT_FILE}' not found.")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = data.get("sentences", [])
if not sentences:
    raise ValueError("No 'sentences' key or empty list found in input JSON.")

# ----------------------------------------------------------
# ✅ FUNCTION TO CALL GEMINI WITH RETRIES
# ----------------------------------------------------------
def simplify_sentence(text, simplification_type, retries=3):
    prompt = (
        f"Simplify the following sentence based on the simplification type: {simplification_type}.\n"
        f"- If 'conceptual', explain or define complex concepts simply.\n"
        f"- If 'linguistic', rephrase to make the sentence clearer and easier to read.\n"
        f"- If 'both', do both conceptual and linguistic simplification.\n\n"
        f"Sentence: {text}"
    )

    for attempt in range(retries):
        try:
            response = model.generate_content([prompt])

            # Handle SDK variations
            if hasattr(response, "text"):
                return response.text.strip()
            elif hasattr(response, "candidates"):
                return response.candidates[0].content.parts[0].text.strip()
            else:
                return str(response).strip()
        except Exception as e:
            print(f"⚠️ Gemini error (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(2)
    return text  # fallback: return original if failed


# ----------------------------------------------------------
# ✅ MAIN PROCESSING LOOP
# ----------------------------------------------------------
simplified_sentences = []

for i, s in enumerate(tqdm(sentences, desc="Simplifying sentences")):
    text = s.get("text", "")
    simplification_type = s.get("simplification_type", "both")

    simplified = simplify_sentence(text, simplification_type)

    simplified_sentences.append({
        "chunk_index": s.get("chunk_index"),
        "theme_id": s.get("theme_id"),
        "original": text,
        "simplified": simplified,
        "simplification_type": simplification_type,
        "metrics": s.get("metrics", {})
    })

# ----------------------------------------------------------
# ✅ SAVE OUTPUT
# ----------------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(simplified_sentences, f, indent=2, ensure_ascii=False)

print(f"\n✅ Simplified sentences saved to '{OUTPUT_FILE}'")
