import json
import os

def load_theme_data(file_path="theme_extraction_result1.json"):
    """Load Level 1 theme extraction output."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}. Please run Level 1 first.")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def generate_theme_summaries(data):
    """Generate readable summaries for each theme using keywords and representative sentences."""
    theme_summaries = {}

    for theme in data["themes"]:
        theme_id = f"theme_{theme['theme_id']}"
        keywords = theme.get("keywords", [])
        reps = theme.get("representative_spans", [])

        # Use top 3 keywords for title
        title = ", ".join(keywords[:3]) if keywords else "General topic"
        # Use 1–2 example sentences
        examples = reps[:2] if reps else ["(No example text available)"]

        theme_summaries[theme_id] = {
            "title": title,
            "examples": examples
        }

    return theme_summaries

def get_user_familiarity(theme_summaries):
    """Ask the user for familiarity with each theme (0–100)."""
    familiarity = {}
    print("\n=== USER FAMILIARITY ESTIMATION ===")
    print("Rate your familiarity with each theme on a scale of 0–100:")
    print("0 = Never heard of it | 100 = Expert level knowledge\n")

    for theme_id, info in theme_summaries.items():
        print(f"\n🧩 {theme_id.upper()} — {info['title']}")
        print("Example sentences:")
        for i, ex in enumerate(info["examples"], 1):
            print(f"   {i}. {ex}")

        while True:
            try:
                val = input(f"\nHow familiar are you with this theme? (0–100): ").strip()
                score = float(val)
                if 0 <= score <= 100:
                    familiarity[theme_id] = round(score / 100, 3)
                    break
                else:
                    print("❌ Please enter a number between 0 and 100.")
            except ValueError:
                print("❌ Invalid input. Please enter a numeric value.")
    return familiarity

def save_familiarity_data(theme_summaries, familiarity, output_file="user_familiarity_result1.json"):
    """Save familiarity scores along with theme info."""
    result = {"themes": {}}
    for theme_id in theme_summaries.keys():
        result["themes"][theme_id] = {
            "title": theme_summaries[theme_id]["title"],
            "examples": theme_summaries[theme_id]["examples"],
            "familiarity_score": familiarity.get(theme_id, 0.0)
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved user familiarity data → {output_file}")

def main():
    data = load_theme_data("theme_extraction_result1.json")
    theme_summaries = generate_theme_summaries(data)
    familiarity = get_user_familiarity(theme_summaries)
    save_familiarity_data(theme_summaries, familiarity)

if __name__ == "__main__":
    main()
