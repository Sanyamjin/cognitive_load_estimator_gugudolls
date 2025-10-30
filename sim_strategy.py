import json

# ---- CONFIG ----
theme_extraction_file = "theme_extraction_result1.json"
theme_difficulty_file = "theme_difficulty_result1.json"
bayes_cognitive_load_file = "bayes_cognitive_load_output.json"
output_file = "simplification_strategy.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def determine_simplification_type(
    theme_id, theme_difficulty, theme_cognitive_load, familiarity, chunk_features
):
    """Decide whether a sentence needs conceptual, linguistic, or both kinds of simplification."""
    conceptual_threshold = 0.4  # high intrinsic difficulty + low familiarity = conceptual simplification
    linguistic_threshold = 0.6  # long, deep, or complex sentences = linguistic simplification

    conceptual = (
        (theme_difficulty > conceptual_threshold)
        or (familiarity < 0.4)
        or (theme_cognitive_load > 0.35)
    )

    # Heuristic linguistic difficulty indicators
    ling_features = [
        chunk_features.get("mean_sentence_length", 0),
        chunk_features.get("parse_depth", 0),
        chunk_features.get("avg_dep_length", 0),
        chunk_features.get("surprisal", 0),
    ]
    linguistic_score = sum(ling_features) / len(ling_features)
    linguistic = linguistic_score > linguistic_threshold

    if conceptual and linguistic:
        return "both"
    elif conceptual:
        return "conceptual"
    elif linguistic:
        return "linguistic"
    else:
        return "none"


def generate_simplification_strategy():
    theme_extraction = load_json(theme_extraction_file)
    theme_difficulty = load_json(theme_difficulty_file)
    bayes_cog = load_json(bayes_cognitive_load_file)

    output = {"sentences": []}

    for chunk in theme_extraction["chunk_assignments"]:
        text = chunk["text"]
        theme_id = chunk["assignments"][0][0]
        theme_key = f"theme_{theme_id}"

        # Fetch associated metrics
        theme_diff = theme_difficulty["themes"][theme_key]["difficulty_score"]
        cog_theme = bayes_cog["themes"][theme_key]
        theme_cog_load = cog_theme["predicted_CL_after_update"]
        familiarity = cog_theme["familiarity"]

        # Get linguistic stats for this chunk
        chunk_features = theme_difficulty["chunks"][chunk["chunk_index"]]

        simplification_type = determine_simplification_type(
            theme_id, theme_diff, theme_cog_load, familiarity, chunk_features
        )

        output["sentences"].append(
            {
                "chunk_index": chunk["chunk_index"],
                "text": text,
                "theme_id": theme_id,
                "simplification_type": simplification_type,
                "metrics": {
                    "theme_difficulty": theme_diff,
                    "theme_cognitive_load": theme_cog_load,
                    "familiarity": familiarity,
                    "linguistic_score": round(
                        sum(
                            [
                                chunk_features["mean_sentence_length"],
                                chunk_features["parse_depth"],
                                chunk_features["avg_dep_length"],
                                chunk_features["surprisal"],
                            ]
                        )
                        / 4,
                        3,
                    ),
                },
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Simplification strategy saved to {output_file}")


if __name__ == "__main__":
    generate_simplification_strategy()
