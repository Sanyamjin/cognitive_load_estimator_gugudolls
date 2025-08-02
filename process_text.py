def analyze_text(text, profile):
    score = len(text) / 10
    label = "High" if score > 30 else "Low"
    return round(score, 2), label
