"""
bayesian_cognitive_load.py

Bayesian weight learning + nonlinear cognitive-load estimator.

Files expected (same folder):
- theme_extraction.json              (level 1)  -- for occupancy + representative spans
- theme_difficulty_result1.json      (level 2)  -- intrinsic difficulty per theme
- user_familiarity_result1.json      (level 3)  -- familiarity per theme

Outputs:
- bayes_weights.json                 (posterior mean/cov saved persistently)
- bayes_weight_history.json          (history of posterior means)
- bayes_cognitive_load_output.json   (per-theme predictions + doc CL)
- weight_evolution.png               (plot of weight evolution)
"""

import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config: tweak as needed
# -----------------------------
THEME_EXTRACTION_FILE = "theme_extraction_result1.json"
INTRINSIC_FILE = "theme_difficulty_result1.json"
FAMILIARITY_FILE = "user_familiarity_result1.json"

POSTERIOR_FILE = "bayes_weights.json"
HISTORY_FILE = "bayes_weight_history.json"
OUTPUT_FILE = "bayes_cognitive_load_output.json"
WEIGHT_EVOL_PNG = "weight_evolution.png"

# Bayesian prior hyperparameters
# prior mean for linear coefficients (can be any small values)
PRIOR_MEAN = np.array([0.5, 0.3, 0.2])  # intrinsic, unfamiliarity, discourse (initial guess)
PRIOR_COV = np.diag([1.0, 1.0, 1.0])    # initial covariance (uncertainty)
LIKELIHOOD_VAR = 0.02  # sigma^2 for observation noise (tuneable)

# Nonlinear exponents for basis features (p, q, r)
EXP_INTRINSIC = 1.8   # intrinsic^p
EXP_UNFAM = 1.6       # (1 - familiarity)^q
EXP_DISCOURSE = 1.4   # discourse^r

# Sigmoid scaling (optional temperature)
SIGMOID_SCALE = 8.0   # larger -> sharper transition; set to ~5-10 typically

# Misc
MAX_FEEDBACKS_PER_SESSION = None  # None => no explicit cap; you can set an integer
AUTO_SAVE_AFTER_UPDATE = True


# -----------------------------
# Utilities
# -----------------------------
def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# -----------------------------
# Bayesian update functions
# -----------------------------
def ensure_posterior_file():
    """If no saved posterior, initialize from prior."""
    if not os.path.exists(POSTERIOR_FILE):
        post = {
            "mu": PRIOR_MEAN.tolist(),
            "Sigma": PRIOR_COV.tolist()
        }
        save_json(POSTERIOR_FILE, post)
        save_json(HISTORY_FILE, [PRIOR_MEAN.tolist()])
        return post
    return load_json(POSTERIOR_FILE)


def bayes_update(mu_prior, Sigma_prior, x_vec, y_obs, sigma2=LIKELIHOOD_VAR):
    """
    Single-observation Bayesian linear regression update.
    Prior: mu_prior (d,), Sigma_prior (dxd).
    Likelihood: y | x ~ N(x^T w, sigma2)
    Posterior:
      Sigma_post = (Sigma_prior^{-1} + (1/sigma2) x x^T)^{-1}
      mu_post = Sigma_post * (Sigma_prior^{-1} mu_prior + (1/sigma2) x y)
    """
    mu_prior = np.asarray(mu_prior, dtype=float)
    Sigma_prior = np.asarray(Sigma_prior, dtype=float)
    x = np.asarray(x_vec, dtype=float).reshape(-1, 1)
    inv_Sigma_prior = np.linalg.inv(Sigma_prior)
    # compute posterior covariance
    A = inv_Sigma_prior + (1.0 / sigma2) * (x @ x.T)
    Sigma_post = np.linalg.inv(A)
    b = inv_Sigma_prior @ mu_prior.reshape(-1, 1) + (1.0 / sigma2) * x * y_obs
    mu_post = (Sigma_post @ b).reshape(-1)
    return mu_post, Sigma_post


# -----------------------------
# Feature transform (nonlinear basis)
# -----------------------------
def transform_features(intrinsic, familiarity, discourse):
    """
    Map raw features to basis used by Bayesian linear model.
    - intrinsic: in [0,1] (higher = more difficult)
    - familiarity: in [0,1] (higher = more familiar)
    - discourse: in [0,1]
    Returns x = [intrinsic^p, (1 - familiarity)^q, discourse^r]
    """
    x1 = max(0.0, float(intrinsic)) ** EXP_INTRINSIC
    x2 = max(0.0, 1.0 - float(familiarity)) ** EXP_UNFAM
    x3 = max(0.0, float(discourse)) ** EXP_DISCOURSE
    return np.array([x1, x2, x3], dtype=float)


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # Load input files
    theme_ex = load_json(THEME_EXTRACTION_FILE)
    intrinsic_data = load_json(INTRINSIC_FILE)
    fam_data = load_json(FAMILIARITY_FILE)

    if theme_ex is None:
        print(f"Error: missing {THEME_EXTRACTION_FILE}. This file provides occupancy and representative spans.")
        return
    if intrinsic_data is None:
        print(f"Error: missing {INTRINSIC_FILE}.")
        return
    if fam_data is None:
        print(f"Error: missing {FAMILIARITY_FILE}.")
        return

    # Ensure posterior saved
    posterior = ensure_posterior_file()
    mu = np.array(posterior["mu"], dtype=float)
    Sigma = np.array(posterior["Sigma"], dtype=float)

    # we'll record weight (mu) history
    weight_history = load_json(HISTORY_FILE, default=[mu.tolist()])

    print("\n=== Bayesian Cognitive Load Estimator ===\n")
    print("This session will show predicted cognitive load (0-1) per theme.")
    print("You may provide a single feedback value per theme (0-1), or press Enter to skip.\n")

    # build per-theme data source:
    # theme_ex["themes"] likely list with theme_id etc.
    # intrinsic_data["themes"] is dict keyed by theme name (theme_0)
    # fam_data["themes"] is dict keyed by theme name (theme_0)
    # occupancy from theme_ex: need occupancy_by_weight or count. We'll compute simple token-based occupancy per theme from chunk_assignments.

    # compute occupancy_by_token per theme using theme_ex chunk_assignments
    occupancy_tokens = {}
    total_tokens = 0
    for chunk in theme_ex.get("chunk_assignments", []):
        tokens = chunk.get("tokens", 0)
        assignments = chunk.get("assignments", [])
        # primary assignment is first element; if multiple, use similarity-weight sum
        if assignments:
            # join fractional: for each (theme_id, sim) add tokens*sim
            for aid, sim in assignments:
                theme_key = f"theme_{aid}"
                occupancy_tokens[theme_key] = occupancy_tokens.get(theme_key, 0.0) + tokens * float(sim)
        else:
            # skip
            pass
        total_tokens += tokens

    # fallback: if occupancy_tokens empty, compute equal occupancy across intrinsic themes
    if not occupancy_tokens:
        for tk in intrinsic_data.get("themes", {}).keys():
            occupancy_tokens[tk] = 1.0
        total_tokens = sum(occupancy_tokens.values())

    # normalize occupancy to [0,1] relative proportions
    occupancy = {k: (v / total_tokens if total_tokens > 0 else 0.0) for k, v in occupancy_tokens.items()}

    # Prepare results container
    results = {}
    feedback_count = 0

    # Iterate themes from intrinsic_data (dict map)
    for theme_key, diff_obj in intrinsic_data.get("themes", {}).items():
        intrinsic_score = float(diff_obj.get("difficulty_score", 0.0))
        fam_obj = fam_data.get("themes", {}).get(theme_key, {})
        familiarity = float(fam_obj.get("familiarity_score", 0.5))
        title = fam_obj.get("title", theme_key)
        examples = fam_obj.get("examples", [])

        # discourse proxy: use average length of examples if available else small default
        if examples:
            avg_len = np.mean([len(s.split()) for s in examples])
            discourse = min(avg_len / 40.0, 1.0)
        else:
            discourse = 0.2

        # transform features
        x = transform_features(intrinsic_score, familiarity, discourse)

        # linear predictor using posterior mean
        linear_pred = float(np.dot(mu, x))
        # apply scaled sigmoid for final CL
        z = SIGMOID_SCALE * linear_pred
        CL_pred = float(1.0 / (1.0 + math.exp(-z)))

        # Show to the user
        print(f"\nTheme: {theme_key} — {title}")
        if examples:
            print("  Example(s):")
            for i, ex in enumerate(examples[:2], 1):
                print(f"    {i}. {ex[:200]}")
        print(f"  Intrinsic difficulty: {intrinsic_score:.3f}")
        print(f"  Familiarity: {familiarity:.3f}")
        print(f"  Discourse proxy: {discourse:.3f}")
        print(f"  Occupancy share: {occupancy.get(theme_key, 0.0):.3f}")
        print(f"  Predicted cognitive load (CL): {CL_pred:.3f}")

        # Ask for feedback: single numeric in [0,1] OR blank to skip
        raw = input("  Enter your perceived difficulty for this theme (0-1), or press Enter to skip: ").strip()
        if raw == "":
            print("  → Skipped feedback for this theme.")
            results[theme_key] = {
                "predicted_CL": CL_pred,
                "linear_pred": linear_pred,
                "features": x.tolist(),
                "familiarity": familiarity,
                "intrinsic": intrinsic_score,
                "discourse": discourse,
                "occupancy": occupancy.get(theme_key, 0.0),
                "feedback": None
            }
            continue

        try:
            y = float(raw)
            if not (0.0 <= y <= 1.0):
                raise ValueError()
        except ValueError:
            print("  Invalid input; skipping feedback for this theme.")
            y = None

        if y is None:
            results[theme_key] = {
                "predicted_CL": CL_pred,
                "linear_pred": linear_pred,
                "features": x.tolist(),
                "familiarity": familiarity,
                "intrinsic": intrinsic_score,
                "discourse": discourse,
                "occupancy": occupancy.get(theme_key, 0.0),
                "feedback": None
            }
            continue

        # We have feedback y in [0,1]. We'll update Bayesian linear model.
        # Note: the linear observation model predicts y_linear ≈ linear_pred. Because CL_pred = sigmoid(SIGMOID_SCALE * linear_pred),
        # observation y is in [0,1] as well. We use y directly as target for linear model optimization.
        # A better approach would invert sigmoid: target_linear ≈ (1/SIGMOID_SCALE) * logit(y) but logit(y) may be unstable if y exactly 0 or 1.
        # We'll use a numerically stable target: t = clamp(logit(y) / SIGMOID_SCALE).
        eps = 1e-6
        y_clip = min(max(y, eps), 1 - eps)
        target_linear = (1.0 / SIGMOID_SCALE) * math.log(y_clip / (1.0 - y_clip))

        # Perform Bayesian update on the linear model with observation target_linear
        mu, Sigma = bayes_update(mu, Sigma, x, target_linear, sigma2=LIKELIHOOD_VAR)
        feedback_count += 1

        # record new mu in history
        weight_history.append(mu.tolist())

        # recompute CL_pred with updated mu for user info
        linear_pred_new = float(np.dot(mu, x))
        CL_pred_new = float(1.0 / (1.0 + math.exp(-SIGMOID_SCALE * linear_pred_new)))
        print(f"  ✅ Updated posterior mean (weights): {mu.round(4).tolist()}")
        print(f"  New predicted CL (after update): {CL_pred_new:.3f}")

        results[theme_key] = {
            "predicted_CL_before_update": CL_pred,
            "predicted_CL_after_update": CL_pred_new,
            "linear_pred_before": linear_pred,
            "linear_pred_after": linear_pred_new,
            "features": x.tolist(),
            "familiarity": familiarity,
            "intrinsic": intrinsic_score,
            "discourse": discourse,
            "occupancy": occupancy.get(theme_key, 0.0),
            "feedback": y
        }

        # persist posterior immediately (optional)
        if AUTO_SAVE_AFTER_UPDATE:
            post = {"mu": mu.tolist(), "Sigma": Sigma.tolist()}
            save_json(POSTERIOR_FILE, post)
            save_json(HISTORY_FILE, weight_history)

        # optionally limit number of feedbacks
        if MAX_FEEDBACKS_PER_SESSION is not None and feedback_count >= MAX_FEEDBACKS_PER_SESSION:
            print("Reached maximum feedback count for this session.")
            break

    # End per-theme loop: compute document-level CL via occupancy-weighted sum of predicted CLs
    # Use after-update CL if available else before-update
    doc_CL = 0.0
    for tk, info in results.items():
        occ = info.get("occupancy", occupancy.get(tk, 0.0))
        cl = info.get("predicted_CL_after_update", info.get("predicted_CL", 0.0))
        doc_CL += occ * cl

    # If occupancies don't sum to 1 (rare), normalize by sum
    occ_sum = sum(occupancy.values()) if occupancy else 0.0
    if occ_sum > 0:
        # occupancy already normalized earlier; this is robust check
        doc_CL = doc_CL / 1.0
    else:
        # fallback: simple mean
        doc_CL = float(np.mean([info.get("predicted_CL_after_update", info.get("predicted_CL", 0.0)) for info in results.values()] or [0.0]))

    # Save final posterior and results
    post_out = {"mu": mu.tolist(), "Sigma": Sigma.tolist()}
    save_json(POSTERIOR_FILE, post_out)
    save_json(HISTORY_FILE, weight_history)
    save_json(OUTPUT_FILE, {"document_cognitive_load": doc_CL, "themes": results})

    print(f"\n=== Done. Document-level cognitive load (occupancy-weighted) = {doc_CL:.3f} ===")
    print(f"Saved posterior -> {POSTERIOR_FILE}, history -> {HISTORY_FILE}, predictions -> {OUTPUT_FILE}")

    # Plot weight evolution if history bigger than 1
    if len(weight_history) >= 2:
        hist = np.array(weight_history)
        steps = np.arange(1, hist.shape[0] + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, hist[:, 0], label="w_intrinsic")
        plt.plot(steps, hist[:, 1], label="w_unfamiliarity")
        plt.plot(steps, hist[:, 2], label="w_discourse")
        plt.xlabel("Update step")
        plt.ylabel("Posterior mean (weight)")
        plt.title("Bayesian posterior mean evolution")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(WEIGHT_EVOL_PNG)
        print(f"Saved weight evolution plot to {WEIGHT_EVOL_PNG}")
        try:
            plt.show()
        except Exception:
            # in headless environments, plt.show may fail; ignore
            pass


if __name__ == "__main__":
    main()
