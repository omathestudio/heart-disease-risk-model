"""Step 6: are the predicted risks believable, and where should the cut-off be?

A risk model has two jobs:
  1. Ranking: put higher-risk people above lower-risk people (ROC-AUC measures this).
  2. Calibration: when it says "20% risk", about 20% of those people should
     actually have heart disease (Brier score and the calibration chart measure this).

class_weight="balanced" helps ranking on rare outcomes but inflates every
predicted risk. This script compares the balanced model with an UNWEIGHTED
logistic regression, which should rank almost as well but give honest risks.
The cut-off for saying "high risk" is then chosen on purpose, instead of
using 50% by default.

Run from the project root:
    python src/calibrate.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline

from config import PROJECT_ROOT
from data import load_data
from explore import BLUE, INK, MUTED, ORANGE, save
from features import RANDOM_STATE, build_preprocessor, split_data

RESULTS_DIR = PROJECT_ROOT / "results"
AQUA = "#1baf7a"


def make_models():
    return {
        "Logistic regression (balanced)": (LogisticRegression(
            class_weight="balanced", max_iter=1000), BLUE),
        "Logistic regression (unweighted)": (LogisticRegression(
            max_iter=1000), AQUA),
        "Gradient boosting (balanced)": (HistGradientBoostingClassifier(
            class_weight="balanced", learning_rate=0.05, max_iter=500,
            early_stopping=True, n_iter_no_change=20,
            random_state=RANDOM_STATE), ORANGE),
    }


def threshold_table(y, prob):
    """Recall, precision and share flagged at a range of cut-offs."""
    rows = []
    for cut in [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        flagged = prob >= cut
        caught = (flagged & (y == 1)).sum()
        rows.append({
            "cut_off": cut,
            "recall": caught / (y == 1).sum(),
            "precision": caught / flagged.sum() if flagged.sum() else 0,
            "pct_flagged": flagged.mean(),
        })
    return pd.DataFrame(rows).round(3)


def main():
    X, y, weights = load_data()
    parts = split_data(X, y, weights)
    X_train, y_train, _ = parts["train"]
    X_val, y_val, _ = parts["val"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.62, 0.66, "Perfectly calibrated", color=MUTED, fontsize=9,
            rotation=40)

    results = []
    probs = {}
    for name, (classifier, colour) in make_models().items():
        model = Pipeline([("preprocess", build_preprocessor()),
                          ("model", classifier)])
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        probs[name] = prob

        results.append({
            "model": name,
            "roc_auc": roc_auc_score(y_val, prob),
            "pr_auc": average_precision_score(y_val, prob),
            "brier": brier_score_loss(y_val, prob),
            "mean_predicted_risk": prob.mean(),
            "actual_rate": y_val.mean(),
        })

        observed, predicted = calibration_curve(
            y_val, prob, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", markersize=5, linewidth=2,
                color=colour, label=name)
        print(f"  done: {name}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted risk (average in each group of 10%)")
    ax.set_ylabel("Actual heart disease rate in that group")
    ax.set_title("Calibration: does predicted risk match reality?")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "calibration.png")

    table = pd.DataFrame(results).set_index("model").round(3)
    print("\nValidation results (lower Brier score = more honest risks):")
    print(table.to_string())

    chosen = "Logistic regression (unweighted)"
    thresholds = threshold_table(y_val.to_numpy(), probs[chosen])
    print(f"\nCut-off options for: {chosen}")
    print("(recall = share of heart disease cases caught; "
          "precision = share of flagged people who have it)")
    print(thresholds.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "calibration.csv")
    thresholds.to_csv(RESULTS_DIR / "thresholds.csv", index=False)
    print("\nSaved results/calibration.csv and results/thresholds.csv")


if __name__ == "__main__":
    main()
