"""Step 8: the final model, scored ONCE on the locked test set, plus a fairness check.

1. Retrain the chosen model (unweighted logistic regression) on train +
   validation together. Every choice has been made, so the validation rows
   can now be used for learning.
2. Score it on the test set, which no decision has touched. This is the
   honest estimate of how the model would do on new people.
3. Fairness check: does the model work equally well for different groups?
   For each group: does it rank well (ROC-AUC), does it catch cases at the
   10% cut-off (recall), and does its average predicted risk match the
   group's real rate (calibration)?
4. Save the model so the app can use it later.

Run from the project root:
    python src/evaluate_final.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)

from config import PROJECT_ROOT, THRESHOLD
from data import load_data
from explain import train_final_model
from features import split_data

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

AGE_BANDS = {
    "18-44": ["18-24", "25-29", "30-34", "35-39", "40-44"],
    "45-64": ["45-49", "50-54", "55-59", "60-64"],
    "65+": ["65-69", "70-74", "75-79", "80+"],
}
INCOME_BANDS = {
    "Under $35k": ["Under $15k", "$15k-$25k", "$25k-$35k"],
    "$35k-$100k": ["$35k-$50k", "$50k-$100k"],
    "$100k+": ["$100k-$200k", "$200k+"],
}


def scores(y, prob):
    """The headline numbers for one set of people."""
    pred = (prob >= THRESHOLD).astype(int)
    both_classes = y.nunique() == 2
    return {
        "people": len(y),
        "actual_rate": y.mean(),
        "mean_predicted_risk": prob.mean(),
        "roc_auc": roc_auc_score(y, prob) if both_classes else np.nan,
        "pr_auc": average_precision_score(y, prob) if both_classes else np.nan,
        "brier": brier_score_loss(y, prob),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "pct_flagged": pred.mean(),
    }


def group_labels(X):
    """Which groups to check the model across."""
    def band(column, bands):
        lookup = {value: name for name, values in bands.items() for value in values}
        return X[column].map(lookup).fillna("Missing")

    return {
        "Sex": X["sex"].fillna("Missing"),
        "Age": band("age_group", AGE_BANDS),
        "Income": band("income", INCOME_BANDS),
        "Education": X["education"].fillna("Missing"),
    }


def main():
    X, y, weights = load_data()
    parts = split_data(X, y, weights)
    X_train, y_train, _ = parts["train"]
    X_val, y_val, _ = parts["val"]
    X_test, y_test, _ = parts["test"]

    # 1. Retrain on train + validation
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    model = train_final_model(X_full, y_full)
    print(f"Final model trained on {len(y_full):,} rows")

    # 2. Score once on the test set
    prob = model.predict_proba(X_test)[:, 1]
    test = pd.Series(scores(y_test, prob)).round(3)
    print(f"\n--- Test set results ({len(y_test):,} people, cut-off {THRESHOLD:.0%}) ---")
    print(test.to_string())

    # 3. Fairness check
    rows = []
    y_arr = y_test.reset_index(drop=True)
    prob_s = pd.Series(prob)
    for group_name, labels in group_labels(X_test.reset_index(drop=True)).items():
        for label in sorted(labels.unique()):
            mask = labels == label
            if mask.sum() < 50:   # too few people to judge
                continue
            rows.append({"group": group_name, "value": label,
                         **scores(y_arr[mask], prob_s[mask].to_numpy())})
    fairness = pd.DataFrame(rows).round(3)

    print("\n--- Fairness check (test set) ---")
    print(fairness[["group", "value", "people", "actual_rate",
                    "mean_predicted_risk", "roc_auc", "recall",
                    "precision"]].to_string(index=False))

    # 4. Save everything
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    test.to_csv(RESULTS_DIR / "test_metrics.csv", header=["value"])
    fairness.to_csv(RESULTS_DIR / "fairness.csv", index=False)
    joblib.dump(model, MODELS_DIR / "heart_risk_model.joblib")
    print("\nSaved results/test_metrics.csv, results/fairness.csv "
          "and models/heart_risk_model.joblib")


if __name__ == "__main__":
    main()
