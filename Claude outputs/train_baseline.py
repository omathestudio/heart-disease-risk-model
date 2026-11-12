"""Step 5a: the first two models.

1. Dummy model: always predicts the average rate. Any real model must beat it.
2. Logistic regression: simple, explainable, and the standard starting point
   for clinical risk scores. class_weight="balanced" makes the rare heart
   disease cases count as much as the common no-disease cases.

Both are scored on the VALIDATION set. The test set stays locked away.

Run from the project root:
    python src/train_baseline.py
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from config import PROJECT_ROOT
from data import load_data
from features import build_preprocessor, split_data

RESULTS_DIR = PROJECT_ROOT / "results"


def evaluate(name, model, X, y):
    """Score a fitted model. Returns one row of results."""
    prob = model.predict_proba(X)[:, 1]     # predicted risk, 0 to 1
    pred = (prob >= 0.5).astype(int)        # yes/no at a 50% cut-off
    return {
        "model": name,
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
    }


def main():
    X, y, weights = load_data()
    parts = split_data(X, y, weights)
    X_train, y_train, _ = parts["train"]
    X_val, y_val, _ = parts["val"]

    print(f"Train: {len(y_train):,} rows   Validation: {len(y_val):,} rows   "
          f"Test: {len(parts['test'][1]):,} rows (locked away)")

    models = {
        "Dummy (always average)": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(
            class_weight="balanced", max_iter=1000),
    }

    results = []
    for name, classifier in models.items():
        model = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", classifier),
        ])
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_val, y_val))

    table = pd.DataFrame(results).set_index("model").round(3)
    print("\nValidation results:")
    print(table.to_string())
    print(f"\nFor PR-AUC, a random guess scores the heart disease rate: "
          f"{y_val.mean():.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "baseline_metrics.csv")
    print(f"Saved to {(RESULTS_DIR / 'baseline_metrics.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
