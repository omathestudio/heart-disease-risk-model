"""Step 5b: gradient boosting, compared with the first two models.

Gradient boosting builds hundreds of small decision trees, each one fixing the
mistakes of the ones before. Unlike logistic regression it can learn curves
(risk rising faster after 60) and interactions (diabetes AND high blood
pressure together), which usually makes it the strongest model for tables.

scikit-learn's HistGradientBoostingClassifier is used: it's fast on large data
and needs no extra install.

All three models are scored on the VALIDATION set. The test set stays locked away.

Run from the project root:
    python src/train_boosting.py
"""

import time

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import PROJECT_ROOT
from data import load_data
from features import RANDOM_STATE, build_preprocessor, split_data
from train_baseline import evaluate

RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    X, y, weights = load_data()
    parts = split_data(X, y, weights)
    X_train, y_train, _ = parts["train"]
    X_val, y_val, _ = parts["val"]

    models = {
        "Dummy (always average)": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(
            class_weight="balanced", max_iter=1000),
        "Gradient boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            learning_rate=0.05,      # small steps: slower but more careful
            max_iter=500,            # up to 500 trees...
            early_stopping=True,     # ...but stop when it stops improving
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    for name, classifier in models.items():
        model = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", classifier),
        ])
        start = time.time()
        model.fit(X_train, y_train)
        seconds = time.time() - start
        row = evaluate(name, model, X_val, y_val)
        row["train_seconds"] = round(seconds, 1)
        results.append(row)
        print(f"  trained {name} in {seconds:.1f}s")

        if name == "Gradient boosting":
            print(f"  (early stopping kept {classifier.n_iter_} trees)")

    table = pd.DataFrame(results).set_index("model").round(3)
    print("\nValidation results:")
    print(table.to_string())
    print(f"\nFor PR-AUC, a random guess scores the heart disease rate: "
          f"{y_val.mean():.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "model_comparison.csv")
    print(f"Saved to {(RESULTS_DIR / 'model_comparison.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
