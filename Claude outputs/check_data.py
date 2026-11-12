"""Step 1 check: how balanced is the target, and how much is missing?

Run from the project root:
    python src/check_data.py
"""

from config import EXCLUDED
from data import load_data


def main():
    X, y, weights = load_data()

    print(f"\nRows for modelling: {len(y):,}")
    print(f"Features: {X.shape[1]}")

    # 1. Class balance: the most important number for the rest of the project
    cases = y.sum()
    raw_rate = 100 * y.mean()
    weighted = 100 * (y * weights).sum() / weights.sum()
    print("\n--- Target balance ---")
    print(f"Heart disease cases: {cases:,} of {len(y):,}")
    print(f"Rate in the sample:      {raw_rate:.1f}%")
    print(f"Rate weighted to the US: {weighted:.1f}%")
    print(f"A model that always says 'no' would be {100 - raw_rate:.1f}% accurate "
          "and completely useless.")

    # 2. Missing values per feature
    print("\n--- Missing values per feature ---")
    missing = X.isna().mean().mul(100).sort_values(ascending=False)
    for column, pct in missing.items():
        print(f"{column:<18} {pct:5.1f}%")

    # 3. Reminder of what's left out and why
    print("\n--- Excluded columns ---")
    for column, reason in EXCLUDED.items():
        print(f"{column:<18} {reason}")


if __name__ == "__main__":
    main()
