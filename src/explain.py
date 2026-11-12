"""Step 7: what drives the model's predictions?

Logistic regression gives each feature a coefficient. Turned into an odds
ratio (OR), it says how much that feature multiplies the odds of heart
disease, with everything else held the same:
  OR = 1    no effect
  OR = 2    doubles the odds
  OR = 0.5  halves the odds

Because every other feature is held the same, these are ADJUSTED effects.
For example, the diabetes OR is the effect of diabetes among people of the
same age, sex, BMI and so on. That's different from the simple with/without
rates in the Heart Health Pipeline.

Run from the project root:
    python src/explain.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import PROJECT_ROOT
from data import load_data
from explore import BLUE, INK, MUTED, save
from features import build_preprocessor, split_data

RESULTS_DIR = PROJECT_ROOT / "results"


def train_final_model(X_train, y_train):
    """The model chosen in Step 6: unweighted logistic regression."""
    model = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    return model.fit(X_train, y_train)


def odds_ratios(model):
    """Turn the model's coefficients into readable odds ratios."""
    preprocess = model.named_steps["preprocess"]
    names = preprocess.get_feature_names_out()
    coef = dict(zip(names, model.named_steps["model"].coef_[0]))

    # Numbers and ordered categories were scaled, so undo the scaling to get
    # the effect of one real step (e.g. one age band, one BMI point).
    scale = {}
    for group in ["numeric", "ordinal"]:
        pipe = preprocess.named_transformers_[group]
        out_names = pipe.get_feature_names_out()
        for name, s in zip(out_names, pipe.named_steps["scale"].scale_):
            scale[f"{group}__{name}"] = s

    def per_step(name, steps=1):
        return np.exp(steps * coef[name] / scale[name])

    def vs(level, reference):
        # One-hot levels are compared with a reference level
        return np.exp(coef[level] - coef[reference])

    rows = [
        ("Age: each 5-year band older", per_step("ordinal__age_group")),
        ("BMI: each 5 points higher", per_step("numeric__bmi", steps=5)),
        ("Education: each level higher", per_step("ordinal__education")),
        ("Income: each band higher", per_step("ordinal__income")),
        ("Sex: male vs female",
         vs("categorical__sex_Male", "categorical__sex_Female")),
        ("Smoking: every day vs never",
         vs("categorical__smoking_status_Current (every day)",
            "categorical__smoking_status_Never")),
        ("Smoking: some days vs never",
         vs("categorical__smoking_status_Current (some days)",
            "categorical__smoking_status_Never")),
        ("Smoking: former vs never",
         vs("categorical__smoking_status_Former",
            "categorical__smoking_status_Never")),
        ("Diabetes", vs("binary__diabetes_Yes", "binary__diabetes_No")),
        ("High blood pressure", vs("binary__high_bp_Yes", "binary__high_bp_No")),
        ("High cholesterol",
         vs("binary__high_cholesterol_Yes", "binary__high_cholesterol_No")),
        ("Cholesterol never checked vs normal",
         vs("binary__high_cholesterol_Missing", "binary__high_cholesterol_No")),
        ("No exercise in past 30 days",
         vs("binary__inactive_Yes", "binary__inactive_No")),
        ("Heavy drinker", vs("binary__heavy_drinker_Yes",
                             "binary__heavy_drinker_No")),
    ]
    return (pd.DataFrame(rows, columns=["factor", "odds_ratio"])
            .sort_values("odds_ratio").reset_index(drop=True))


def chart_odds_ratios(table):
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(table) + 1.5))
    y = np.arange(len(table))
    ax.hlines(y, 1, table["odds_ratio"], color=BLUE, linewidth=2)
    ax.plot(table["odds_ratio"], y, "o", color=BLUE, markersize=8)
    ax.axvline(1, color=INK, linewidth=1)

    for i, value in enumerate(table["odds_ratio"]):
        offset = 1.06 if value >= 1 else 1 / 1.06
        ax.text(value * offset, i, f"{value:.2f}", va="center",
                ha="left" if value >= 1 else "right", color=INK, fontsize=9)

    ax.set_xscale("log")
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    low, high = table["odds_ratio"].min(), table["odds_ratio"].max()
    ax.set_xlim(min(0.4, low / 1.5), max(3, high * 1.5))
    ax.set_yticks(y, table["factor"])
    ax.tick_params(axis="y", colors=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Odds ratio (log scale). Right of 1 = higher odds of heart disease")
    ax.set_title("What raises the odds of heart disease? (adjusted for all other factors)")
    save(fig, "odds_ratios.png",
         note="Source: CDC BRFSS 2025. Unweighted logistic regression, training set.")


def main():
    X, y, weights = load_data()
    parts = split_data(X, y, weights)
    X_train, y_train, _ = parts["train"]

    model = train_final_model(X_train, y_train)
    table = odds_ratios(model)
    chart_odds_ratios(table)

    print("\nOdds ratios, strongest first:")
    for _, row in table.sort_values("odds_ratio", ascending=False).iterrows():
        print(f"  {row['factor']:<38} {row['odds_ratio']:5.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    table.round(3).to_csv(RESULTS_DIR / "odds_ratios.csv", index=False)
    print("\nSaved results/odds_ratios.csv")


if __name__ == "__main__":
    main()
