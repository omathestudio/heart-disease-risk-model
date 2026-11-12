"""Step 3: exploratory analysis for modelling.

The pipeline already showed who has heart disease. These charts answer
questions the model needs answered:
  1. Do blank answers carry information? (decides how we fill gaps)
  2. How does BMI differ between people with and without heart disease?
  3. How does the heart disease rate change across each category,
     including "Missing"? (checks the ordered features really are ordered)

Rates here are NOT weighted: they describe the sample the model learns from.

Run from the project root:
    python src/explore.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (BINARY_FEATURES, CATEGORICAL_FEATURES, ORDINAL_FEATURES,
                    PROJECT_ROOT, all_features)
from data import load_data

CHARTS_DIR = PROJECT_ROOT / "charts"

# --- Same colours and styling as the Heart Health Pipeline -------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BACKGROUND = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
    "text.parse_math": False,   # show "$15k" as text, not maths
})

SOURCE_NOTE = "Source: CDC BRFSS 2025. Unweighted rates (the sample the model learns from)."


def save(fig, name):
    """Add the source note and save the chart as a PNG."""
    fig.text(0.01, -0.04, SOURCE_NOTE, fontsize=8, color=MUTED)
    CHARTS_DIR.mkdir(exist_ok=True)
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(PROJECT_ROOT)}")


def as_labels(series):
    """Turn a column into readable labels, with blanks shown as 'Missing'."""
    if series.name in BINARY_FEATURES:
        series = series.map({1: "Yes", 0: "No"})
    return series.fillna("Missing").astype(str)


# --- Chart 1: does a blank answer predict heart disease? ---------------------

def chart_missing(X, y):
    overall = 100 * y.mean()
    rows = []
    for column in all_features():
        blank = X[column].isna()
        if blank.sum() >= 100:   # too few blanks to say anything
            rows.append((column, 100 * y[blank].mean(), blank.sum()))
    table = pd.DataFrame(rows, columns=["feature", "rate", "n"]).sort_values("rate")

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(table) + 1.2))
    ax.barh(table["feature"], table["rate"], color=BLUE, height=0.6)
    ax.axvline(overall, color=INK, linewidth=1, linestyle="--")
    ax.text(overall, -0.75, f" Everyone: {overall:.1f}%",
            color=INK, fontsize=9, va="center")
    for i, (rate, n) in enumerate(zip(table["rate"], table["n"])):
        ax.text(rate + 0.3, i, f"{rate:.1f}%  (n={n:,})", va="center",
                color=INK, fontsize=9,
                bbox=dict(facecolor=BACKGROUND, edgecolor="none", pad=1))
    ax.set_title("Heart disease rate among people who left each question blank")
    ax.set_xlabel("Heart disease (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, table["rate"].max() * 1.35)
    ax.set_ylim(-1.2, len(table) - 0.5)
    save(fig, "eda_missing_values.png")
    return table


# --- Chart 2: BMI for people with and without heart disease ------------------

def chart_bmi(X, y):
    bmi = X["bmi"]
    bins = np.arange(12, 60.5, 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for value, colour, label in [(0, BLUE, "No heart disease"),
                                 (1, ORANGE, "Heart disease")]:
        data = bmi[(y == value) & bmi.notna()].clip(upper=60)
        ax.hist(data, bins=bins, density=True, histtype="step",
                linewidth=2, color=colour, label=f"{label} (median {data.median():.1f})")
    ax.set_title("BMI distribution, by heart disease status")
    ax.set_xlabel("BMI (values above 60 grouped at 60)")
    ax.set_ylabel("Share of group")
    ax.legend(frameon=False)
    save(fig, "eda_bmi.png")


# --- Chart 3: heart disease rate across each category ------------------------

def chart_categories(X, y):
    columns = list(ORDINAL_FEATURES) + CATEGORICAL_FEATURES + BINARY_FEATURES
    n_cols = 2
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.2 * n_rows))
    overall = 100 * y.mean()

    for ax, column in zip(axes.flat, columns):
        labels = as_labels(X[column])
        rates = (100 * y.groupby(labels).mean())
        # Keep the natural order for ordered features, with Missing last
        if column in ORDINAL_FEATURES:
            order = [c for c in ORDINAL_FEATURES[column] if c in rates.index]
        else:
            order = sorted(c for c in rates.index if c != "Missing")
        if "Missing" in rates.index:
            order.append("Missing")
        rates = rates[order]

        colours = [MUTED if label == "Missing" else BLUE for label in rates.index]
        ax.barh(rates.index, rates.values, color=colours, height=0.65)
        ax.axvline(overall, color=INK, linewidth=0.8, linestyle="--")
        ax.invert_yaxis()
        ax.set_title(column)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)

    for ax in list(axes.flat)[len(columns):]:
        ax.axis("off")

    fig.suptitle("Heart disease rate (%) by category. Dashed line = everyone. "
                 "Grey = left blank.", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save(fig, "eda_by_category.png")


def main():
    X, y, _ = load_data()
    print("Making charts:")
    missing = chart_missing(X, y)
    chart_bmi(X, y)
    chart_categories(X, y)

    print("\nHeart disease rate when the answer is blank:")
    print(f"  (everyone: {100 * y.mean():.1f}%)")
    for _, row in missing.sort_values("rate", ascending=False).iterrows():
        print(f"  {row['feature']:<18} {row['rate']:5.1f}%   n={row['n']:,}")


if __name__ == "__main__":
    main()
