"""Project settings: where the data lives, and which columns the model uses.

Every column from the Heart Health Pipeline's `respondents` table is listed here,
either as a feature or in EXCLUDED with the reason it's left out.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The model reads the SQLite database made by the Heart Health Pipeline.
# By default it expects the two repos to sit side by side:
#   projects/
#   ├── heart-health-pipeline/data/heart.db
#   └── heart-disease-risk-model/
# Set the HEART_DB environment variable to point somewhere else.
DB_PATH = Path(os.environ.get(
    "HEART_DB",
    PROJECT_ROOT.parent / "heart-health-pipeline" / "data" / "heart.db",
))
TABLE_NAME = "respondents"


# --- Target ---
# 1 = ever told they had coronary heart disease or a heart attack (CDC's _MICHD)
TARGET = "heart_disease"

# --- Survey weight ---
# Not a feature. Used later so evaluation describes the US adult population.
WEIGHT = "survey_weight"


# --- Features ---

# Numbers
NUMERIC_FEATURES = ["bmi"]

# Yes / no (1 / 0, with some blanks)
BINARY_FEATURES = [
    "high_bp",
    "high_cholesterol",   # blank for people never checked: kept as its own group
    "diabetes",
    "inactive",
    "heavy_drinker",
]

# Categories that have a natural order (low -> high)
ORDINAL_FEATURES = {
    "age_group": [
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
        "55-59", "60-64", "65-69", "70-74", "75-79", "80+",
    ],
    "education": [
        "No high school diploma", "High school graduate",
        "Some college", "College graduate",
    ],
    "income": [   # about 1 in 5 blank: kept as its own group
        "Under $15k", "$15k-$25k", "$25k-$35k", "$35k-$50k",
        "$50k-$100k", "$100k-$200k", "$200k+",
    ],
}

# Categories with no natural order
CATEGORICAL_FEATURES = ["sex", "smoking_status"]


# --- Left out on purpose ---
EXCLUDED = {
    "heart_attack":   "Leakage: part of the definition of the target.",
    "coronary_hd":    "Leakage: part of the definition of the target.",
    "general_health": "Reverse causation: having heart disease makes people rate "
                      "their health worse, so it describes the outcome rather than "
                      "predicting it. Tested later as an experiment.",
    "stroke":         "Another outcome of the same disease process, not a risk "
                      "factor. Leaving it out keeps the model about risk factors.",
    "state_fips":     "Left out of version 1 to keep the model simple and usable "
                      "outside the US. Can be tested later.",
}


def all_features():
    """Every feature column, in one list."""
    return (NUMERIC_FEATURES + BINARY_FEATURES
            + list(ORDINAL_FEATURES) + CATEGORICAL_FEATURES)
