"""Step 4: split the data and build the preprocessing pipeline.

Everything that changes the data before the model sees it lives here, inside one
scikit-learn pipeline. That way exactly the same steps run when training, when
testing and when the app makes a prediction later.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

from config import (BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                    ORDINAL_FEATURES)

RANDOM_STATE = 42   # same number every time, so the split is repeatable


# --- 1. Train / validation / test split --------------------------------------

def split_data(X, y, weights):
    """Split into 70% train, 15% validation, 15% test.

    Stratified, so each part has the same heart disease rate (about 9%).
    - train:      the model learns from this
    - validation: used to compare models and tune settings
    - test:       locked away until the very end, used once for the final score
    """
    X_train, X_rest, y_train, y_rest, w_train, w_rest = train_test_split(
        X, y, weights, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
        X_rest, y_rest, w_rest, test_size=0.50, stratify=y_rest,
        random_state=RANDOM_STATE)
    return {
        "train": (X_train, y_train, w_train),
        "val": (X_val, y_val, w_val),
        "test": (X_test, y_test, w_test),
    }


# --- 2. Preprocessing --------------------------------------------------------

def yes_no_missing(X):
    """Turn 1 / 0 / blank into 'Yes' / 'No' / 'Missing'.

    The exploratory analysis showed blanks carry information (for example,
    people never checked for cholesterol have half the average heart disease
    rate), so 'Missing' is kept as its own group instead of being guessed.
    """
    return X.apply(lambda col: col.map({1: "Yes", 0: "No"}).fillna("Missing"))


def build_preprocessor():
    """Return a ColumnTransformer that turns the raw features into numbers.

    - BMI: fill blanks with the median, add a 'was blank' flag, then scale.
    - Yes/no: three groups (Yes / No / Missing), one-hot encoded.
    - Ordered categories (age, education, income): turned into 0, 1, 2 ...
      in their natural order, blanks filled with the median plus a
      'was blank' flag, then scaled.
    - Other categories (sex, smoking): blanks become 'Missing', one-hot encoded.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])

    binary = Pipeline([
        ("to_text", FunctionTransformer(yes_no_missing,
                                        feature_names_out="one-to-one")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ordinal = Pipeline([
        ("encode", OrdinalEncoder(
            categories=list(ORDINAL_FEATURES.values()),
            handle_unknown="use_encoded_value", unknown_value=np.nan,
            encoded_missing_value=np.nan)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("binary", binary, BINARY_FEATURES),
        ("ordinal", ordinal, list(ORDINAL_FEATURES)),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
