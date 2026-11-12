"""Load the clean data from the Heart Health Pipeline and get it ready for modelling."""

import sqlite3

import pandas as pd

from config import DB_PATH, TABLE_NAME, TARGET, WEIGHT, all_features


def load_data(db_path=DB_PATH):
    """Read the respondents table and return (X, y, weights).

    X: the feature columns
    y: the target (1 = heart disease, 0 = none)
    weights: the survey weight for each row

    Rows where the target is blank are dropped, because the model
    can't learn from people whose answer we don't know.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"No database at {db_path}. Run the Heart Health Pipeline first, "
            "or set the HEART_DB environment variable to the heart.db path."
        )

    columns = all_features() + [TARGET, WEIGHT]
    query = f"SELECT {', '.join(columns)} FROM {TABLE_NAME}"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(query, conn)

    rows_before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"Dropped {rows_before - len(df):,} rows with no heart disease answer")

    X = df[all_features()]
    y = df[TARGET].astype(int)
    weights = df[WEIGHT]
    return X, y, weights
