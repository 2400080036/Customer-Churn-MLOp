"""Loading, validating and splitting the dataset (used by training AND evaluation)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (BINARY, CATEGORICAL, DATA_PATH, FEATURES, ID_COLUMN,
                        NUMERIC, RANDOM_STATE, TARGET, TEST_SIZE)


def validate(df: pd.DataFrame) -> None:
    """Raise a clear error if the data does not match the expected schema."""
    missing_cols = [c for c in [ID_COLUMN, TARGET] + FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    if len(df) < 500:
        raise ValueError(f"Only {len(df)} rows. At least 500 are needed for a meaningful model.")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate customer_id values found.")
    if not set(df[TARGET].dropna().unique()) <= {"Yes", "No"} or df[TARGET].isna().any():
        raise ValueError("Column 'churn' must contain only Yes / No.")
    for col, allowed in CATEGORICAL.items():
        bad = set(df[col].dropna().unique()) - set(allowed)
        if bad:
            raise ValueError(f"Column '{col}' has unexpected values: {sorted(bad)}")
    if not df["senior_citizen"].dropna().isin([0, 1]).all():
        raise ValueError("Column 'senior_citizen' must be 0 or 1.")
    if not df["tenure"].dropna().between(0, 120).all():
        raise ValueError("Column 'tenure' must be between 0 and 120 months.")
    if not df["monthly_charges"].dropna().between(0, 1000).all():
        raise ValueError("Column 'monthly_charges' must be between 0 and 1000.")
    worst = df[FEATURES].isna().mean().max()
    if worst > 0.05:
        raise ValueError(f"More than 5% missing values in one column ({worst:.1%}).")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate(df)
    return df


def split(df: pd.DataFrame):
    """Stratified train/test split. The test part is never used for training or model selection."""
    X = df[FEATURES]
    y = (df[TARGET] == "Yes").astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


if __name__ == "__main__":
    data = load_data()
    print(f"OK: {len(data)} rows, churn rate {(data[TARGET] == 'Yes').mean():.1%}")
