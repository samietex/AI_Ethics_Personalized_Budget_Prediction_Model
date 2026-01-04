from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "Budget (in dollars)",
    "Age",
    "Gender",
    "Education_Level",
    "With children?",
    "Recommended_Activity",
]


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and validate the schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    _validate_schema(df)
    return df


def to_model_frame(df: pd.DataFrame, *, threshold: float) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where y is 1 if Budget >= threshold else 0."""
    _validate_schema(df)

    # Target
    y = (pd.to_numeric(df["Budget (in dollars)"], errors="coerce") >= float(threshold)).astype(int)

    # Features (keep original column names for traceability)
    X = df.drop(columns=["Budget (in dollars)"]).copy()

    return X, y
