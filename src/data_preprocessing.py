from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw data from a CSV file.

    Parameters:
        filepath: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Apply basic cleaning operations to the input DataFrame.

    Parameters:
        df: Raw input DataFrame.
        drop_duplicates: Whether to remove duplicate rows.

    Returns:
        Cleaned DataFrame.
    """
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    return cleaned


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into training and test feature/target sets.

    Parameters:
        df: Input DataFrame containing features and target.
        target_column: Name of target column.
        test_size: Fraction of data to reserve for testing.
        random_state: Random seed for deterministic splitting.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in input data.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be a float between 0 and 1.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
