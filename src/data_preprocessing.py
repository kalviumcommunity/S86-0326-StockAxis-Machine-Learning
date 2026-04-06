from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


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
    stratify: bool = False,
    time_column: str | None = None,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into training and test feature/target sets.

    Parameters:
        df: Input DataFrame containing features and target.
        target_column: Name of target column.
        test_size: Fraction of data to reserve for testing.
        random_state: Random seed for deterministic splitting.
        stratify: Whether to preserve target class proportions (classification).
        time_column: Optional datetime/order column for chronological splitting.
        shuffle: Whether to shuffle before random splitting.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in input data.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be a float between 0 and 1.")

    if time_column is not None and time_column not in df.columns:
        raise ValueError(f"time_column '{time_column}' does not exist in input data.")

    if time_column is not None:
        df_sorted = df.sort_values(time_column)
        split_index = int(len(df_sorted) * (1 - test_size))

        train_df = df_sorted.iloc[:split_index]
        test_df = df_sorted.iloc[split_index:]

        X_train = train_df.drop(columns=[target_column])
        X_test = test_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        y_test = test_df[target_column]
        return X_train, X_test, y_train, y_test

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_values = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_values,
    )
