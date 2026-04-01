from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw data from CSV without applying ML transformations.

    Parameters:
        filepath: Path to the source CSV file.

    Returns:
        Raw pandas DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Loaded dataset is empty: {path}")

    # Normalize whitespace in column names while preserving raw values and dtypes.
    df.columns = [str(column).strip() for column in df.columns]
    return df
