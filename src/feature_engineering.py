from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight derived features useful across ML workflows.

    Parameters:
        df: Input DataFrame.

    Returns:
        DataFrame with derived features added.
    """
    engineered = df.copy()
    engineered["missing_value_count"] = engineered.isna().sum(axis=1)
    return engineered


def infer_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: Sequence[str] | None = None,
    numerical_columns: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Infer or validate categorical and numerical feature columns.

    Parameters:
        df: Input DataFrame containing features and target.
        target_column: Name of target column to exclude from features.
        categorical_columns: Optional user-provided categorical columns.
        numerical_columns: Optional user-provided numerical columns.

    Returns:
        Tuple of (categorical_cols, numerical_cols).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in input data.")

    feature_df = df.drop(columns=[target_column])

    if categorical_columns and numerical_columns:
        missing_columns = [
            col
            for col in list(categorical_columns) + list(numerical_columns)
            if col not in feature_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Configured feature columns not found in data: {missing_columns}")
        return list(categorical_columns), list(numerical_columns)

    inferred_categorical = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    inferred_numerical = feature_df.select_dtypes(include=[np.number]).columns.tolist()

    return inferred_categorical, inferred_numerical


def build_preprocessing_pipeline(
    categorical_cols: Sequence[str],
    numerical_cols: Sequence[str],
    scale_numerical: bool = True,
) -> ColumnTransformer:
    """Build sklearn preprocessing pipeline for categorical and numerical features.

    Parameters:
        categorical_cols: Categorical feature names.
        numerical_cols: Numerical feature names.

    Returns:
        A configured sklearn ColumnTransformer.
    """
    if not categorical_cols and not numerical_cols:
        raise ValueError("At least one categorical or numerical column is required.")

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numerical_cols:
        numeric_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numerical:
            numeric_steps.append(("scaler", StandardScaler()))

        numeric_pipeline = Pipeline(steps=numeric_steps)
        transformers.append(("num", numeric_pipeline, list(numerical_cols)))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipeline, list(categorical_cols)))

    return ColumnTransformer(transformers=transformers)


def fit_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit preprocessing pipeline on training features."""
    return preprocessor.fit(X_train)


def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> np.ndarray:
    """Apply a fitted preprocessing pipeline to a feature DataFrame."""
    transformed = preprocessor.transform(X)
    return transformed.toarray() if hasattr(transformed, "toarray") else transformed
