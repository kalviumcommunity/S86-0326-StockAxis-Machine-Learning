from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer


def save_artifacts(
    model: Any,
    pipeline: ColumnTransformer,
    model_path: str | Path,
    pipeline_path: str | Path,
) -> None:
    """Persist trained model and fitted preprocessing pipeline to disk."""
    model_target = Path(model_path)
    pipeline_target = Path(pipeline_path)

    model_target.parent.mkdir(parents=True, exist_ok=True)
    pipeline_target.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_target)
    joblib.dump(pipeline, pipeline_target)


def load_artifacts(
    model_path: str | Path,
    pipeline_path: str | Path,
) -> tuple[Any, ColumnTransformer]:
    """Load persisted model and preprocessing pipeline from disk."""
    model_target = Path(model_path)
    pipeline_target = Path(pipeline_path)

    if not model_target.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_target}")
    if not pipeline_target.exists():
        raise FileNotFoundError(f"Pipeline artifact not found: {pipeline_target}")

    model = joblib.load(model_target)
    pipeline = joblib.load(pipeline_target)
    return model, pipeline


def predict_new_data(
    new_data: pd.DataFrame,
    model: Any,
    pipeline: ColumnTransformer,
    expected_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate predictions for new records using saved artifacts.

    Parameters:
        new_data: Raw feature DataFrame for inference.
        model: Fitted model loaded from disk.
        pipeline: Fitted preprocessing pipeline loaded from disk.
        expected_columns: Optional list of expected feature columns.

    Returns:
        DataFrame with prediction and probability columns.
    """
    if expected_columns:
        missing = [column for column in expected_columns if column not in new_data.columns]
        if missing:
            raise ValueError(f"Prediction data is missing required columns: {missing}")

    transformed = pipeline.transform(new_data)
    transformed_array = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    prediction_df = new_data.copy()
    prediction_df["prediction"] = model.predict(transformed_array)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_array)
        if probabilities.shape[1] == 2:
            prediction_df["prediction_probability"] = probabilities[:, 1]
        else:
            prediction_df["prediction_probability"] = np.max(probabilities, axis=1)

    return prediction_df
