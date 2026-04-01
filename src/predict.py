from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer


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
