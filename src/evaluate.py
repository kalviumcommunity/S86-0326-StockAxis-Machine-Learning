from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    model,
    X_test: np.ndarray | pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a regression model and return a metrics dictionary.

    Parameters:
        model: Fitted regression model implementing predict.
        X_test: Preprocessed test features.
        y_test: Test labels.

    Returns:
        Dictionary with mse, rmse, mae, and r2 metrics.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    metrics: dict[str, float] = {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    return metrics
