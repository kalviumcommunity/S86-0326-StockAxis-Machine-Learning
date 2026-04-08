from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def train_model(
    X_train: np.ndarray | pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    model_params: dict[str, Any] | None = None,
) -> LinearRegression:
    """Train a LinearRegression model using explicit configuration.

    Parameters:
        X_train: Preprocessed training features.
        y_train: Training labels.
        random_state: Random seed for deterministic model behavior.
        model_params: Optional hyperparameters for LinearRegression.

    Returns:
        Fitted LinearRegression instance.
    """
    params = {
        "fit_intercept": True,
        "positive": False,
    }
    if model_params:
        params.update(model_params)

    model = LinearRegression(**params)
    model.fit(X_train, y_train)
    return model
