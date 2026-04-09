from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def train_model(
    X_train: np.ndarray | pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    problem_type: str = "regression",
    model_params: dict[str, Any] | None = None,
) -> LinearRegression | LogisticRegression:
    """Train a regression or classification model using explicit configuration.

    Parameters:
        X_train: Preprocessed training features.
        y_train: Training labels.
        random_state: Random seed for deterministic model behavior.
        problem_type: Either "regression" or "classification".
        model_params: Optional hyperparameters for the selected model type.

    Returns:
        Fitted LinearRegression or LogisticRegression instance.
    """
    if problem_type == "regression":
        params: dict[str, Any] = {
            "fit_intercept": True,
            "positive": False,
        }
        model_cls = LinearRegression
    elif problem_type == "classification":
        params = {
            "max_iter": 1000,
            "random_state": random_state,
        }
        model_cls = LogisticRegression
    else:
        raise ValueError("Unsupported problem_type. Use 'regression' or 'classification'.")

    if model_params:
        params.update(model_params)

    model = model_cls(**params)
    model.fit(X_train, y_train)
    return model
