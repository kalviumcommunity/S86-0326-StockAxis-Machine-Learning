from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_model(
    X_train: np.ndarray | pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    model_params: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """Train a RandomForestClassifier using explicit configuration.

    Parameters:
        X_train: Preprocessed training features.
        y_train: Training labels.
        random_state: Random seed for deterministic model behavior.
        model_params: Optional hyperparameters for RandomForestClassifier.

    Returns:
        Fitted RandomForestClassifier instance.
    """
    params = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }
    if model_params:
        params.update(model_params)

    model = RandomForestClassifier(random_state=random_state, **params)
    model.fit(X_train, y_train)
    return model
