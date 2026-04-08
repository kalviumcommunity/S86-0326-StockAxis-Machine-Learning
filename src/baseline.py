from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.evaluate import evaluate_model


def evaluate_regression_baseline(
    X_train: np.ndarray | pd.DataFrame,
    y_train: pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: pd.Series,
    strategy: str = "mean",
    random_state: int = 42,
) -> dict[str, float]:
    """Train and evaluate a DummyRegressor baseline on held-out test data.

    Parameters:
        X_train: Preprocessed training features.
        y_train: Training targets.
        X_test: Preprocessed test features.
        y_test: Test labels.
        strategy: Dummy baseline strategy.
        random_state: Random seed used by stochastic strategies.

    Returns:
        Regression metrics for the baseline model.
    """
    model = DummyRegressor(strategy=strategy)
    model.fit(X_train, y_train)
    return evaluate_model(model=model, X_test=X_test, y_test=y_test)
