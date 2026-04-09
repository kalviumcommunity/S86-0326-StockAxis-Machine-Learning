from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    X_test: np.ndarray | pd.DataFrame,
    y_test: pd.Series,
    problem_type: str = "regression",
) -> dict[str, float]:
    """Evaluate a regression or classification model and return metrics.

    Parameters:
        model: Fitted sklearn-compatible model implementing predict.
        X_test: Preprocessed test features.
        y_test: Test labels.
        problem_type: Either "regression" or "classification".

    Returns:
        Dictionary with task-appropriate metrics.
    """
    predictions = model.predict(X_test)

    if problem_type == "regression":
        mse = mean_squared_error(y_test, predictions)
        metrics: dict[str, float] = {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
        }
        return metrics

    if problem_type != "classification":
        raise ValueError("Unsupported problem_type. Use 'regression' or 'classification'.")

    unique_labels = np.unique(y_test)
    is_binary = len(unique_labels) == 2
    average_mode = "binary" if is_binary else "weighted"

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average=average_mode, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average=average_mode, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average=average_mode, zero_division=0)),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        try:
            if is_binary:
                metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_test, probabilities, multi_class="ovr", average="weighted")
                )
        except ValueError:
            # ROC-AUC requires both classes present in y_test.
            pass

    return metrics
