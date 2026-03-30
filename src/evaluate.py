from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(
    model,
    X_test: np.ndarray | pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a classifier and return a metrics dictionary.

    Parameters:
        model: Fitted classification model implementing predict.
        X_test: Preprocessed test features.
        y_test: Test labels.

    Returns:
        Dictionary with accuracy, precision, recall, f1, and roc_auc metrics.
    """
    predictions = model.predict(X_test)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        if probabilities.shape[1] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(y_test, probabilities, multi_class="ovr", average="weighted")
            )

    return metrics
