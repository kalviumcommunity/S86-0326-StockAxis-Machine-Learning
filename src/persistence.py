from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
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
