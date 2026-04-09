from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    """Centralized project configuration for the ML workflow."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_path: Path = field(default_factory=lambda: Path("data/raw/dataset.csv"))
    processed_data_path: Path = field(default_factory=lambda: Path("data/processed/cleaned_dataset.csv"))
    model_path: Path = field(default_factory=lambda: Path("models/linear_regression_model.joblib"))
    pipeline_path: Path = field(default_factory=lambda: Path("models/preprocessing_pipeline.joblib"))
    metrics_path: Path = field(default_factory=lambda: Path("reports/metrics.json"))
    predictions_path: Path = field(default_factory=lambda: Path("reports/predictions.csv"))
    coefficients_path: Path = field(default_factory=lambda: Path("reports/coefficients.csv"))
    residuals_path: Path = field(default_factory=lambda: Path("reports/residuals.csv"))
    classification_report_path: Path = field(default_factory=lambda: Path("reports/classification_report.txt"))

    problem_type: str = "regression"
    target_column: str = "target"
    test_size: float = 0.2
    random_state: int = 42

    model_params: dict[str, Any] = field(default_factory=dict)

    categorical_columns: list[str] = field(default_factory=list)
    numerical_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)
    numerical_scaler: str = "standard"
    baseline_strategy: str = "mean"
    cv_folds: int = 5

    def resolve_path(self, path_like: Path) -> Path:
        """Resolve project-relative paths to absolute paths."""
        if path_like.is_absolute():
            return path_like
        return self.project_root / path_like

    def ensure_directories(self) -> None:
        """Create project directories needed for outputs and reports."""
        directories = [
            self.resolve_path(Path("data/raw")),
            self.resolve_path(Path("data/processed")),
            self.resolve_path(Path("data/external")),
            self.resolve_path(Path("models")),
            self.resolve_path(Path("reports")),
            self.resolve_path(Path("logs")),
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
