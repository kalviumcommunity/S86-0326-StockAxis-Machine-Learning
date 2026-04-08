from __future__ import annotations

import json
import math

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.baseline import evaluate_regression_baseline
from src.config import Config
from src.data_loader import load_data
from src.data_preprocessing import clean_data, split_data
from src.evaluate import evaluate_model
from src.feature_engineering import (
    add_derived_features,
    build_preprocessing_pipeline,
    fit_preprocessor,
    infer_feature_columns,
    transform_features,
)
from src.persistence import save_artifacts
from src.predict import predict_new_data
from src.train import train_model


def run_training_pipeline(config: Config) -> dict[str, float]:
    """Run the end-to-end training pipeline using modular functions."""
    config.ensure_directories()

    raw_data_path = config.resolve_path(config.data_path)
    processed_data_path = config.resolve_path(config.processed_data_path)
    model_path = config.resolve_path(config.model_path)
    pipeline_path = config.resolve_path(config.pipeline_path)
    metrics_path = config.resolve_path(config.metrics_path)
    predictions_path = config.resolve_path(config.predictions_path)
    coefficients_path = config.resolve_path(config.coefficients_path)
    residuals_path = config.resolve_path(config.residuals_path)

    df_raw = load_data(raw_data_path)
    df_clean = clean_data(df_raw)
    df_features = add_derived_features(df_clean)
    df_features.to_csv(processed_data_path, index=False)

    categorical_cols, numerical_cols = infer_feature_columns(
        df_features,
        target_column=config.target_column,
        categorical_columns=config.categorical_columns,
        numerical_columns=config.numerical_columns,
    )

    X_train, X_test, y_train, y_test = split_data(
        df_features,
        target_column=config.target_column,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    preprocessor = build_preprocessing_pipeline(
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        numerical_scaler=config.numerical_scaler,
    )
    fitted_preprocessor = fit_preprocessor(preprocessor, X_train)

    X_train_processed = transform_features(fitted_preprocessor, X_train)
    X_test_processed = transform_features(fitted_preprocessor, X_test)

    model = train_model(
        X_train=X_train_processed,
        y_train=y_train,
        random_state=config.random_state,
        model_params=config.model_params,
    )

    baseline_metrics = evaluate_regression_baseline(
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        y_test=y_test,
        strategy=config.baseline_strategy,
        random_state=config.random_state,
    )

    model_metrics = evaluate_model(model=model, X_test=X_test_processed, y_test=y_test)
    model_predictions = model.predict(X_test_processed)

    metrics: dict[str, float] = {}
    for metric_name, metric_value in baseline_metrics.items():
        metrics[f"baseline_{metric_name}"] = float(metric_value)

    for metric_name, metric_value in model_metrics.items():
        metrics[f"model_{metric_name}"] = float(metric_value)

    mean_target = float(y_test.mean())
    if mean_target != 0:
        metrics["model_mae_pct_of_mean_target"] = float((model_metrics["mae"] / abs(mean_target)) * 100)
        metrics["baseline_mae_pct_of_mean_target"] = float((baseline_metrics["mae"] / abs(mean_target)) * 100)

    lower_is_better = {"mse", "rmse", "mae"}
    higher_is_better = {"r2"}

    for metric_name, metric_value in model_metrics.items():
        if metric_name in baseline_metrics and metric_name in lower_is_better:
            metrics[f"improvement_{metric_name}"] = float(baseline_metrics[metric_name] - metric_value)
        elif metric_name in baseline_metrics and metric_name in higher_is_better:
            metrics[f"improvement_{metric_name}"] = float(metric_value - baseline_metrics[metric_name])

    cv_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessing_pipeline(
                    categorical_cols=categorical_cols,
                    numerical_cols=numerical_cols,
                    numerical_scaler=config.numerical_scaler,
                ),
            ),
            ("model", LinearRegression(**config.model_params)),
        ]
    )
    cv_r2_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=config.cv_folds, scoring="r2")
    metrics["cv_r2_mean"] = float(cv_r2_scores.mean())
    metrics["cv_r2_std"] = float(cv_r2_scores.std())
    for fold_index, fold_score in enumerate(cv_r2_scores, start=1):
        metrics[f"cv_r2_fold_{fold_index}"] = float(fold_score)

    cv_neg_mae_scores = cross_val_score(
        cv_pipeline,
        X_train,
        y_train,
        cv=config.cv_folds,
        scoring="neg_mean_absolute_error",
    )
    cv_mae_scores = -cv_neg_mae_scores
    metrics["cv_mae_mean"] = float(cv_mae_scores.mean())
    metrics["cv_mae_std"] = float(cv_mae_scores.std())
    for fold_index, fold_score in enumerate(cv_mae_scores, start=1):
        metrics[f"cv_mae_fold_{fold_index}"] = float(fold_score)

    cv_neg_mse_scores = cross_val_score(
        cv_pipeline,
        X_train,
        y_train,
        cv=config.cv_folds,
        scoring="neg_mean_squared_error",
    )
    cv_mse_scores = -cv_neg_mse_scores
    metrics["cv_mse_mean"] = float(cv_mse_scores.mean())
    metrics["cv_mse_std"] = float(cv_mse_scores.std())
    for fold_index, fold_score in enumerate(cv_mse_scores, start=1):
        metrics[f"cv_mse_fold_{fold_index}"] = float(fold_score)

    cv_rmse_scores = [math.sqrt(score) for score in cv_mse_scores]
    metrics["cv_rmse_mean"] = float(sum(cv_rmse_scores) / len(cv_rmse_scores))
    rmse_variance = sum((score - metrics["cv_rmse_mean"]) ** 2 for score in cv_rmse_scores) / len(cv_rmse_scores)
    metrics["cv_rmse_std"] = float(math.sqrt(rmse_variance))
    for fold_index, fold_score in enumerate(cv_rmse_scores, start=1):
        metrics[f"cv_rmse_fold_{fold_index}"] = float(fold_score)

    feature_names = fitted_preprocessor.get_feature_names_out()
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_,
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("abs_coefficient", ascending=False)
    coefficients.to_csv(coefficients_path, index=False)
    metrics["model_intercept"] = float(model.intercept_)

    save_artifacts(model=model, pipeline=fitted_preprocessor, model_path=model_path, pipeline_path=pipeline_path)

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    prediction_output = predict_new_data(
        new_data=X_test,
        model=model,
        pipeline=fitted_preprocessor,
        expected_columns=X_train.columns.tolist(),
    )
    prediction_output.insert(0, "actual", y_test.reset_index(drop=True))
    prediction_output.to_csv(predictions_path, index=False)

    residuals = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "predicted": pd.Series(model_predictions),
        }
    )
    residuals["residual"] = residuals["predicted"] - residuals["actual"]
    residuals["absolute_error"] = residuals["residual"].abs()
    residuals.to_csv(residuals_path, index=False)

    return metrics


def main() -> None:
    """CLI entrypoint for running the ML training workflow."""
    config = Config()
    metrics = run_training_pipeline(config)

    print("Training pipeline completed successfully.")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
