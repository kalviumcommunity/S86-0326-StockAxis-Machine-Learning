from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.baseline import evaluate_classification_baseline, evaluate_regression_baseline
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
    problem_type = config.problem_type.strip().lower()
    if problem_type not in {"regression", "classification"}:
        raise ValueError("Config.problem_type must be either 'regression' or 'classification'.")

    raw_data_path = config.resolve_path(config.data_path)
    processed_data_path = config.resolve_path(config.processed_data_path)
    model_path = config.resolve_path(config.model_path)
    if problem_type == "classification" and model_path.name == "linear_regression_model.joblib":
        model_path = model_path.with_name("logistic_regression_model.joblib")
    pipeline_path = config.resolve_path(config.pipeline_path)
    metrics_path = config.resolve_path(config.metrics_path)
    predictions_path = config.resolve_path(config.predictions_path)
    coefficients_path = config.resolve_path(config.coefficients_path)
    residuals_path = config.resolve_path(config.residuals_path)
    classification_report_path = config.resolve_path(config.classification_report_path)
    confusion_matrix_path = config.resolve_path(config.confusion_matrix_path)

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
        stratify=(problem_type == "classification"),
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
        problem_type=problem_type,
        model_params=config.model_params,
    )

    if problem_type == "regression":
        baseline_metrics = evaluate_regression_baseline(
            X_train=X_train_processed,
            y_train=y_train,
            X_test=X_test_processed,
            y_test=y_test,
            strategy=config.baseline_strategy,
            random_state=config.random_state,
        )
    else:
        baseline_strategy = (
            config.baseline_strategy
            if config.baseline_strategy in {"most_frequent", "stratified", "uniform", "prior", "constant"}
            else "most_frequent"
        )
        baseline_metrics = evaluate_classification_baseline(
            X_train=X_train_processed,
            y_train=y_train,
            X_test=X_test_processed,
            y_test=y_test,
            strategy=baseline_strategy,
            random_state=config.random_state,
        )

    model_metrics = evaluate_model(
        model=model,
        X_test=X_test_processed,
        y_test=y_test,
        problem_type=problem_type,
    )
    model_predictions = model.predict(X_test_processed)

    metrics: dict[str, float] = {}
    for metric_name, metric_value in baseline_metrics.items():
        metrics[f"baseline_{metric_name}"] = float(metric_value)

    for metric_name, metric_value in model_metrics.items():
        metrics[f"model_{metric_name}"] = float(metric_value)

    if problem_type == "regression":
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

        cv_splitter = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        cv_r2_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=cv_splitter, scoring="r2")
        metrics["cv_r2_mean"] = float(cv_r2_scores.mean())
        metrics["cv_r2_std"] = float(cv_r2_scores.std())
        for fold_index, fold_score in enumerate(cv_r2_scores, start=1):
            metrics[f"cv_r2_fold_{fold_index}"] = float(fold_score)

        cv_neg_mae_scores = cross_val_score(
            cv_pipeline,
            X_train,
            y_train,
            cv=cv_splitter,
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
            cv=cv_splitter,
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
    else:
        higher_is_better = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"}
        for metric_name, metric_value in model_metrics.items():
            if metric_name in baseline_metrics and metric_name in higher_is_better:
                metrics[f"improvement_{metric_name}"] = float(metric_value - baseline_metrics[metric_name])

        logistic_params = {"max_iter": 1000, "random_state": config.random_state}
        logistic_params.update(config.model_params)

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
                ("model", LogisticRegression(**logistic_params)),
            ]
        )

        cv_splitter = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        is_binary = y_train.nunique() == 2
        auc_scoring = "roc_auc" if is_binary else "roc_auc_ovr_weighted"
        f1_scoring = "f1" if is_binary else "f1_weighted"

        cv_auc_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=cv_splitter, scoring=auc_scoring)
        metrics["cv_roc_auc_mean"] = float(cv_auc_scores.mean())
        metrics["cv_roc_auc_std"] = float(cv_auc_scores.std())
        for fold_index, fold_score in enumerate(cv_auc_scores, start=1):
            metrics[f"cv_roc_auc_fold_{fold_index}"] = float(fold_score)

        cv_accuracy_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=cv_splitter, scoring="accuracy")
        metrics["cv_accuracy_mean"] = float(cv_accuracy_scores.mean())
        metrics["cv_accuracy_std"] = float(cv_accuracy_scores.std())
        for fold_index, fold_score in enumerate(cv_accuracy_scores, start=1):
            metrics[f"cv_accuracy_fold_{fold_index}"] = float(fold_score)

        cv_balanced_accuracy_scores = cross_val_score(
            cv_pipeline,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring="balanced_accuracy",
        )
        metrics["cv_balanced_accuracy_mean"] = float(cv_balanced_accuracy_scores.mean())
        metrics["cv_balanced_accuracy_std"] = float(cv_balanced_accuracy_scores.std())
        for fold_index, fold_score in enumerate(cv_balanced_accuracy_scores, start=1):
            metrics[f"cv_balanced_accuracy_fold_{fold_index}"] = float(fold_score)

        cv_f1_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=cv_splitter, scoring=f1_scoring)
        metrics["cv_f1_mean"] = float(cv_f1_scores.mean())
        metrics["cv_f1_std"] = float(cv_f1_scores.std())
        for fold_index, fold_score in enumerate(cv_f1_scores, start=1):
            metrics[f"cv_f1_fold_{fold_index}"] = float(fold_score)

    feature_names = fitted_preprocessor.get_feature_names_out()
    if problem_type == "regression":
        coefficients = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": model.coef_,
            }
        )
        coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
        coefficients = coefficients.sort_values("abs_coefficient", ascending=False)
        metrics["model_intercept"] = float(model.intercept_)
    else:
        coef_array = model.coef_
        if coef_array.ndim == 1 or coef_array.shape[0] == 1:
            coefficient_values = coef_array if coef_array.ndim == 1 else coef_array[0]
            coefficients = pd.DataFrame(
                {
                    "feature": feature_names,
                    "coefficient": coefficient_values,
                    "odds_ratio": np.exp(coefficient_values),
                }
            )
            coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
            coefficients = coefficients.sort_values("abs_coefficient", ascending=False)
        else:
            coefficient_frames: list[pd.DataFrame] = []
            for class_index, class_label in enumerate(model.classes_):
                class_coefficients = coef_array[class_index]
                class_df = pd.DataFrame(
                    {
                        "class_label": class_label,
                        "feature": feature_names,
                        "coefficient": class_coefficients,
                        "odds_ratio": np.exp(class_coefficients),
                    }
                )
                class_df["abs_coefficient"] = class_df["coefficient"].abs()
                coefficient_frames.append(class_df)
            coefficients = pd.concat(coefficient_frames, ignore_index=True)
            coefficients = coefficients.sort_values(["class_label", "abs_coefficient"], ascending=[True, False])

        metrics["model_intercept_mean"] = float(np.mean(model.intercept_))

    coefficients.to_csv(coefficients_path, index=False)

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

    if problem_type == "regression":
        residuals = pd.DataFrame(
            {
                "actual": y_test.reset_index(drop=True),
                "predicted": pd.Series(model_predictions),
            }
        )
        residuals["residual"] = residuals["predicted"] - residuals["actual"]
        residuals["absolute_error"] = residuals["residual"].abs()
        residuals.to_csv(residuals_path, index=False)
    else:
        classification_audit = pd.DataFrame(
            {
                "actual": y_test.reset_index(drop=True),
                "predicted": pd.Series(model_predictions),
            }
        )
        classification_audit["correct"] = classification_audit["actual"] == classification_audit["predicted"]
        classification_audit["error"] = (~classification_audit["correct"]).astype(int)
        classification_audit.to_csv(residuals_path, index=False)

        report_text = classification_report(y_test, model_predictions, zero_division=0)
        with open(classification_report_path, "w", encoding="utf-8") as report_file:
            report_file.write(report_text)

        labels = sorted(pd.unique(pd.concat([y_test, pd.Series(model_predictions)])))
        cm = confusion_matrix(y_test, model_predictions, labels=labels)
        confusion_matrix_df = pd.DataFrame(cm, index=[f"actual_{label}" for label in labels])
        confusion_matrix_df.columns = [f"predicted_{label}" for label in labels]
        confusion_matrix_df.to_csv(confusion_matrix_path)

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
