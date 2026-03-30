from __future__ import annotations

import json

from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.evaluate import evaluate_model
from src.feature_engineering import (
    add_derived_features,
    build_preprocessing_pipeline,
    fit_preprocessor,
    infer_feature_columns,
    transform_features,
)
from src.predict import predict_new_data, save_artifacts
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

    metrics = evaluate_model(model=model, X_test=X_test_processed, y_test=y_test)

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
