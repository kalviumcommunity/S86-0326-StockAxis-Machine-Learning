# S86-0326-StockAxis-Machine-Learning

Refactored machine learning workflow into modular, reusable Python functions across dedicated modules.

## Project Structure

```text
project_root/
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- reports/
|-- logs/
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_preprocessing.py
|   |-- feature_engineering.py
|   |-- train.py
|   |-- evaluate.py
|   `-- predict.py
|-- main.py
|-- requirements.txt
`-- README.md
```

## Modules

- `src/config.py`: Centralized configuration for paths, columns, model params, and reproducibility.
- `src/data_preprocessing.py`: Data loading, cleaning, and train-test split functions.
- `src/feature_engineering.py`: Feature inference, derived features, and preprocessing pipeline setup.
- `src/train.py`: Model training function with explicit parameters.
- `src/evaluate.py`: Model evaluation function returning a metrics dictionary.
- `src/predict.py`: Artifact save/load and prediction on new data.
- `main.py`: Orchestration script that executes the full training workflow in sequence.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your dataset at:

```text
data/raw/dataset.csv
```

4. Ensure your dataset includes a target column matching `target_column` in `src/config.py` (default: `target`).

## Run Pipeline

```bash
python main.py
```

## Outputs

After execution, the pipeline writes:

- `data/processed/cleaned_dataset.csv`
- `models/random_forest_model.joblib`
- `models/preprocessing_pipeline.joblib`
- `reports/metrics.json`
- `reports/predictions.csv`

## Notes for Reproducibility

- Randomness is controlled by `random_state` in `src/config.py`.
- Core logic functions return values instead of printing, enabling easier testing and reuse.
- Imports are explicit and modular to support maintainability and collaboration.
