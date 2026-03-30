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

1. Create a virtual environment inside the project:

```bash
python -m venv .venv
```

2. Activate the environment.

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) verify installed packages:

```bash
pip list
```

5. Place your dataset at:

```text
data/raw/dataset.csv
```

6. Ensure your dataset includes a target column matching `target_column` in `src/config.py` (default: `target`).

7. Deactivate the environment when finished:

```bash
deactivate
```

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
- Virtual environment folders are ignored via `.gitignore` to keep machine-specific files out of version control.

## Structuring Python Files and Modules Milestone

This branch demonstrates the refactor from script-style ML code to a modular package layout. The workflow is separated into focused modules for preprocessing, feature engineering, training, evaluation, and prediction.

### Reviewer Checklist

- Source code is organized under the `src/` package with one responsibility per module.
- `main.py` orchestrates execution while reusable logic stays inside module functions.
- Configuration is centralized in `src/config.py`.
- Training and prediction responsibilities are separated across `src/train.py` and `src/predict.py`.
- Dependencies are pinned in `requirements.txt`.
