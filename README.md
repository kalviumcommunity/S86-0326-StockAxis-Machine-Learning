# S86-0326-StockAxis-Machine-Learning

Refactored machine learning workflow into modular, reusable Python functions across dedicated modules.

## Project Structure

```text
project_root/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- external/
|-- notebooks/
|-- models/
|-- reports/
|-- logs/
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- data_preprocessing.py
|   |-- feature_engineering.py
|   |-- train.py
|   |-- evaluate.py
|   |-- predict.py
|   `-- persistence.py
|-- main.py
|-- requirements.txt
`-- README.md
```

## Modules

- `src/config.py`: Centralized configuration for paths, columns, model params, and reproducibility.
- `src/data_loader.py`: Raw data loading and file-level input validation.
- `src/data_preprocessing.py`: Cleaning plus train-test split helpers (standard, stratified, and time-based).
- `src/feature_engineering.py`: Feature inference, derived features, and preprocessing pipeline setup.
- `src/train.py`: Model training function with explicit parameters.
- `src/evaluate.py`: Model evaluation function returning a metrics dictionary.
- `src/predict.py`: Prediction on new data using already-fitted artifacts.
- `src/persistence.py`: Centralized save/load functions for model and preprocessing artifacts.
- `main.py`: Orchestration script that executes the full training workflow in sequence.

## Data Responsibility Rules

- `data/raw/` stores original source-of-truth files and should remain unchanged.
- `data/processed/` stores reproducible outputs from preprocessing and feature steps.
- `data/external/` stores third-party or reference datasets.

## Pipeline Separation

Training flow:

`raw data -> preprocessing -> feature engineering -> fit model -> save artifacts -> evaluate`

Prediction flow:

`new data -> load artifacts -> transform -> predict`

## Responsibility Boundaries

- Data loading is isolated in `src/data_loader.py` and does not split, fit, or engineer features.
- Training logic (including fitting preprocessors and models) runs in the orchestration pipeline.
- Inference logic in `src/predict.py` only applies learned transformations and predicts.
- Artifact save/load is isolated in `src/persistence.py`.

## Setup

Python version: **3.10**

This project uses exact dependency pinning in `requirements.txt` to keep training, evaluation,
and model serialization behavior reproducible across machines.

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

4. Verify that pinned versions are installed:

```bash
pip freeze
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

## Dependency Management

Dependencies are tracked in `requirements.txt` and pinned with `==`.

Current direct dependencies:

- `joblib==1.4.2`
- `numpy==2.2.4`
- `pandas==2.2.3`
- `scikit-learn==1.6.1`

Install from the file with:

```bash
pip install -r requirements.txt
```

If dependencies change, update `requirements.txt` in the same commit.
For sprint-grade reproducibility, rebuild from a clean environment before submission:

1. Delete `.venv`
2. Recreate and activate virtual environment
3. Run `pip install -r requirements.txt`
4. Run `python main.py`

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

## Learning Guides

- `FEATURE_TARGET_DEFINITION_GUIDE.md`
- `FEATURE_TARGET_IMPLEMENTATION.md`
- `FEATURE_DISTRIBUTIONS_QUICK_REFERENCE.md`
- `FEATURE_DISTRIBUTION_IMPLEMENTATION.md`
- `PROBLEM_DEFINITION.md`
- `PROBLEM_TYPE_REFERENCE_GUIDE.md`
- `DATA_SPLITTING_GUIDE.md`

## Notes for Reproducibility

- Randomness is controlled by `random_state` in `src/config.py`.
- Python version is fixed to 3.10 and package versions are pinned in `requirements.txt`.
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
