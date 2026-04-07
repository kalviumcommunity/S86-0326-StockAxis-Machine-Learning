# Creating a Baseline Model Using Simple Heuristics

Before tuning complex models, establish a baseline. A baseline is a deliberately simple reference model that sets the minimum performance bar your trained model must beat.

Without a baseline, evaluation numbers are not grounded.

## Why Baselines Matter

A baseline helps you:

- Set a minimum acceptable performance threshold
- Detect misleading metrics on imbalanced data
- Catch leakage or pipeline bugs when results look suspicious
- Quantify real improvement from modeling effort

Example: If the majority class is 80%, a model with 80% accuracy may add no value.

## Common Baselines

For classification:

- Most frequent class (`DummyClassifier(strategy="most_frequent")`)
- Stratified random predictions (`strategy="stratified")`
- Uniform random predictions (`strategy="uniform")`
- Rule-based heuristics from domain knowledge

For regression:

- Predict mean (`DummyRegressor(strategy="mean")`)
- Predict median (`DummyRegressor(strategy="median")`)

## Leakage-Safe Baseline Workflow

Use the same split and evaluation protocol as your real model:

1. Split train/test first
2. Fit baseline on training data only
3. Evaluate on held-out test data
4. Compare baseline and model on identical metrics

Never inspect test labels to design heuristic rules.

## Project Implementation

This repository now includes classification baseline evaluation in the training pipeline.

- Baseline logic: [src/baseline.py](src/baseline.py)
- Baseline configuration: [src/config.py](src/config.py)
- Integrated comparison output: [main.py](main.py)

Config option:

- `baseline_strategy` in [src/config.py](src/config.py)

Supported strategies are those accepted by scikit-learn DummyClassifier, including:

- `most_frequent`
- `stratified`
- `uniform`
- `prior`

## Metrics Output

The metrics report includes three groups:

- `baseline_*` metrics
- `model_*` metrics
- `improvement_*` metrics where:

$$
improvement = model - baseline
$$

This keeps model claims tied to a meaningful reference.

## Recommended Interpretation

When reporting results, include both absolute values and deltas.

Example style:

- Baseline F1: 0.65
- Model F1: 0.79
- Improvement: +0.14

This is far more informative than reporting only model accuracy.

## Common Mistakes

- Skipping the baseline entirely
- Comparing baseline and model with different metrics
- Fitting baseline on full data before splitting
- Optimizing a model that barely beats baseline with no business impact

## Production Guidance

Track baseline metrics in experiment logs and monitor them over data versions. If baseline behavior shifts suddenly, investigate data drift or pipeline changes.

A baseline is not a formality. It is the floor your model must justify surpassing.
