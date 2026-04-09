# Logistic Regression Classification Model Guide

This guide explains how to train, evaluate, and interpret a Logistic Regression classifier in this project.

## Why Logistic Regression

Logistic Regression is a probabilistic classifier for categorical targets. It predicts:

- Class labels via `predict`
- Class probabilities via `predict_proba`

It is a strong first model after a majority-class baseline because it is fast, interpretable, and often competitive when features are engineered well.

## Core Idea

Logistic Regression starts with a linear score:

- `z = w1*x1 + w2*x2 + ... + wn*xn + b`

Then converts the score to probability with the sigmoid function:

- `p(y=1|x) = 1 / (1 + exp(-z))`

Decision rule (default threshold 0.5):

- If `p >= 0.5`, predict class `1`
- Else, predict class `0`

## Loss Function

The model is trained with log loss (cross-entropy), not MSE.

- Log loss heavily penalizes confident wrong predictions.
- This makes probability estimates more meaningful for ranking and thresholding.

## Project Implementation

Classification support is integrated into the same end-to-end pipeline in `main.py`.

### 1. Enable classification mode

Set this in `src/config.py`:

```python
problem_type: str = "classification"
```

### 2. Split strategy

When classification mode is enabled, splitting uses:

- Stratified train/test split (`stratify=True`)

This preserves class proportions in train and test sets.

### 3. Preprocessing

The preprocessing pipeline remains leakage-safe:

- Fit on `X_train`
- Transform `X_train` and `X_test` with the fitted transformer

### 4. Model training

Training uses `sklearn.linear_model.LogisticRegression` via `src/train.py`.

Default classification parameters are:

- `max_iter=1000`
- `random_state=42`

You can override with `model_params` in `src/config.py`.

### 5. Baseline comparison

The baseline in classification mode uses `DummyClassifier`.

If `baseline_strategy` is invalid for classification, the pipeline falls back to:

- `most_frequent`

### 6. Evaluation metrics

Classification mode reports:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC (when probabilities and label distribution allow it)

All metrics are written to `reports/metrics.json` with `baseline_` and `model_` prefixes.

### 7. Cross-validation

Classification mode runs stratified CV and reports:

- `cv_roc_auc_mean`, `cv_roc_auc_std`, per-fold values
- `cv_f1_mean`, `cv_f1_std`, per-fold values

### 8. Coefficient interpretation

For binary classification, `reports/coefficients.csv` includes:

- `feature`
- `coefficient` (log-odds scale)
- `odds_ratio` (`exp(coefficient)`)

Interpretation examples:

- Coefficient `+0.69` -> odds ratio about `2.0` (odds roughly double)
- Coefficient `-0.69` -> odds ratio about `0.5` (odds roughly halve)

### 9. Classification reports and predictions

Outputs in classification mode include:

- `reports/classification_report.txt`
- `reports/predictions.csv`
- `reports/residuals.csv` (classification audit: actual, predicted, correctness)

## Practical Checklist

Before declaring success:

- Model beats majority baseline on ROC-AUC and F1 (not just accuracy)
- Stratified split is used
- No convergence warnings (`max_iter` sufficient)
- Cross-validation mean and std are stable
- Coefficient directions are sensible for domain logic

## Common Pitfalls

- Reporting only accuracy on imbalanced data
- Forgetting to use probabilities for ROC-AUC
- Using non-stratified splits for class-imbalanced targets
- Treating large coefficient magnitude as importance without scaling context

## Recommended Next Step

After Logistic Regression is stable, compare against a tree-based classifier (for non-linear boundaries) while keeping Logistic Regression as the interpretable benchmark.
