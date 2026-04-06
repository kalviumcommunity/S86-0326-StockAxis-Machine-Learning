# Splitting Data into Training and Testing Sets

Before a model can be trusted, it must be evaluated on data it has never seen before.
If you train and evaluate on the same dataset, you are measuring memorization, not generalization.

This guide explains why data splitting is essential, how to do it correctly, and how to avoid leakage mistakes that silently invalidate evaluation.

## Why Data Splitting Is Necessary

Machine learning models are designed to generalize to unseen data.
If evaluation uses training data:

- The model may memorize patterns.
- Overfitting can go undetected.
- Metrics can look artificially high.
- Real deployment performance often drops.

A proper split simulates reality:

- Training data represents historical observations.
- Testing data represents unseen or future observations.
- Test metrics estimate production performance.

## Training Set vs Testing Set

### Training Set

The training set is used to:

- Fit preprocessing transformations
- Learn model parameters
- Tune hyperparameters (typically with cross-validation)

### Testing Set

The testing set is used to:

- Evaluate final model performance
- Compute unbiased metrics
- Simulate unseen inputs

The test set should remain untouched until final evaluation.

## Standard Train-Test Split

A common split ratio is 80% train and 20% test.

```python
from src.data_preprocessing import split_data

X_train, X_test, y_train, y_test = split_data(
    df,
    target_column="target",
    test_size=0.2,
    random_state=42,
)
```

Important parameters:

- `test_size`: proportion for test set
- `random_state`: reproducible split
- `shuffle`: shuffle rows before split (default `True`)

## Stratified Split for Classification

For imbalanced classification, random splitting may distort class proportions.
Use stratification to preserve label distribution in both sets.

```python
X_train, X_test, y_train, y_test = split_data(
    df,
    target_column="target",
    test_size=0.2,
    random_state=42,
    stratify=True,
)
```

Use stratification for binary and multi-class problems when class balance matters.

## Split Before Any Fitting

Critical rule: split data before fitting transformations.

Incorrect:

```python
# WRONG
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

Correct:

```python
# RIGHT
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

In this project, preprocessing is fit only on `X_train` in [main.py](main.py#L50).

## Time-Based Split for Temporal Data

For time-dependent data, random shuffling is not valid.
Train on earlier records and test on later records.

```python
X_train, X_test, y_train, y_test = split_data(
    df,
    target_column="target",
    test_size=0.2,
    time_column="date",
)
```

When `time_column` is provided, the function sorts chronologically and performs a forward split.

## Common Leakage Mistakes

- Fitting scalers before splitting
- Feature selection on full dataset
- Target-based encoding/statistics from full dataset
- Oversampling before splitting

SMOTE example:

```python
# RIGHT
X_train, X_test, y_train, y_test = split_data(df, target_column="target")
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
```

Apply resampling only on training data.

## Verify Your Split

Always validate split quality:

```python
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("Train distribution:")
print(y_train.value_counts(normalize=True))

print("Test distribution:")
print(y_test.value_counts(normalize=True))
```

## Cross-Validation and Test Set

Train-test split provides one holdout estimate.
Cross-validation gives a more stable estimate during model selection.

Use cross-validation only on training data.
Keep test data for final evaluation only.

## Best Practices Checklist

- Split before fitting any transformation
- Use stratified split for classification when appropriate
- Use chronological split for temporal data
- Fix `random_state` for reproducibility
- Keep test set untouched until final evaluation
- Document strategy clearly in project docs

## Suggested Documentation Block

```markdown
## Data Splitting Strategy

- Dataset split into 80% training and 20% testing
- Stratified split for classification tasks
- random_state=42 for reproducibility
- Preprocessing fitted only on training data
- Test set untouched until final evaluation
```

## Closing Reflection

A strong model evaluated on contaminated splits is not trustworthy.
Protect the train-test boundary, and your metrics will remain meaningful.
