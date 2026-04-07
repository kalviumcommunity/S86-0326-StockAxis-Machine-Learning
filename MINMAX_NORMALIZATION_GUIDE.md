# Normalizing Features Using MinMaxScaler

Not all models require features to be centered at zero with unit variance. In many workflows, the main goal is to map numerical features into a shared bounded range. This is normalization, and MinMaxScaler is a common implementation.

This guide explains:

- What normalization is
- How MinMaxScaler transforms values
- When MinMax scaling helps and when it does not
- How to apply it without data leakage
- How this project supports MinMax scaling in a production-friendly pipeline

## What Is Normalization?

Normalization rescales feature values to a fixed range, usually [0, 1].

MinMaxScaler applies:

$$
x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

Where $x_{min}$ and $x_{max}$ are learned from training data only.

After fitting on training data:

- The minimum training value maps to 0
- The maximum training value maps to 1
- Other values are mapped proportionally between them

## Why MinMax Scaling Helps

MinMax scaling is useful when algorithms are sensitive to feature magnitude.

Typical examples:

- k-Nearest Neighbors
- k-Means clustering
- Support Vector Machines (especially RBF)
- Neural networks
- PCA and other distance or variance-sensitive methods

Benefits include:

- Comparable feature magnitudes
- Better-conditioned optimization
- More stable and faster training for scale-sensitive methods

## When MinMax Scaling Is Not Required

Tree-based models are usually scale-invariant:

- Decision Trees
- Random Forest
- Gradient Boosted Trees

These models split on thresholds and do not rely on Euclidean magnitude in the same way as distance-based models.

## MinMaxScaler vs StandardScaler

MinMaxScaler:

- Bounded output (commonly [0, 1])
- Preserves rank and proportional spacing
- Highly sensitive to outliers

StandardScaler:

- Mean-centered, unit variance output
- Unbounded values
- Often more robust than MinMax when outliers are present

If outliers are extreme, consider clipping, log transforms, or RobustScaler before deciding on MinMax.

## Leakage-Safe Workflow

Always split first, then fit transforms on training data only.

Correct order:

1. Split train and test
2. Fit MinMaxScaler on X_train numerical columns
3. Transform X_train
4. Transform X_test with the same fitted scaler

Never fit on full data before splitting.

## Scale Only Numerical Features

Do not scale categorical columns directly. Handle them with encoding (for example OneHotEncoder).

Use ColumnTransformer to apply different transforms by feature type.

## Project Implementation

This repository supports configurable numerical scaling in the preprocessing pipeline.

In [src/config.py](src/config.py), set:

- numerical_scaler = "minmax" for normalization
- numerical_scaler = "standard" for standardization
- numerical_scaler = "none" to skip scaling

Pipeline behavior is implemented in [src/feature_engineering.py](src/feature_engineering.py) and used by [main.py](main.py).

Because preprocessing is fit only on X_train and reused for X_test and inference, this remains leakage-safe.

## Outlier Caveat

MinMaxScaler uses observed min and max. One extreme value can stretch the scale and compress most values near zero.

Practical mitigations:

- Clip extreme values using train-derived thresholds
- Apply log1p to highly skewed positive features
- Use StandardScaler or RobustScaler when outliers dominate

## Verification

After fitting MinMaxScaler on training numeric columns, expected train stats per feature are approximately:

- min near 0
- max near 1

Test values may go below 0 or above 1 if test data falls outside the training range. This is expected and should not trigger refitting.

## Production Guidance

Persist the fitted preprocessing artifact and reuse it in prediction. Never refit on inference-time data.

This project already saves and loads preprocessing artifacts together with model artifacts through [src/persistence.py](src/persistence.py).
