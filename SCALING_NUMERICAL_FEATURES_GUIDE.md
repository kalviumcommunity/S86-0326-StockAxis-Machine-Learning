# Scaling Numerical Features Using StandardScaler

Many machine learning models are sensitive to the scale of numerical features. When features exist on very different ranges, one variable can dominate learning simply because its values are numerically larger.

This guide explains:

- Why scaling is necessary
- Which model families require scaling
- How StandardScaler works
- How to implement scaling without leakage
- How this project applies scaling in a reproducible way

Scaling is not optional decoration for many algorithms. It is a mathematical requirement for stable optimization.

## Why Scaling Is Necessary

Consider numerical features with very different ranges:

- Age (18 to 70)
- MonthlyCharges (0 to 150)
- TotalCharges (0 to 10,000)

Without scaling in gradient-based or distance-based models:

- Larger-range features can dominate optimization.
- Convergence can become slow or unstable.
- Distances become distorted.

Scaling brings numerical features to a comparable scale, improving optimization and making coefficients or distances more meaningful.

## Which Models Require Scaling?

Scaling is strongly recommended for:

- Logistic Regression
- Linear Regression (especially Ridge/Lasso)
- Support Vector Machines
- k-Nearest Neighbors
- Neural Networks
- Principal Component Analysis

Scaling is often unnecessary for:

- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)

Tree models split by thresholds and are largely scale-invariant.

## What StandardScaler Does

For each feature, StandardScaler applies:

$$
z = \frac{x - \mu}{\sigma}
$$

Where:

- $\mu$ is the feature mean (computed on training data)
- $\sigma$ is the feature standard deviation (computed on training data)

After transformation, each scaled feature has approximately:

- Mean of 0
- Standard deviation of 1

This makes values dimensionless and comparable across columns.

## Correct Workflow and Leakage Prevention

Always follow this sequence:

1. Separate features and target.
2. Split into train and test sets.
3. Fit scaler only on training features.
4. Transform training features.
5. Transform test features using the same fitted scaler.

Incorrect (leakage):

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # WRONG: full dataset used
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

Correct:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Never call fit on test data.

## Scale Only Numerical Features

Only numerical columns should be standardized.

Do not apply StandardScaler directly to:

- Categorical columns
- One-hot encoded flags without reason
- Target labels

In mixed-type datasets, preprocess numerical and categorical columns separately.

## Recommended: ColumnTransformer + Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)
```

This keeps preprocessing reproducible, leakage-safe, and deployment-friendly.

## How This Project Implements Scaling

This repository scales numerical features inside the preprocessing pipeline, fit on training data only during training orchestration.

Implementation points:

- Split happens before fitting preprocessors.
- Numerical columns are median-imputed, then optionally standardized.
- Categorical columns are imputed and one-hot encoded.
- The fitted preprocessing pipeline is saved with the model for inference reuse.

Configuration option:

- Set scale_numerical_features in src/config.py.
- Default is True.
- Set to False when running scale-invariant tree-only workflows.

## Verifying Scaling

After scaling numeric training features, check approximate mean and variance:

```python
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))
```

Expected: means near 0, standard deviations near 1.

## Common Mistakes

- Scaling before train-test split
- Fitting scaler on full data
- Calling fit_transform on both train and test
- Scaling categorical features by mistake
- Failing to apply identical fitted preprocessing at inference time
- Saving model without preprocessing artifact

## Saving for Production

Always persist the fitted preprocessing pipeline (or scaler) with the trained model, then reuse it in inference.

This project already does that through artifact persistence and loading utilities.

## StandardScaler vs MinMaxScaler

StandardScaler:

- Centers at mean 0 and scales to unit variance
- Often preferred for linear models and SVMs

MinMaxScaler:

- Scales into a fixed range, typically [0, 1]
- Useful when bounded inputs are required

## Best Practices Checklist

- Split before fitting any transform
- Scale only numerical features
- Fit transforms on training data only
- Transform test/inference data with the fitted transformer
- Save preprocessing artifacts with the model
- Document when scaling is enabled or intentionally disabled

## Closing Reflection

Scaling is part of model correctness, not cosmetic preprocessing.

If your algorithm depends on gradients, dot products, or distances, scaling is usually required. Respect the train-test boundary and keep preprocessing consistent from training to deployment.
