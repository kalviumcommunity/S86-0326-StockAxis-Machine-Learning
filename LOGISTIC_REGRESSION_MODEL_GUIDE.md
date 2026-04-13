# Logistic Regression Model Guide

After working with regression models that predict continuous values, the next step is classification: predicting discrete categories.

In classification problems, the target is categorical:

- Spam vs Not Spam
- Churn vs No Churn
- Fraud vs Legitimate
- Disease vs No Disease

Logistic Regression is one of the most important baseline classification algorithms because it is fast, interpretable, and often surprisingly competitive.

## Why Logistic Regression Matters

Logistic Regression is usually:

- The first serious model after a majority-class baseline
- A strong benchmark that more complex models must justify beating
- A practical model when probabilities and interpretability matter

Despite its name, it is not a regression model in the usual sense. It predicts class probabilities, then maps those probabilities to class labels.

## What Logistic Regression Predicts

For binary classification, Logistic Regression estimates:

$$
P(y=1\mid x)
$$

It first computes a linear score:

$$
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$

Then transforms that score into a probability with the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

So:

$$
P(y=1\mid x) = \sigma(z)
$$

This guarantees outputs between 0 and 1.

## Why Not Linear Regression for Classification

Using Linear Regression for binary targets causes three major issues:

1. Predictions can be outside [0, 1], so they are invalid as probabilities.
2. MSE is not ideal for binary probability modeling.
3. The relationship near the boundary is non-linear in probability space.

Logistic Regression addresses all three by using a probability link function (sigmoid) and a classification-appropriate loss (log loss).

## Understanding the Sigmoid Function

The sigmoid maps any real number into (0, 1):

| Linear score z | Probability $\sigma(z)$ | Interpretation |
| --- | --- | --- |
| -5 | ~0.007 | Strong class 0 confidence |
| -1 | ~0.269 | Leans class 0 |
| 0 | 0.5 | Maximum uncertainty |
| +1 | ~0.731 | Leans class 1 |
| +5 | ~0.993 | Strong class 1 confidence |

Default decision rule:

- If $P(y=1\mid x) \ge 0.5$, predict class 1
- Else predict class 0

The threshold can be tuned based on business cost of false positives vs false negatives.

## Decision Boundary

The decision boundary is where the model is exactly uncertain:

$$
P(y=1\mid x)=0.5 \Longleftrightarrow z=0
$$

This means Logistic Regression creates a linear boundary:

- Line in 2D
- Plane in 3D
- Hyperplane in higher dimensions

Strength: simple and robust.
Limitation: may underfit curved or highly non-linear boundaries.

## Training Objective: Log Loss

Logistic Regression minimizes binary cross-entropy (log loss):

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]
$$

Where:

- $y_i \in \{0,1\}$ is true class
- $\hat{p}_i$ is predicted probability of class 1

Log loss heavily penalizes confident wrong predictions and encourages calibrated probabilities.

## scikit-learn Implementation Pattern

### 1) Imports

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	roc_auc_score,
)
```

### 2) Train-test split with stratification

```python
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
	stratify=y,
)
```

Stratification keeps class proportions similar in train and test.

### 3) Build leakage-safe pipeline

```python
pipeline = Pipeline([
	("scaler", StandardScaler()),
	("model", LogisticRegression(max_iter=1000, random_state=42)),
])

pipeline.fit(X_train, y_train)
```

Notes:

- Scale features for stable optimization and interpretable coefficient comparison.
- Increase `max_iter` to avoid convergence warnings.

### 4) Predict labels and probabilities

```python
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
```

Both are needed for complete evaluation.

## Evaluation: What to Report

### Accuracy

```python
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")
```

Useful, but can be misleading on imbalanced data.

### Precision, Recall, F1

```python
print(classification_report(y_test, y_pred))
```

- Precision: among predicted positives, how many were truly positive?
- Recall: among actual positives, how many did we catch?
- F1: harmonic mean of precision and recall.

### ROC-AUC

```python
auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {auc:.3f}")
```

ROC-AUC evaluates ranking quality across thresholds. Random ranking is near 0.5.

## Baseline Comparison (Mandatory)

```python
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)
baseline_prob = baseline.predict_proba(X_test)[:, 1]

print("Baseline Accuracy:", accuracy_score(y_test, baseline_pred))
print("Baseline ROC-AUC:", roc_auc_score(y_test, baseline_prob))

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Model ROC-AUC:", roc_auc_score(y_test, y_prob))
```

If Logistic Regression does not clearly beat baseline, investigate features, class imbalance, and data quality.

## Coefficient and Odds Ratio Interpretation

Coefficients are in log-odds units.

Odds ratio is computed as:

$$
\mathrm{odds\ ratio} = e^{\beta_j}
$$

```python
model_step = pipeline.named_steps["model"]

coef_df = pd.DataFrame({
	"Feature": X.columns,
	"Coefficient": model_step.coef_[0],
	"Odds Ratio": np.exp(model_step.coef_[0]),
}).sort_values("Coefficient", key=abs, ascending=False)

print(f"Intercept: {model_step.intercept_[0]:.3f}")
print(coef_df.to_string(index=False))
```

Quick interpretation:

- Coefficient +0.69 -> odds ratio about 2.0 (odds roughly double)
- Coefficient -0.69 -> odds ratio about 0.5 (odds roughly halve)

## Regularization and Hyperparameter C

scikit-learn uses L2 regularization by default.

- Smaller `C` -> stronger regularization
- Larger `C` -> weaker regularization

```python
param_grid = {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)
print("Best C:", grid.best_params_["model__C"])
```

For L1 regularization (sparse coefficients), use a compatible solver such as `liblinear`.

## Cross-Validation for Stability

```python
cv_auc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")

print(f"CV AUC: {cv_auc.mean():.3f} +- {cv_auc.std():.3f}")
print(f"CV F1:  {cv_f1.mean():.3f} +- {cv_f1.std():.3f}")
```

Low variance across folds indicates stable generalization.

## When Logistic Regression Works Well

- Relationship is approximately linear in log-odds space
- You need interpretability and calibrated probabilities
- Features are engineered and scaled
- You need a fast, reliable benchmark

## When It Struggles

- Strong non-linear boundaries
- Important feature interactions are missing
- Severe class imbalance without weighting or resampling
- Very high-dimensional correlated feature spaces

For class imbalance, try `class_weight="balanced"` as an early intervention.

## Project-Specific Usage in StockAxis

This repository already supports Logistic Regression in the shared pipeline.

1. In `src/config.py`, set:

```python
problem_type: str = "classification"
```

2. Run:

```bash
python main.py
```

3. Classification outputs include:

- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.csv`
- `reports/precision_recall_curve.csv` (binary)
- `reports/coefficients.csv` (with odds ratios)

## Practical Checklist

- Model beats majority baseline on ROC-AUC and F1
- Stratified split used
- No convergence warnings
- Accuracy reported with precision/recall/F1 and ROC-AUC
- Cross-validation mean and std reported
- Coefficients directionally sensible
- Regularization strength (`C`) considered

## Common Pitfalls

- Reporting only accuracy on imbalanced data
- Forgetting probability-based metrics
- Non-stratified splitting
- Comparing raw coefficient magnitudes without scaling

## Closing Reflection

Logistic Regression remains one of the most valuable models in practical ML. It balances performance, interpretability, and robustness, and provides a benchmark that more complex models should clearly outperform.

Build the baseline. Train Logistic Regression. Evaluate honestly. Improve deliberately.

## Bonus References

- Scikit-learn `LogisticRegression` API docs
- Scikit-learn linear model user guide
- Scikit-learn classification metrics guide
