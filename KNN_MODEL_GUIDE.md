# Training a K-Nearest Neighbors (KNN) Model

After linear and logistic regression, KNN introduces a very different learning style.

KNN does not learn coefficients or intercepts. It stores training data and defers most computation to prediction time.

Core principle:

- Similar inputs should produce similar outputs.

KNN is instance-based (lazy) learning. Training is lightweight, but prediction is distance-heavy.

## What KNN Is

KNN is a supervised, non-parametric algorithm for:

- Classification
- Regression

Prediction steps:

1. Choose K (number of neighbors).
2. Compute distance from new point to all training points.
3. Select K nearest points.
4. Aggregate targets:
   - Classification: majority vote
   - Regression: average target value

No gradient descent and no explicit function fitting are required.

## Simple Classification Example

Suppose we predict churn using two features: MonthlyCharges and Tenure.

Set K = 3. For a new customer, nearest neighbors are:

| Neighbor | Distance | Class    |
| -------- | -------- | -------- |
| 1        | 0.41     | Churn    |
| 2        | 0.67     | No Churn |
| 3        | 0.89     | Churn    |

Majority vote is 2 vs 1, so prediction is Churn.

## Simple Regression Example

Set K = 5. Nearest neighbor prices are:

- [52, 50, 48, 55, 51] lakhs

Prediction:

$$
\hat{y} = \frac{52 + 50 + 48 + 55 + 51}{5} = 51.2
$$

By default, neighbors are equally weighted. A distance-weighted variant can give closer points more influence.

## Distance Metrics

Distance defines similarity. Metric choice directly changes neighbor selection and predictions.

### Euclidean (default)

$$
d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_{i=1}^{p}(x_i - z_i)^2}
$$

### Manhattan

$$
d(\mathbf{x}, \mathbf{z}) = \sum_{i=1}^{p}|x_i - z_i|
$$

### Minkowski (general form)

$$
d(\mathbf{x}, \mathbf{z}) = \left(\sum_{i=1}^{p}|x_i - z_i|^q\right)^{1/q}
$$

- q = 1 gives Manhattan
- q = 2 gives Euclidean

### Cosine distance

Often useful for sparse text vectors where direction matters more than magnitude.

```python
from sklearn.neighbors import KNeighborsClassifier

# Euclidean
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")

# Manhattan
knn = KNeighborsClassifier(n_neighbors=5, metric="manhattan")

# Cosine
knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
```

## Why Feature Scaling Is Non-Negotiable

KNN is purely distance-based. Large-scale features dominate distance unless scaled.

Example ranges:

- Age: 18 to 60
- Annual Income: 20,000 to 2,000,000

Without scaling, income overwhelms age in Euclidean distance.

Always use a leakage-safe pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
```

Rule: never run KNN on unscaled features.

## Choosing K and Bias-Variance Trade-Off

K is KNN's main hyperparameter.

Small K (for example K = 1):

- Low bias
- High variance
- Overfitting risk

Large K (for example K = 50):

- High bias
- Low variance
- Underfitting risk

Starting heuristic:

$$
K \approx \sqrt{N_{train}}
$$

Use this only as a starting point, then validate with cross-validation.

## Select K with Cross-Validation

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

param_grid = {"knn__n_neighbors": range(1, 31)}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    return_train_score=True
)
grid.fit(X_train, y_train)

print(f"Best K: {grid.best_params_['knn__n_neighbors']}")
print(f"Best CV Accuracy: {grid.best_score_:.3f}")

k_values = list(range(1, 31))
train_scores = grid.cv_results_["mean_train_score"]
cv_scores = grid.cv_results_["mean_test_score"]

plt.figure(figsize=(9, 5))
plt.plot(k_values, train_scores, linestyle="--", label="Train Accuracy")
plt.plot(k_values, cv_scores, marker="o", label="CV Accuracy")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN: K vs Accuracy")
plt.grid(True)
plt.legend()
plt.show()
```

## KNN Classification in scikit-learn

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"Mean CV Accuracy: {cv_scores.mean():.3f} +- {cv_scores.std():.3f}")
```

## Baseline Comparison

Always compare against majority-class baseline:

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

baseline_acc = accuracy_score(y_test, baseline_pred)
knn_acc = accuracy_score(y_test, y_pred)

print(f"Baseline Accuracy: {baseline_acc:.3f}")
print(f"KNN Accuracy:      {knn_acc:.3f}")
print(f"Improvement:       +{knn_acc - baseline_acc:.3f}")
```

If KNN does not beat baseline clearly, inspect feature quality, scaling, noise, and K selection.

## KNN for Regression

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

pipeline_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])

pipeline_reg.fit(X_train, y_train)
y_pred = pipeline_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_r2 = r2_score(y_test, baseline.predict(X_test))

print(f"KNN RMSE: {rmse:.3f}")
print(f"KNN R2:   {r2:.3f}")
print(f"Baseline R2: {baseline_r2:.3f}")
```

Regression limitation: KNN does not extrapolate beyond observed target ranges.

## Curse of Dimensionality

As feature count increases:

- distances become less discriminative
- nearest and farthest points become similar in distance
- local neighborhoods become sparse
- required data grows rapidly

Practical impact: KNN often degrades on high-dimensional data.

A common mitigation is dimensionality reduction before KNN:

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
```

## Strengths

- Simple and intuitive
- No strong distribution assumptions
- Can model non-linear local structure
- Prediction explanations can reference actual neighbors
- Useful baseline on small, low-dimensional datasets

## Weaknesses

- Prediction can be slow on large training sets
- Memory usage is high (stores full training data)
- Sensitive to irrelevant features
- Sensitive to scaling errors
- Often weak in high-dimensional spaces

## Common Mistakes

- skipping feature scaling
- choosing K arbitrarily
- using KNN on very large datasets without latency checks
- keeping many irrelevant features
- failing to compare against baseline
- reporting only one split without CV stability

## Practical Checklist Before Reporting

- features scaled inside a pipeline
- K selected via cross-validation
- K vs performance curve inspected
- model beats baseline meaningfully
- confusion matrix reviewed for classification
- classification report reviewed (especially minority class)
- CV mean and std reported
- feature relevance reviewed
- prediction-time cost assessed for deployment

## When KNN Works Well

- small to medium datasets
- low-dimensional feature spaces
- clear local clustering structure
- interpretable neighbor-based reasoning is useful

## When KNN Struggles

- high-dimensional data without reduction
- very large datasets with strict latency constraints
- noisy datasets with many irrelevant features
- real-time systems needing fast inference

## Closing Reflection

KNN is simple but instructive. It forces careful thinking about similarity, scaling, local structure, and dimensionality.

Even when KNN is not the final production model, mastering it improves model evaluation discipline and feature engineering intuition across algorithms.
