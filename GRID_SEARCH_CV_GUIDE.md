# Improving Model Performance Using GridSearchCV

After training a model and evaluating its performance, the next question is simple: can we do better?

Most machine learning models expose hyperparameters, which are configuration settings that control model behavior but are not learned from the data. They are chosen by the practitioner before training begins, and they can make a large difference in the final result. The same algorithm with different hyperparameters can underfit, overfit, or generalize well.

This lesson covers:

- What hyperparameters are and why they matter
- How GridSearchCV works
- How to implement GridSearchCV correctly with pipelines
- How to interpret and visualize search results
- Why the scoring metric matters as much as the grid
- How to avoid data leakage during tuning
- When RandomizedSearchCV is the better choice
- A practical coarse-to-fine tuning strategy

## What Are Hyperparameters?

Hyperparameters are settings defined before training that govern how the model learns, how flexible it can be, and how strongly it is regularized.

They differ from model parameters:

| Type            | Example                      | Learned from data? | Set by           |
| --------------- | ---------------------------- | ------------------ | ---------------- |
| Model parameter | Regression coefficients      | Yes                | The algorithm    |
| Hyperparameter  | K in KNN                     | No                 | The practitioner |
| Hyperparameter  | max_depth in a decision tree | No                 | The practitioner |

Hyperparameters influence:

- Model complexity
- Regularization strength
- Bias-variance balance
- Computational cost

Improper choices have predictable consequences:

- Too restrictive means underfitting and high bias
- Too permissive means overfitting and high variance
- Wrong regularization can destabilize coefficients or harm minority-class performance

## Why Tuning Is Necessary

The optimal hyperparameter value is dataset-specific. It depends on the true structure in the data, the amount of noise, and the evaluation metric you care about.

Decision tree example:

| max_depth | Train Accuracy | Test Accuracy | Diagnosis                  |
| --------- | -------------- | ------------- | -------------------------- |
| 2         | 72%            | 70%           | High bias, underfitting    |
| 4         | 85%            | 83%           | Good balance               |
| 10        | 99%            | 71%           | High variance, overfitting |
| None      | 100%           | 68%           | Extreme overfitting        |

KNN example:

- K = 1 tends to memorize training points and overfit
- K = 50 can smooth the boundary too much and underfit

The correct value has to be found through systematic search.

## How GridSearchCV Works

GridSearchCV performs five steps:

1. Define a grid of hyperparameter values.
2. Generate every combination in the Cartesian product of the grid.
3. Cross-validate each combination on the training set.
4. Select the combination with the best mean validation score.
5. Refit the best configuration on all training data.

That last step matters. After selecting the best hyperparameters, GridSearchCV retrains on the full training set so the final estimator has seen as much training data as possible.

## Implementing GridSearchCV Correctly

Always wrap preprocessing inside a pipeline so cross-validation stays leakage-safe.

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": range(1, 21),
    "knn__weights": ["uniform", "distance"]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    return_train_score=True,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print(f"Best Parameters: {grid.best_params_}")
print(f"Best CV Score: {grid.best_score_:.3f}")

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

The pipeline matters because preprocessing outside the pipeline can leak information across folds. For example, scaling the full training set before cross-validation lets validation samples influence the scaler statistics used to transform themselves. That inflates scores.

## Accessing and Interpreting Results

The full search output is stored in `grid.cv_results_`.

```python
results_df = pd.DataFrame(grid.cv_results_)[[
    "param_knn__n_neighbors",
    "param_knn__weights",
    "mean_train_score",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]]

print(results_df.sort_values("rank_test_score").head(10).to_string(index=False))
```

Key attributes:

- `grid.best_params_`: winning hyperparameter configuration
- `grid.best_score_`: mean CV score at the best configuration
- `grid.best_estimator_`: fitted estimator using the best configuration
- `grid.cv_results_`: full results for every combination and fold

When reviewing the results, do not stop at the top score. Also check the standard deviation across folds and inspect near-best configurations. A slightly simpler model with nearly identical CV performance is often preferable.

## Visualizing Hyperparameter Effects

Visualization turns the search results into a bias-variance picture.

```python
import matplotlib.pyplot as plt
import numpy as np

uniform_mask = results_df["param_knn__weights"] == "uniform"
distance_mask = results_df["param_knn__weights"] == "distance"

k_values = results_df[uniform_mask]["param_knn__n_neighbors"].astype(int)

plt.figure(figsize=(10, 5))
plt.plot(k_values, results_df[uniform_mask]["mean_train_score"], label="Train (uniform)", linestyle="--", marker="o", alpha=0.7)
plt.plot(k_values, results_df[uniform_mask]["mean_test_score"], label="CV (uniform)", marker="o")
plt.plot(k_values, results_df[distance_mask]["mean_test_score"], label="CV (distance)", marker="s", linestyle=":")
plt.fill_between(
    k_values,
    results_df[uniform_mask]["mean_test_score"] - results_df[uniform_mask]["std_test_score"],
    results_df[uniform_mask]["mean_test_score"] + results_df[uniform_mask]["std_test_score"],
    alpha=0.15,
)
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("GridSearchCV: K vs Cross-Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

What the plot usually shows:

- Small K: train accuracy is high, CV accuracy is unstable, and the model overfits
- As K increases: train accuracy drops and CV accuracy rises, then eventually plateaus or falls
- The best K is where CV performance peaks

If the CV curve keeps improving all the way to the edge of the grid, your search range is probably too small.

## The Scoring Metric Matters

GridSearchCV chooses the best model by maximizing the scoring metric you specify. If that metric is wrong, the tuning process is optimizing the wrong objective.

```python
from sklearn.model_selection import GridSearchCV

# For imbalanced classification
grid_f1 = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")

# For medical or fraud-style use cases
grid_recall = GridSearchCV(pipeline, param_grid, cv=5, scoring="recall")

# For regression
grid_reg = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error")
```

For regression metrics like MSE and MAE, scikit-learn uses negative scoring because it follows a higher-is-better convention internally. When reporting, convert the sign back.

Choosing the right metric:

| Problem Type   | Situation                  | Recommended scoring              |
| -------------- | -------------------------- | -------------------------------- |
| Classification | Balanced classes           | `accuracy`                       |
| Classification | Imbalanced classes         | `f1` or `roc_auc`                |
| Classification | False negatives are costly | `recall`                         |
| Classification | False positives are costly | `precision`                      |
| Regression     | Any                        | `neg_mean_squared_error` or `r2` |

## Decision Tree Example

```python
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    "max_depth": [2, 4, 6, 8, 10, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    return_train_score=True,
    n_jobs=-1,
)

grid.fit(X_train, y_train)

print(f"Best Parameters: {grid.best_params_}")
print(f"Best CV F1: {grid.best_score_:.3f}")

y_pred = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
```

This grid evaluates 48 combinations and, with 5-fold cross-validation, 240 model fits. That is manageable for a decision tree, but it can get expensive quickly for larger models.

## Avoiding Data Leakage During Tuning

Never let test-set information influence hyperparameter selection.

Correct workflow:

1. Split the data first.
2. Tune only on the training set with internal cross-validation.
3. Evaluate on the test set exactly once after tuning is complete.

Wrong workflows:

- Running GridSearchCV on the full dataset before splitting
- Looking at test accuracy for each hyperparameter setting and choosing the best one
- Tuning again after seeing test results

If you tune on the test set, the reported score is no longer an unbiased estimate of generalization.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

grid.fit(X_train, y_train)
final_pred = grid.best_estimator_.predict(X_test)
print(f"Final Test Score: {accuracy_score(y_test, final_pred):.3f}")
```

## Computational Considerations

Grid search evaluates every combination. The number of model fits grows as:

$$
\text{fits} = \left(\prod_{i=1}^{m} n_i\right) \times k
$$

where $n_i$ is the number of values for the $i$th hyperparameter and $k$ is the number of cross-validation folds.

Strategies for managing cost:

- Use `n_jobs=-1` to parallelize across cores
- Narrow the grid to the most meaningful values
- Use fewer folds during exploratory work
- Tune the most important parameter first
- Switch to RandomizedSearchCV for large search spaces

## RandomizedSearchCV as a Practical Alternative

RandomizedSearchCV samples a fixed number of configurations instead of evaluating the full Cartesian product.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distributions = {
    "knn__n_neighbors": randint(1, 50),
    "knn__weights": ["uniform", "distance"]
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=50,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_train, y_train)

print(f"Best Parameters: {random_search.best_params_}")
print(f"Best CV Score: {random_search.best_score_:.3f}")
```

Prefer RandomizedSearchCV when:

| Situation                                | Recommendation     |
| ---------------------------------------- | ------------------ |
| Narrow search with a few hyperparameters | GridSearchCV       |
| Wide search or many hyperparameters      | RandomizedSearchCV |
| Tight compute budget                     | RandomizedSearchCV |
| Continuous distributions are needed      | RandomizedSearchCV |

## Coarse-to-Fine Tuning Strategy

Brute-force wide-grid search is inefficient. A better approach is to search coarsely first and then zoom in.

Step 1: coarse search over a wide range.

```python
param_grid_coarse = {"max_depth": [2, 4, 8, 16, None]}
```

Step 2: fine search around the promising region.

```python
param_grid_fine = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_samples_leaf": [1, 5, 10]
}
```

Step 3: evaluate the final model on the test set exactly once.

This strategy often reduces the number of fits by a large factor while preserving final performance.

## Reporting Results Properly

A complete experiment report should include:

- Dataset description
- Model and preprocessing pipeline
- Hyperparameter grid
- CV strategy
- Scoring metric
- Baseline score
- Untuned model score
- Tuned CV mean and standard deviation
- Final test score
- Best hyperparameters

Example reporting style:

- Baseline F1: 0.00
- Untuned model F1: 0.64
- Tuned model CV F1: 0.71 ± 0.03
- Test F1: 0.69
- Best hyperparameters:

```python
{'knn__n_neighbors': 11, 'knn__weights': 'distance'}
```

## Common Mistakes to Avoid

- Tuning on the test set
- Leaving preprocessing outside the pipeline
- Optimizing the wrong metric
- Reporting only the best CV score without the standard deviation
- Building a grid that is too coarse or too narrow
- Ignoring compute cost for large models

## Practical Checklist Before Finalizing

- Train/test split completed before tuning begins
- Preprocessing inside a pipeline
- Grid covers a meaningful range
- Scoring metric matches the business objective
- `return_train_score=True` used when overfitting inspection matters
- Best CV score and standard deviation reported
- Test set evaluated exactly once
- Full search table reviewed
- Improvement over baseline documented

## Closing Reflection

GridSearchCV turns hyperparameter selection from guesswork into systematic, reproducible optimization. It makes the bias-variance trade-off visible, keeps test data out of the tuning loop, and produces results that can be explained and defended.

The key is to tune for the real objective, not just for a convenient metric. A small gain is only meaningful if it matters for the actual application.
