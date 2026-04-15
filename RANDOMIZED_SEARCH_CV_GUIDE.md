# Improving Model Performance Using RandomizedSearchCV

In the previous lesson, you learned how GridSearchCV evaluates every combination inside a fixed grid. That is rigorous, but it does not scale well.

If a search space has multiple hyperparameters with several values each, exhaustive search grows multiplicatively. For small models, that can still be manageable. For larger models like Random Forests or Gradient Boosting, it quickly becomes too expensive.

RandomizedSearchCV solves that problem by sampling a fixed number of configurations from specified distributions instead of evaluating every combination.

This lesson covers:

- Why grid search becomes inefficient in high-dimensional spaces
- The intuition behind random search
- How to define parameter distributions
- How to implement RandomizedSearchCV in scikit-learn
- When to use randint, uniform, and loguniform
- How to tune n_iter effectively
- How to avoid leakage during randomized search
- How to combine random search with grid search

## Why Grid Search Becomes Inefficient

Grid search evaluates every combination in the grid. That creates three problems:

- The number of fits grows multiplicatively as you add hyperparameters.
- Some hyperparameters matter much more than others, so grid search wastes effort on weak dimensions.
- Fixed grid points can miss the true optimum in continuous spaces.

If only a few hyperparameters strongly affect performance, random search tends to explore the important dimensions more efficiently than grid search.

## The Key Intuition

Suppose only two of five hyperparameters have a large impact on model quality. A grid with 10 values per dimension gives each important hyperparameter only 10 distinct values. A random search with 50 iterations gives each dimension many more opportunities to vary across the sampled configurations.

The practical result is simple: random search usually finds near-optimal settings much faster when only some dimensions really matter.

## Implementing RandomizedSearchCV

The example below uses a Random Forest classifier.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(random_state=42)

param_dist = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(2, 30),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 15),
    "max_features": ["sqrt", "log2", None],
}

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring="accuracy",
    random_state=42,
    return_train_score=True,
    n_jobs=-1,
)

random_search.fit(X_train, y_train)

print(f"Best Parameters: {random_search.best_params_}")
print(f"Best CV Score: {random_search.best_score_:.3f}")

y_pred = random_search.best_estimator_.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

Always set `random_state`. Without it, every run samples a different set of configurations and results will not be reproducible.

## Understanding `n_iter`

`n_iter` controls how many random configurations are evaluated. It is the main exploration budget.

| Search Space Complexity              | Recommended `n_iter` |
| ------------------------------------ | -------------------- |
| 2-3 hyperparameters, narrow ranges   | 20-30                |
| 3-5 hyperparameters, moderate ranges | 50-100               |
| 5+ hyperparameters, wide ranges      | 100-200+             |
| Fast exploratory work                | 20-30                |
| Final tuning for production          | 100+                 |

One useful approach is to test several `n_iter` values and check whether the best score plateaus.

```python
cv_scores_by_iter = []
for n in [10, 25, 50, 75, 100]:
    rs = RandomizedSearchCV(
        rf,
        param_dist,
        n_iter=n,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    rs.fit(X_train, y_train)
    cv_scores_by_iter.append((n, rs.best_score_))
    print(f"n_iter={n:3d} -> Best CV: {rs.best_score_:.4f}")
```

If the score stops improving, more iterations are unlikely to help.

## Defining Parameter Distributions

The distribution you choose determines what values can be sampled.

### `randint` for discrete integers

Use `randint` for integer parameters such as `n_estimators`, `max_depth`, `min_samples_split`, and K in KNN.

```python
from scipy.stats import randint

"n_estimators": randint(50, 500)
```

### `uniform` for continuous ranges

Use `uniform` when differences in magnitude are roughly equally important across the search range.

```python
from scipy.stats import uniform

"learning_rate": uniform(0.01, 0.29)
```

### `loguniform` for regularization and scale-sensitive parameters

Use `loguniform` when values span multiple orders of magnitude, especially for regularization parameters.

```python
from scipy.stats import loguniform

"C": loguniform(1e-4, 1e2)
```

Log-uniform sampling gives each order of magnitude equal attention, which is usually the right behavior for regularization strength.

### Lists for categorical choices

Use plain lists when the parameter is categorical.

```python
"max_features": ["sqrt", "log2", None]
```

## Logistic Regression Example

When preprocessing is involved, always use a pipeline so the preprocessing step is fitted inside each CV fold.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

param_dist = {
    "model__C": loguniform(1e-4, 1e2),
    "model__penalty": ["l2"],
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_train, y_train)

print(f"Best Parameters: {random_search.best_params_}")
print(f"Best CV F1: {random_search.best_score_:.3f}")
```

## Visualizing Random Search Results

Random search creates a scattered set of sampled points rather than a full grid.

```python
import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.DataFrame(random_search.cv_results_)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    results_df["param_n_estimators"],
    results_df["mean_test_score"],
    alpha=0.6,
    c="steelblue",
)
axes[0].set_xlabel("n_estimators")
axes[0].set_ylabel("CV Accuracy")
axes[0].set_title("n_estimators vs CV Performance")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(
    results_df["param_max_depth"],
    results_df["mean_test_score"],
    alpha=0.6,
    c="darkorange",
)
axes[1].set_xlabel("max_depth")
axes[1].set_ylabel("CV Accuracy")
axes[1].set_title("max_depth vs CV Performance")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

What to look for:

- A clear trend means the parameter matters and the range may be worth extending.
- Flat scatter suggests the parameter has little effect and can likely be fixed.
- High variance at fixed values suggests interaction with other parameters.

## RandomizedSearchCV vs GridSearchCV

| Dimension                            | GridSearchCV             | RandomizedSearchCV |
| ------------------------------------ | ------------------------ | ------------------ |
| Search type                          | Exhaustive               | Random sampling    |
| High-dimensional spaces              | Poor fit                 | Good fit           |
| Continuous ranges                    | Awkward                  | Natural            |
| Guarantees best within sampled space | Yes                      | No                 |
| Best for                             | Narrow, final refinement | Broad exploration  |

The practical takeaway is simple: use random search to explore and grid search to refine.

## Hybrid Coarse-to-Fine Strategy

The best workflow often combines both methods.

Phase 1: broad exploration with RandomizedSearchCV.

```python
param_dist_coarse = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(2, 30),
    "min_samples_leaf": randint(1, 20),
    "max_features": ["sqrt", "log2", None],
}

rs = RandomizedSearchCV(
    rf,
    param_dist_coarse,
    n_iter=50,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
)

rs.fit(X_train, y_train)
print(f"Coarse best: {rs.best_params_}")
```

Phase 2: refinement with GridSearchCV.

```python
from sklearn.model_selection import GridSearchCV

param_grid_fine = {
    "n_estimators": [200, 250, 300],
    "max_depth": [6, 7, 8, 9, 10],
    "min_samples_leaf": [3, 5, 7],
}

gs = GridSearchCV(
    rf,
    param_grid_fine,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

gs.fit(X_train, y_train)
print(f"Fine best: {gs.best_params_}")
```

Phase 3: evaluate once on the test set.

```python
from sklearn.metrics import f1_score

y_pred = gs.best_estimator_.predict(X_test)
print(f"Final Test F1: {f1_score(y_test, y_pred):.3f}")
```

This combined strategy usually finds strong results with much less compute than exhaustive grid search alone.

## Avoiding Data Leakage

The same leakage rules apply here as in GridSearchCV.

- Split the data before tuning.
- Tune only on the training set.
- Evaluate the final model on the test set exactly once.
- Keep preprocessing steps inside a pipeline.

If you scale or impute outside the pipeline, the preprocessing step can see the full training set before cross-validation splits are applied, which leaks information into the validation folds.

## Reporting Results Properly

A complete report should include:

- Model name
- Search method
- Parameter distributions
- `n_iter`
- CV strategy
- Scoring metric
- Baseline score
- Untuned model score
- Tuned CV mean and standard deviation
- Final test score
- Best configuration

Example reporting style:

- Baseline F1: 0.00
- Untuned RF F1: 0.71
- RandomizedSearch CV F1: 0.82 ± 0.03
- Final Test F1: 0.80
- Best configuration: `{'n_estimators': 247, 'max_depth': 8, 'min_samples_leaf': 4, 'max_features': 'sqrt'}`

## Common Mistakes to Avoid

- Too few iterations for the search space
- Not setting `random_state`
- Using uniform sampling for regularization parameters that need log-scale search
- Leaving preprocessing outside the pipeline
- Ignoring the train/CV gap
- Comparing models with different CV strategies

## Practical Checklist Before Finalizing

- Train/test split done before tuning
- Preprocessing inside a pipeline
- `random_state` set
- `n_iter` matches the search budget
- Distribution types match parameter semantics
- `return_train_score=True` used when overfitting needs inspection
- Scoring metric matches the business objective
- Final test set evaluated exactly once
- Best params and CV mean/std reported

## Closing Reflection

RandomizedSearchCV brings scalability to hyperparameter tuning. It accepts that you do not need to check every point in a large search space to find a strong configuration.

Used on its own, it is a fast way to explore broad ranges. Used with GridSearchCV, it becomes a practical coarse-to-fine workflow that balances efficiency and precision.

When the grid grows too large, random search is usually the better default.
