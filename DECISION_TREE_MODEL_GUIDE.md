# Decision Tree Model Guide

So far, you've worked with models that rely on smooth mathematical relationships:

- **Linear models** (Linear Regression, Logistic Regression) — fit hyperplanes through data
- **Distance-based models** (KNN) — measure similarity via geometric distance

Decision Trees take a completely different approach. They don't fit lines. They don't measure distances. They don't assume anything about the shape of the relationship between features and target.

Instead, they learn by repeatedly asking one kind of question:

**"Is feature X less than value Y?"**

Through a sequence of binary questions, a Decision Tree partitions the feature space into rectangular regions and assigns a prediction to each region. The result is a model that can be read like a flowchart — traversing from the root node through a series of yes/no decisions until you reach a leaf that gives the prediction.

## Why Decision Trees Matter

Decision Trees have several properties that no linear or distance-based model can match:

- **Naturally non-linear** — each split can capture local patterns
- **Handle feature interactions automatically** — a split at node A can create context for a subsequent split at node B
- **Require no feature scaling** — splits are based on ranked order, not magnitude
- **Produce human-readable rules** — every prediction traces a path you can explain in plain language

However, they come with a serious vulnerability: without constraints, they grow until every training sample occupies its own leaf — achieving 100% training accuracy and catastrophic overfitting.

Decision Trees are foundational. Random Forests and Gradient Boosting — two of the most powerful ML families in tabular machine learning — are built directly on them.

## How a Decision Tree Works

A Decision Tree builds its model by recursively partitioning the training dataset into smaller and smaller subsets.

The algorithm is greedy: at each step, it scans every possible split (every feature, every threshold) and selects the one that produces the greatest improvement in **purity** — the degree to which each resulting subset is dominated by a single class (classification) or has low target variance (regression).

### Classification Example

At each node, the algorithm asks questions like:

- Is Tenure ≤ 12 months?
- Is MonthlyCharges > 80?
- Is ContractType = "Month-to-month"?

It chooses whichever question best separates churners from non-churners. The resulting path reads as a rule:

```
IF Tenure ≤ 12 AND MonthlyCharges > 80 → Predict Churn
```

### Regression Example

The algorithm splits the data to minimize the variance of target values in each child node. Each leaf predicts the mean target value of all training samples that fall into that region.

The process continues recursively until a stopping condition is reached.

## Impurity Measures for Classification

The choice of split at each node is governed by an **impurity measure** — a function that quantifies how mixed the classes are in a node. The best split is the one that maximally reduces impurity after the partition.

### Gini Impurity (Default in Scikit-Learn)

$$
\text{Gini} = 1 - \sum_{c} p_c^2
$$

Where $p_c$ is the proportion of class $c$ in the node.

- Gini = 0 → Perfectly pure node (every sample is one class)
- Gini = 0.5 → Maximum impurity in binary classification (50/50 split)

**Example:** A node with 60% class 0 and 40% class 1:

$$
\text{Gini} = 1 - (0.6^2 + 0.4^2) = 1 - 0.52 = 0.48
$$

After splitting into:
- Left child: 80% class 0, 20% class 1 → Gini = 0.32
- Right child: 20% class 0, 80% class 1 → Gini = 0.32

The weighted average impurity after the split is lower than 0.48 — the split is worthwhile.

### Entropy and Information Gain

$$
\text{Information Gain} = \text{Entropy before} - \text{Weighted avg entropy after}
$$

$$
\text{Entropy} = -\sum_{c} p_c \log_2(p_c)
$$

- Entropy = 0 → Pure node
- Entropy = 1.0 → Maximum disorder (50/50 split)

**In practice:** Gini and Entropy produce nearly identical trees. Gini is slightly faster (no logarithm). Entropy can produce marginally more balanced splits. The choice rarely matters enough to optimize.

```python
from sklearn.tree import DecisionTreeClassifier

tree_gini    = DecisionTreeClassifier(criterion="gini",    max_depth=4, random_state=42)
tree_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
```

### Impurity Measure for Regression

For regression trees, impurity is measured by the variance of target values within each node:

$$
\text{Variance} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \bar{y})^2
$$

The algorithm selects the split that minimizes the weighted average variance across the two child nodes — equivalent to minimizing MSE.

Each leaf predicts the mean target value of its training samples. **Key limitation:** regression trees can never extrapolate beyond the range of observed targets.

## The Tree Growth Algorithm

The full recursive partitioning process:

1. Start with the complete training set at the root node
2. Evaluate every possible (feature, threshold) split
3. Select the split that maximally reduces impurity (or variance)
4. Create two child nodes: samples where the condition is true go left; others go right
5. Recursively repeat for each child node
6. Stop when a stopping criterion is satisfied

### Default Stopping Criteria

- **max_depth** — maximum number of levels below the root
- **min_samples_split** — minimum samples required to split a node
- **min_samples_leaf** — minimum samples required in a leaf node
- Node is already pure (Gini = 0)
- No split improves impurity

**Critical warning:** Without any stopping criteria, the tree grows until every leaf contains exactly one training sample. Training accuracy reaches 100%. Test accuracy collapses.

## Worked Example: Classification

Predicting customer churn with 100 training samples:

**Root node:** 100 samples (60 No Churn, 40 Churn)

**Split on Tenure ≤ 12 months:**

| Node | Samples | No Churn | Churn | Gini |
| --- | --- | --- | --- | --- |
| Left (Tenure ≤ 12) | 30 | 5 | 25 | 0.333 |
| Right (Tenure > 12) | 70 | 55 | 15 | 0.430 |

**Weighted impurity after split:**

$$
0.30 \times 0.333 + 0.70 \times 0.430 = 0.399
$$

**Impurity reduction:** $0.48 - 0.399 = 0.081$ — a meaningful improvement. The tree accepts this split.

The left child now has 83% churn rate — predominantly churners. The right child has 79% no-churn rate. The split has meaningfully separated the two classes. The tree continues growing each child recursively.

## Overfitting in Decision Trees

An unconstrained decision tree on any real dataset will memorize the training data:

```
Train Accuracy: 100%
Test Accuracy:  68%
Train/Test Gap: 32%
```

This is high variance in its most extreme form. Without constraints, the tree creates one leaf per training sample (or close to it), perfectly fitting every pattern — and every noise artifact — in the training set.

**Why does this happen uniquely badly for trees?** Unlike regularized linear models where overfitting is gradual, a decision tree can go from "reasonable" to "fully memorized" simply by removing the max_depth constraint.

## Controlling Tree Complexity

The four most important hyperparameters, in rough order of impact:

| Parameter | Effect of Increasing | Bias | Variance |
| --- | --- | --- | --- |
| max_depth | More levels, finer partitions | ↓ | ↑ |
| min_samples_split | Harder to split a node | ↑ | ↓ |
| min_samples_leaf | Larger minimum leaf size | ↑ | ↓ |
| max_features | Fewer features considered per split | ↑ (slightly) | ↓ |

These are all expressions of the same underlying trade-off: more constraints → higher bias, lower variance. Less constraints → lower bias, higher variance.

The right combination depends on dataset size and noise level. Small, noisy datasets need more aggressive constraints. Large, clean datasets can tolerate deeper trees.

## Cross-Validation for Depth Selection

Never choose max_depth by guessing or by looking at test accuracy. Use cross-validation:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

param_grid = {"max_depth": range(1, 21)}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    return_train_score=True
)
grid.fit(X_train, y_train)

print(f"Best depth:    {grid.best_params_['max_depth']}")
print(f"Best CV score: {grid.best_score_:.3f}")

# Visualize the bias–variance trade-off across depths
depths       = range(1, 21)
cv_scores    = grid.cv_results_["mean_test_score"]
train_scores = grid.cv_results_["mean_train_score"]

plt.figure(figsize=(9, 5))
plt.plot(depths, train_scores, label="Train Accuracy", linestyle="--", marker="o")
plt.plot(depths, cv_scores, label="CV Accuracy", marker="o")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Depth vs. Accuracy")
plt.legend()
plt.grid(True)
plt.show()
```

The plot will show the classic pattern:

- At shallow depths: both train and CV accuracy are low (underfitting / high bias)
- As depth increases: train accuracy rises toward 1.0, CV accuracy peaks then declines
- The optimal depth is **where CV accuracy peaks before beginning to fall**

This is the bias-variance trade-off made directly visible.

## Implementing Decision Tree: Classification

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print(f"Train Accuracy: {tree.score(X_train, y_train):.3f}")
print(f"Test Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Train/Test Gap: {tree.score(X_train, y_train) - accuracy_score(y_test, y_pred):.3f}")
print()
print(classification_report(y_test, y_pred))

# Cross-validation stability check
cv_scores = cross_val_score(tree, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

**Always print the train/test gap.** A gap of 2–5% is typical and acceptable. A gap of 20%+ means the tree is still overfitting — reduce max_depth or increase min_samples_leaf.

## Visualizing the Tree

```python
plt.figure(figsize=(16, 8))
plot_tree(
    tree,
    feature_names=X.columns.tolist(),
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree — Churn Prediction")
plt.tight_layout()
plt.show()
```

The visualization reveals:

- **Splitting features and thresholds** at each node — the actual questions the model asks
- **Sample counts** in each node — how much data supports each decision
- **Class distributions** — the purity of each node (darker fill = purer node)
- **Gini impurity** at each node — the residual uncertainty at that decision point

This is one of the most powerful interpretability tools in ML. You can hand the printed tree to a domain expert and ask: "Do these rules make sense?" Their answer is valuable diagnostic information.

If the tree is splitting on nonsensical features or in counterintuitive directions, that reveals problems with the data or the model that metrics alone would miss.

## Feature Importance

For large trees, feature importance gives a more compact summary:

```python
import pandas as pd

importance_df = pd.DataFrame({
    "Feature":    X.columns,
    "Importance": tree.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance_df.to_string(index=False))
```

**Feature importance** in decision trees measures the total impurity reduction attributed to each feature, weighted by the number of samples at each node where it was used.

It is not a perfect measure of causality, but it reliably identifies which features the model is relying on most.

## Decision Tree for Regression

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

tree_reg = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
)
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

# Compare to mean baseline
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_r2 = r2_score(y_test, baseline.predict(X_test))

print(f"Tree RMSE:      {rmse:.3f}")
print(f"Tree R²:        {r2:.3f}")
print(f"Baseline R²:    {baseline_r2:.3f}")
print(f"Train R²:       {tree_reg.score(X_train, y_train):.3f}")
print(f"Train/Test gap: {tree_reg.score(X_train, y_train) - r2:.3f}")
```

**Regression tree limitation:** Because each leaf predicts a constant value (the mean of its training samples), regression trees produce step-function predictions. They cannot extrapolate — if you ask for a prediction beyond the range of training values, the tree returns the mean of the nearest leaf, bounded by the training distribution.

For problems requiring smooth interpolation or extrapolation, this is a fundamental limitation.

## Comparing Against Baseline

```python
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
tree_acc     = accuracy_score(y_test, y_pred)

print(f"Baseline Accuracy: {baseline_acc:.3f}")
print(f"Tree Accuracy:     {tree_acc:.3f}")
print(f"Improvement:       +{tree_acc - baseline_acc:.3f}")
```

If a decision tree cannot beat the majority-class baseline, investigate immediately. Likely causes:

- max_depth is set too low (high bias)
- Features are not informative
- Target is highly imbalanced without class weighting
- Data pipeline bug

## Strengths of Decision Trees

- **No feature scaling required** — splits are based on feature rank order, not magnitude
- **Non-linear relationships handled naturally** — each split can capture a local pattern
- **Automatic interaction detection** — a split on Tenure creates context for a subsequent split on MonthlyCharges
- **Highly interpretable** — the entire model visualizes as a flowchart
- **Handles mixed feature types** — numerical and categorical features coexist
- **Robust to outliers** — a split threshold is determined by relative order

## Weaknesses of Decision Trees

- **Prone to overfitting** — the most significant limitation; unconstrained trees are extreme high-variance learners
- **High instability** — small changes in training data produce dramatically different trees
- **Poor extrapolation in regression** — cannot predict beyond the range of training targets
- **Axis-aligned splits only** — decision boundaries are perpendicular to feature axes
- **Often outperformed by ensembles** — Random Forests and Gradient Boosting almost always outperform a single tree

## Common Mistakes to Avoid

### Leaving max_depth Unconstrained

A single line of code (max_depth=None, the default) produces a fully memorized tree. **Always set a depth limit and validate it with cross-validation.**

### Not Printing the Train/Test Gap

Reporting only test accuracy hides overfitting. **Always print both.** A 30-point train/test gap tells you more about the model than the test accuracy alone.

### Interpreting Feature Importance Without Domain Validation

Feature importance tells you what the model used, not necessarily what's causally important. High importance for a correlated proxy feature can mislead. **Always sanity-check importance rankings against domain knowledge.**

### Not Comparing to Baseline

A decision tree at max_depth=4 on an imbalanced dataset might achieve 88% accuracy — which sounds good until you realize the majority-class baseline achieves 86%. The tree has barely learned anything useful.

### Treating a Single Decision Tree as a Final Model

Decision trees are foundational, but in practice they are almost always used as a stepping stone to Random Forests or Gradient Boosting. If you find yourself tuning a single tree extensively for production, reconsider whether an ensemble would serve better.

## Practical Checklist Before Reporting Results

- Train/test split done correctly with stratification for classification
- Train accuracy and test accuracy both computed — gap assessed
- max_depth (and min_samples_leaf) tuned via cross-validation
- Depth vs. accuracy curve plotted to understand the bias-variance trade-off
- Baseline computed and tree clearly outperforms it
- Confusion matrix inspected for classification tasks
- Feature importances reviewed and validated against domain knowledge
- Cross-validation stability checked — mean and std reported
- Tree visualized at the chosen depth — rules sanity-checked

## When Decision Trees Work Well

- Structured tabular data with clear, local decision boundaries
- Problems where interpretability is a hard requirement
- Feature engineering discovery — trees reveal which features drive predictions
- Moderate-sized datasets where ensemble methods may be overkill
- Mixed-type feature spaces where scaling would be cumbersome

## When Decision Trees Struggle

- High-variance, noisy datasets where the greedy algorithm overfits aggressively
- High-dimensional sparse data where axis-aligned splits are inefficient
- Smooth continuous relationships where step-function predictions underperform (regression)
- Very small datasets where the tree can't develop stable splits
- Any problem where accuracy is primary and interpretability is not required — an ensemble will outperform

## Closing Reflection

Decision Trees are powerful because they:

- Require almost no preprocessing
- Capture non-linearity naturally
- Produce human-readable rules

But they are high-variance learners. Without constraints, they overfit catastrophically. With proper tuning — max_depth, min_samples_leaf, validated by cross-validation — they generalize well and serve as both a strong standalone model and an invaluable diagnostic tool.

They are rarely the final model in production. But they are the foundation of Random Forests and Gradient Boosting, the algorithms that dominate modern tabular machine learning.

**Understanding Decision Trees deeply means understanding the backbone of the most widely deployed ML methods in the world.**

**Train carefully.** **Control complexity.** **Visualize the tree.** **Always compare against baseline.**

## Bonus References

- Scikit-learn DecisionTreeClassifier API Reference
- Scikit-learn Decision Trees User Guide
- Scikit-learn plot_tree Visualization Reference
