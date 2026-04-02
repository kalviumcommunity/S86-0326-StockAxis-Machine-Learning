# StockAxis: Problem Type Analysis & Classification Framework

## Executive Summary

StockAxis is a **supervised learning classification project** that predicts categorical outcomes based on quantitative stock market features. This document applies the foundational concepts of supervised learning problem types to the StockAxis architecture and validates that the current approach aligns with ML best practices.

---

## 1. Problem Type Identification

### The Fundamental Question: What are we predicting?

**Answer**: A discrete category from a fixed set of possible classes.

**Problem Type**: **Classification**

**Classification Subtype**: Currently configured for either **Binary or Multi-class Classification** (determined by the target variable structure in your data)

**Evidence from Architecture**:

- Model choice: `RandomForestClassifier` (classification algorithm)
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC (classification metrics)
- Probability outputs: Model uses `predict_proba()` for probability distributions
- No continuous value prediction

---

## 2. Target Variable Analysis

### What are we predicting?

Based on your project configuration, the **target column** is stored in `config.py`:

```python
target_column: str = "target"
```

**Current Status**: The target variable name is abstracted as "target" and will be determined by your actual dataset.

### Determining Classification Subtype

To identify whether StockAxis is **Binary** or **Multi-class** classification, examine your target variable:

#### Binary Classification (2 classes)

```python
# Example: If predicting stock direction
df['target'].unique()
→ array([0, 1])  # or ['Down', 'Up']
```

**Characteristics**:

- Exactly 2 possible outcomes
- Model outputs single probability (probability of positive class)
- ROC-AUC is natural evaluation metric
- Positive/Negative class distinction is meaningful

#### Multi-class Classification (3+ classes)

```python
# Example: If predicting market regime
df['target'].unique()
→ array([0, 1, 2])  # or ['Bear', 'Neutral', 'Bull']
```

**Characteristics**:

- Three or more mutually exclusive outcomes
- Model outputs probability distribution over all classes (sums to 1.0)
- Must use weighted/macro/micro averaging for metrics
- ROC-AUC computed with multi_class='ovr' (one-vs-rest)

### Your Evaluation Code Handles Both Cases

Your `evaluate.py` is already designed for flexibility:

```python
if probabilities.shape[1] == 2:
    metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
else:
    metrics["roc_auc"] = float(
        roc_auc_score(y_test, probabilities, multi_class="ovr", average="weighted")
    )
```

**This is correct**: Automatically detects binary vs multi-class and applies appropriate ROC-AUC computation.

---

## 3. Problem Specification Framework

Apply these four questions to document your specific StockAxis problem:

### Question 1: What am I predicting?

**Frame it in business terms first**:

- "I want to predict [__________]"
- Examples:
  - "I want to predict if a stock will move up or down"
  - "I want to predict market regime (bull, bear, sideways)"
  - "I want to predict stock price action category"

**Then verify it's classification**:

- ✓ Output is a category (discrete label)
- ✓ No continuous magnitude (not a price or percentage return)
- ✓ Finite set of possibilities
- ✓ Probability distribution makes sense

### Question 2: How many possible outcomes?

**Document your class distribution**:

```python
# Add this analysis to your notebooks
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True))  # proportions
```

Examples:

- **Binary (2 classes)**: 57% Class A, 43% Class B
- **Multi-class (3+ classes)**: 40% Class 1, 35% Class 2, 25% Class 3

### Question 3: What does the target variable look like in data?

**Verify it's classification**:

```python
print(df['target'].dtype)           # Look for object/string or int
print(df['target'].unique())        # Should be categorical, not continuous floats
print(len(df['target'].unique()))   # Count of distinct values
```

**This should show**:

- Dtype: object (string) or int (encoded categories) — NOT float
- Unique values: Small number (2-10) — NOT hundreds of different values
- No NaN values in target during training

### Question 4: What does success look like?

**Define your business metrics**:

| Criterion                     | Your Value      | Rationale                |
| ----------------------------- | --------------- | ------------------------ |
| **Accuracy required**         | \_\_\_ %        | Overall correctness      |
| **Precision priority**        | High/Medium/Low | Cost of false positives  |
| **Recall priority**           | High/Medium/Low | Cost of false negatives  |
| **ROC-AUC requirement**       | \_\_\_ (0.7+)   | Discrimination ability   |
| **Class imbalance tolerance** | Yes/No          | Weighted metrics needed? |

---

## 4. Current Architecture Alignment

### Algorithm Choice: Why Random Forest?

Your project uses `RandomForestClassifier`. This is appropriate for classification because:

| Criterion            | Random Forest          | Why Good for StockAxis        |
| -------------------- | ---------------------- | ----------------------------- |
| Problem Type         | Binary & Multi-class ✓ | Handles both naturally        |
| Non-linearity        | Yes ✓                  | Stock patterns are non-linear |
| Feature Interactions | Captures ✓             | Market correlations matter    |
| Interpretability     | Feature Importance ✓   | Can explain decisions         |
| Robustness           | Ensemble ✓             | Reduces overfitting           |
| Imbalanced Classes   | Can handle ✓           | With class_weight tuning      |

### Evaluation Metrics: Appropriate for Classification

Your `evaluate.py` computes:

```python
metrics = {
    "accuracy": accuracy_score(...),           # Overall correctness
    "precision": precision_score(...),         # False positive rate control
    "recall": recall_score(...),               # False negative rate control
    "f1": f1_score(...),                       # Balanced metric
    "roc_auc": roc_auc_score(...)              # Threshold-independent discrimination
}
```

**Assessment**: ✓ These are exactly the right metrics for classification.

**Note**: Using `average="weighted"` is correct for:

- Binary classification (weights don't affect binary)
- Multi-class with potential class imbalance

---

## 5. Common Pitfalls: StockAxis Specific

### Pitfall 1: Treating Classification as Regression ❌

**Wrong**: Encoding stock movements as continuous values (0.0, 0.5, 1.0)

**Why it's wrong**: Implies ordering and magnitude that don't exist. Movement is categorical.

**Status**: ✓ Your project avoids this — clearly using classification framework.

---

### Pitfall 2: Ignoring Class Imbalance ❌

**Risk**: If your stock classes are imbalanced (e.g., 80% "Up", 20% "Down"):

- Accuracy alone is misleading
- Model that always predicts majority class gets 80% accuracy
- Minority class predictions fail

**Your current approach**:

```python
precision_score(..., average="weighted")  # Accounts for imbalance
recall_score(..., average="weighted")     # Accounts for imbalance
```

**Recommendation**:

- [ ] Analyze class distribution in `config.py`
- [ ] If imbalanced (>70/30 split), add `class_weight='balanced'` to RandomForestClassifier
- [ ] Monitor precision and recall separately, not just accuracy

---

### Pitfall 3: Misinterpreting ROC-AUC ❌

**Wrong**: "ROC-AUC of 0.6 is bad, my model is failing."

**Why it's wrong**: ROC-AUC measures ranking quality, not classification correctness. Context matters:

- 0.5 = no discrimination (random guessing)
- 0.7+ = good discrimination
- 0.9+ = excellent discrimination

**Your implementation**: ✓ Correctly computes ROC-AUC (binary) and multi-class ROC-AUC.

---

### Pitfall 4: Threshold Selection (Binary Only) ⚠️

**Issue**: Default classification threshold is 0.5, but may not be optimal.

```python
# This is what your model does internally:
if probability_class_1 > 0.5:
    predict_class_1
else:
    predict_class_0
```

**When 0.5 is wrong**:

- If false positives are costly, raise threshold to 0.6 or 0.7
- If false negatives are costly, lower threshold to 0.3 or 0.4

**Your current approach**: Uses default 0.5 threshold

**Recommendation**: If problem has asymmetric costs, consider threshold optimization.

---

## 6. Feature + Target Validation Checklist

Before training any model, verify:

```python
# TODO: Add to your data validation pipeline

# 1. Target variable checks
assert target_column in df.columns, "Target column not found"
assert len(df[target_column].unique()) >= 2, "Need at least 2 classes"
assert len(df[target_column].unique()) <= 50, "Too many classes (likely data error)"
assert not df[target_column].isnull().any(), "Target has missing values"

# 2. Features checks
assert not df[features].isnull().all().any(), "Feature(s) all null"
assert len(df) > len(df[features].columns), "More features than samples"

# 3. Class distribution checks
class_dist = df[target_column].value_counts()
min_class_freq = class_dist.min()
max_class_freq = class_dist.max()
assert min_class_freq >= 1, "Class with only 1 sample (unreliable)"
# Note: Severe imbalance (>90/10) requires special handling

print(f"Target distribution:\n{class_dist / len(df)}")  # Proportions
```

---

## 7. Problem Type Definition for StockAxis

### Complete Problem Statement (Template)

**Fill in the blanks for your specific data**:

```
PROBLEM TYPE: Classification
SUBTYPE: Binary / Multi-class [choose one after analyzing target]
TARGET VARIABLE: [e.g., "stock_direction" or "market_regime"]
NUMBER OF CLASSES: [e.g., 2 or 3+]

INPUT FEATURES: [list examples of features your model uses]
  - Examples: indicators, price ratios, volatility measures, etc.

OUTPUT SPACE: {all possible class values}
  - Examples: {0, 1} or {"Bull", "Bear", "Neutral"}

SUCCESS METRIC (in priority order):
  1. [Primary metric, e.g., "Recall ≥ 90%"]
  2. [Secondary metric, e.g., "Precision ≥ 80%"]
  3. [Tertiary metric, e.g., "ROC-AUC ≥ 0.75"]

CLASS IMBALANCE CONTEXT:
  - Most common class: ___% of data
  - Least common class: ___% of data
  - Handling strategy: [weight classes / oversample / undersample / accept imbalance]

ALGORITHM JUSTIFICATION:
  - Random Forest chosen because: [non-linearity / interpretability / robustness / etc.]
```

---

## 8. Practical Exercises: Apply to StockAxis

### Exercise 1: Verify Problem Type

```python
# Run this in a notebook to confirm classification problem

import pandas as pd
from pathlib import Path

# Load your data
df = pd.read_csv(Path("data/raw/dataset.csv"))

# Check target variable
print("=== Target Variable Analysis ===")
print(f"Data type: {df['target'].dtype}")
print(f"Unique values: {df['target'].unique()}")
print(f"Number of classes: {len(df['target'].unique())}")
print(f"\nClass distribution:")
print(df['target'].value_counts())
print(f"\nClass proportions:")
print(df['target'].value_counts(normalize=True))

# ✓ EXPECTED:
#   - dtype should be 'object' or 'int64', NOT 'float64'
#   - unique values: 2-10 categories, NOT hundreds
#   - proportions: reasonable for your business case
```

### Exercise 2: Confirm Metrics Are Appropriate

```python
# Check evaluation metrics

from sklearn.metrics import classification_report

# After training, generate classification report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Class1', 'Class2']))

# This should show:
# - Per-class precision, recall, f1
# - Support (number of samples per class)
# - Macro and weighted averages
```

### Exercise 3: Detect Class Imbalance Issues

```python
# Analyze if class imbalance requires special handling

class_dist = df['target'].value_counts()
max_prop = class_dist.max() / len(df)
min_prop = class_dist.min() / len(df)
imbalance_ratio = max_prop / min_prop

print(f"Most common class: {max_prop:.1%}")
print(f"Least common class: {min_prop:.1%}")
print(f"Imbalance ratio: {imbalance_ratio:.1f}x")

# ⚠️ If imbalance_ratio > 3:
#    Action: Update RandomForestClassifier to use class_weight='balanced'
```

---

## 9. Next Steps for StockAxis

### Phase 1: Problem Validation (DO THIS FIRST)

- [ ] Verify target variable is categorical (run Exercise 1)
- [ ] Determine if binary or multi-class
- [ ] Analyze class distribution (run Exercise 3)
- [ ] Document business success criteria (fill template in Section 7)

### Phase 2: Algorithm Tuning

- [ ] If imbalanced: Add `class_weight='balanced'` to RandomForestClassifier
- [ ] If binary + threshold-sensitive: Implement threshold optimization
- [ ] Cross-validate to confirm metrics are stable

### Phase 3: Evaluation & Deployment

- [ ] Report all metrics (accuracy, precision, recall, F1, ROC-AUC)
- [ ] Create confusion matrix visualization
- [ ] Document final problem definition (Section 7 template)

---

## 10. Key Takeaways for StockAxis

| Concept                 | StockAxis Application                          | Status      |
| ----------------------- | ---------------------------------------------- | ----------- |
| **Supervised Learning** | ✓ Labeled historical data + target predictions | Implemented |
| **Problem Type**        | ✓ Classification (not regression)              | Implemented |
| **Subtype**             | ? Binary or Multi-class (verify with data)     | **TODO**    |
| **Algorithm**           | ✓ RandomForestClassifier appropriate           | Implemented |
| **Metrics**             | ✓ Accuracy, Precision, Recall, F1, ROC-AUC     | Implemented |
| **Class Imbalance**     | ? Depends on data distribution                 | **TODO**    |
| **Success Definition**  | ? Specify business criteria                    | **TODO**    |

---

## Conclusion

StockAxis is correctly architected as a **supervised learning classification system**. Your choice of Random Forest and evaluation metrics aligns with classification best practices.

**Critical next step**: Analyze your actual data to determine:

1. Whether you have binary or multi-class classification
2. Class distribution and imbalance level
3. Specific business success criteria

Once these are confirmed, the foundation is solid for model training, evaluation, and deployment.

**Reference**: See the educational lesson "Understanding Supervised Learning Problem Types" for complete framework and examples.
