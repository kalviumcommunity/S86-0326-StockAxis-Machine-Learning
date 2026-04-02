# Problem Type Decision Tree & Reference Guide

## Quick Decision: What is Your Problem?

```
START: I need to build a predictive model

    ↓

Q1: What am I predicting?
    │
    ├─→ A CATEGORY (Yes/No, Category from fixed set)
    │   ↓
    │   → CLASSIFICATION
    │
    │   ├─ Q2a: How many classes?
    │   │   ├─→ Exactly 2 → BINARY CLASSIFICATION
    │   │   └─→ 3+ → MULTI-CLASS CLASSIFICATION
    │   │
    │   └─ Q2b: Can multiple apply simultaneously?
    │       ├─→ Yes → MULTI-LABEL CLASSIFICATION
    │       └─→ No → STANDARD CLASSIFICATION
    │
    └─→ A NUMBER (continuous value)
        ↓
        → REGRESSION

        ├─ Q2c: What's the range?
        │   ├─→ Unbounded continuous → LINEAR REGRESSION
        │   ├─→ Non-negative integers → COUNT REGRESSION
        │   └─→ Bounded (0-100%) → BOUNDED REGRESSION
        │
        └─ Q2d: Is relationship linear?
            ├─→ Yes → Linear Regression, Ridge, Lasso
            └─→ No → Polynomial, Tree-based, Neural Networks
```

---

## 🎯 StockAxis Specific Path

### Your Project Status: ✓ CLASSIFICATION IDENTIFIED

```
Your Algorithm: RandomForestClassifier
Your Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
Your Framework: RIGHT FOR CLASSIFICATION
```

### Your Next Steps

**TO DETERMINE YOUR EXACT PROBLEM TYPE, CHECK YOUR DATA:**

```python
import pandas as pd

# Load your actual dataset
df = pd.read_csv('data/raw/dataset.csv')

# 1. IDENTIFY TARGET VARIABLE
print("Step 1: Target Variable Analysis")
print(f"Column name: 'target'")
print(f"Data type: {df['target'].dtype}")
print(f"Unique values: {df['target'].unique()}")
print(f"Number of classes: {len(df['target'].unique())}")

# 2. UNDERSTAND DISTRIBUTION
print("\nStep 2: Class Distribution")
print(df['target'].value_counts())
print("\nProportions:")
print(df['target'].value_counts(normalize=True))

# 3. CHECK FOR IMBALANCE
counts = df['target'].value_counts()
imbalance_ratio = counts.max() / counts.min()
print(f"\nImbalance ratio: {imbalance_ratio:.2f}x")
if imbalance_ratio > 3:
    print("⚠️  WARNING: Class imbalance detected!")
    print("   ACTION: Add class_weight='balanced' to RandomForestClassifier")

# 4. DEFINE YOUR SUCCESS METRIC
print("\nStep 3: Success Metric")
print("Questions to answer:")
print("  - How costly are false positives?")
print("  - How costly are false negatives?")
print("  - Do you need class rankings or final predictions?")
```

---

## Decision Matrix: Which Metric to Optimize?

| Scenario                             | Recommended Metric         | Reasoning                              |
| ------------------------------------ | -------------------------- | -------------------------------------- |
| **Both classes equally important**   | Accuracy, F1               | Balanced approach                      |
| **Minimize false positives**         | Precision                  | Important when false alarms are costly |
| **Minimize false negatives**         | Recall                     | Important when missed cases are costly |
| **Catch rare positive class**        | ROC-AUC                    | For imbalanced classification          |
| **Need ranking, not binary**         | ROC-AUC, Rank metrics      | For scoring/ranking decisions          |
| **Both precision AND recall matter** | F1 Score, Weighted metrics | Balance both concerns                  |

### Examples for StockAxis

If predicting **stock direction (UP/DOWN)**:

- Symmetric importance → **F1 Score**
- Want to catch all UPs → **Recall**
- Want to avoid false UPs → **Precision**

---

## Implementation Checklist

### Phase 1: Problem Definition (DO FIRST)

- [ ] Load dataset and examine target variable
- [ ] Determine: Binary or Multi-class?
- [ ] Calculate class distribution
- [ ] Check imbalance ratio
- [ ] Document your business success metric
- [ ] Select optimization metric
- [ ] **Document findings in this format:**

```
PROBLEM DEFINITION: StockAxis
================================
Target Variable: [your target name]
Data Type: [int/object/float]
Number of Classes: [2 or 3+]
Classes: {list them}
Distribution: [percentages]
Imbalance Ratio: [x.xx]
Success Metric: [which one?]
```

### Phase 2: Model Configuration

- [ ] If imbalanced (>3x): Update config or add `class_weight='balanced'`
- [ ] Confirm evaluation metrics in `src/evaluate.py`
- [ ] Run baseline model
- [ ] Validate metrics computation

### Phase 3: Evaluation & Iteration

- [ ] Generate classification report (per-class metrics)
- [ ] Create confusion matrix visualization
- [ ] Compare metrics to success threshold
- [ ] Document findings

---

## Metric Reference Quick Guide

### Classification Metrics Explained

**Accuracy** = (TP + TN) / Total

- Simple but misleading with imbalance
- Use when: Classes balanced

**Precision** = TP / (TP + FP)

- False positives costly (avoid too many alarms)
- Answer: "Of predictions I make, how many correct?"

**Recall** = TP / (TP + FN)

- False negatives costly (don't miss important cases)
- Answer: "Of actual positives, how many do I catch?"

**F1** = Harmonic mean of Precision & Recall

- Best single metric for imbalanced data
- Balances both concerns

**ROC-AUC** = Area under curve (0 to 1)

- 0.5 = random, 1.0 = perfect
- Use for threshold-independent comparison
- Great for imbalanced problems

### Regression Metrics (If Applicable)

**MAE** = Average |Prediction - Actual|

- Interpretable in same units as target
- Robust to outliers

**RMSE** = √(average of squared errors)

- Penalizes large errors more
- Same units as target

**R²** = Fraction of variance explained

- 0 = no better than mean
- 1 = perfect predictions
- Bounded 0-1, easy to interpret

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Using Accuracy for Imbalanced Classification

```python
# WRONG
model = RandomForestClassifier()
metric = accuracy_score(y_test, predictions)
print(f"Accuracy: {metric}")  # Misleading if imbalanced!

# RIGHT
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))  # Shows per-class metrics
```

### ❌ Mistake 2: Ignoring Class Imbalance

```python
# WRONG
model = RandomForestClassifier(n_estimators=100)

# RIGHT (if imbalanced)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
```

### ❌ Mistake 3: Optimizing Wrong Metric

```python
# WRONG
# Minimizing false negatives but optimizing precision only

# RIGHT
from sklearn.metrics import recall_score
recall = recall_score(y_test, predictions)  # Minimize false negatives
```

---

## Verification Checklist for StockAxis

Before claiming your model is ready:

- [ ] Target variable is clearly categorical (not continuous)
- [ ] All classes have at least 1-2 samples
- [ ] Class distribution documented
- [ ] Appropriate metrics chosen based on business goal
- [ ] Model uses `RandomForestClassifier` ✓ (already correct)
- [ ] Evaluation computes Precision, Recall, F1 ✓ (already correct)
- [ ] ROC-AUC computed correctly (binary or multi-class) ✓ (already correct)
- [ ] Confusion matrix generated (for visualization)
- [ ] Results compared to success threshold
- [ ] Everything documented for reproducibility

---

## Resources Within Your Project

### 📄 Documentation

- **PROBLEM_DEFINITION.md** — Full problem type analysis for StockAxis
- **supervised_learning_problem_types.ipynb** — Complete lesson with examples

### 🔧 Code to Examine

- **src/train.py** — RandomForestClassifier implementation
- **src/evaluate.py** — Metric computation
- **src/config.py** — Configuration management

### 🚀 Next Actions

1. Run the analysis code in the notebook (fill in your target)
2. Update PROBLEM_DEFINITION.md with your specific findings
3. Adjust config.py if imbalance detected
4. Train your first model with full understanding of the problem

---

## When You're Stuck

### "I don't know if this is classification or regression"

→ **Examine the target variable.** If it's categories or has few unique values → Classification. If it's a number with many unique values → Regression.

### "My classes are very imbalanced"

→ **Add `class_weight='balanced'`** to RandomForestClassifier. Use ROC-AUC and F1, not Accuracy.

### "I don't know which metric to prioritize"

→ **Ask the business:** "Is it worse to have false positives or false negatives?" Answer determines whether to optimize Precision or Recall.

### "My model seems too good to be true"

→ **Check for class imbalance.** A model that always predicts majority class can have very high accuracy without learning anything.

---

## Final Validation Question

**Before you train your model, answer this in one sentence:**

"My model will predict: **************\_\_\_**************"

Example answers:

- ✓ "My model will predict whether a stock will move UP or DOWN tomorrow"
- ✓ "My model will predict the market regime (BULL, NEUTRAL, or BEAR)"
- ✓ "My model will classify stock price as BULLISH or BEARISH"

If you can't answer clearly → Go back to "Problem Definition" phase.

---

**Remember**: Problem Type Identification → Metric Selection → Algorithm Choice → Training → Evaluation

Get step 1 right. Everything follows from that.
