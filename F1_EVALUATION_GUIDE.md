# Evaluating Classification Models Using F1-Score

After learning Precision and Recall, the next practical question is: what if both matter?

In many real classification tasks, optimizing only one of these metrics leads to failure:

- A fraud model with very high recall but too many false alarms overwhelms operations.
- A fraud model with very high precision but very low recall misses too many true fraud cases.

F1-Score is the standard way to evaluate this balance.

## What F1-Score Measures

F1-Score combines Precision and Recall using the harmonic mean:

$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

Substituting confusion-matrix terms:

$$
F1 = \frac{2TP}{2TP + FP + FN}
$$

Important implication:

- F1 depends on TP, FP, and FN
- True Negatives do not affect F1

F1 is intentionally focused on positive-class performance.

## Why Harmonic Mean (Not Arithmetic Mean)

Given two values, the harmonic mean penalizes imbalance more than arithmetic mean.

- Arithmetic mean: tolerant of one high and one low value
- Harmonic mean: collapses when either value is low

Example:

- Precision = 0.90
- Recall = 0.10

Arithmetic mean:

$$
\frac{0.90 + 0.10}{2} = 0.50
$$

F1:

$$
\frac{2 \cdot 0.90 \cdot 0.10}{0.90 + 0.10} = 0.18
$$

The arithmetic mean suggests moderate performance; F1 correctly flags near-failure.

## Intuition

- Precision asks: when the model predicts positive, how often is it correct?
- Recall asks: of all actual positives, how many are found?
- F1 asks: how well are correctness and coverage balanced?

F1 rewards models that do both well and penalizes one-sided behavior.

## Worked Example

Suppose:

- Precision = 0.80
- Recall = 0.60

$$
F1 = \frac{2 \cdot 0.80 \cdot 0.60}{0.80 + 0.60} = 0.686
$$

So F1 is approximately 0.69.

Sensitivity to imbalance:

| Precision | Recall | Arithmetic Mean | F1    |
| --------- | ------ | --------------- | ----- |
| 0.90      | 0.90   | 0.90            | 0.90  |
| 0.90      | 0.50   | 0.70            | 0.643 |
| 0.90      | 0.20   | 0.55            | 0.327 |
| 0.90      | 0.05   | 0.475           | 0.095 |

As recall drops, F1 drops quickly, reflecting real operational risk.

## F1 vs Accuracy

On imbalanced data, accuracy can be dangerously optimistic.

Example: 95% negative, 5% positive, model predicts all negatives:

- Accuracy = 95%
- Precision = 0
- Recall = 0
- F1 = 0

Accuracy looks strong while F1 correctly marks failure on the minority class.

## When to Use F1 as Primary Metric

Use F1 when:

- classes are imbalanced
- both false positives and false negatives matter
- you need one metric for model ranking and tuning

Common domains:

- fraud detection
- spam filtering
- disease detection
- intrusion detection
- churn prediction

## When F1 Is Not the Best Primary Metric

Use another metric when:

- one error type is much more costly than the other (use F-beta)
- true negatives are important (consider MCC or balanced accuracy)
- probability calibration matters (consider ROC-AUC or Brier score)
- operating threshold will change often (consider PR-AUC)

## Compute F1 in Scikit-Learn

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.3f}")
```

Always pair F1 with Precision and Recall:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, digits=3))
```

The same F1 can come from very different precision-recall trade-offs.

## Baseline Comparison

Use majority-class baseline as your floor:

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, classification_report

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print("=== Baseline ===")
print(f"F1-Score: {f1_score(y_test, baseline_pred, zero_division=0):.3f}")
print(classification_report(y_test, baseline_pred, zero_division=0))

print("=== Model ===")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

Interpret model gains against this baseline, not in isolation.

## Multi-Class F1 Variants

When there are more than two classes, specify averaging explicitly.

### Macro F1

Equal weight per class:

```python
f1_macro = f1_score(y_test, y_pred, average="macro")
```

Use when every class should matter equally, including rare classes.

### Micro F1

Global TP/FP/FN before computing F1:

```python
f1_micro = f1_score(y_test, y_pred, average="micro")
```

Use when overall prediction volume matters most. In single-label multi-class, micro F1 equals accuracy.

### Weighted F1

Per-class F1 weighted by class support:

```python
f1_weighted = f1_score(y_test, y_pred, average="weighted")
```

Use when class frequency should influence aggregate score while retaining class-level contribution.

Practical default on imbalanced multi-class tasks:

- Primary: macro F1
- Context: weighted F1
- Always inspect per-class report

## Threshold Optimization for F1

Default threshold 0.5 is often suboptimal.

```python
import numpy as np
from sklearn.metrics import f1_score

y_prob = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

print(f"Best threshold: {best_threshold:.2f}")
print(f"Best F1:        {best_f1:.3f}")
print(f"Default F1:     {f1_score(y_test, (y_prob >= 0.5).astype(int)):.3f}")
```

Critical rule: tune threshold on validation data, not on test data.

```python
from sklearn.model_selection import train_test_split

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

model.fit(X_train_sub, y_train_sub)
val_prob = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)
val_f1s = [f1_score(y_val, (val_prob >= t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(val_f1s)]

test_pred = (model.predict_proba(X_test)[:, 1] >= best_t).astype(int)
print(f"Test F1 at optimized threshold ({best_t:.2f}): {f1_score(y_test, test_pred):.3f}")
```

## Cross-Validation with F1

Binary:

```python
from sklearn.model_selection import cross_val_score

cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
print(f"CV F1 scores: {cv_f1.round(3)}")
print(f"Mean CV F1:   {cv_f1.mean():.3f} +- {cv_f1.std():.3f}")
```

Multi-class:

```python
cv_f1_macro = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
print(f"Mean CV Macro F1: {cv_f1_macro.mean():.3f} +- {cv_f1_macro.std():.3f}")
```

High fold variance is a stability warning and should be investigated.

## Common Mistakes

- Reporting F1 without Precision and Recall
- Using weighted F1 alone on heavily imbalanced data
- Not specifying average method in multi-class tasks
- Optimizing threshold on test data
- Reporting only training F1
- Treating F1 as always better than accuracy
- Ignoring baseline F1

## Practical Checklist Before Reporting F1

- F1 computed on held-out test data
- Baseline F1 computed and compared
- Precision and Recall reported separately
- Classification report reviewed
- Averaging method stated and justified
- Threshold evaluated and, if tuned, tuned on validation set
- Cross-validation mean and std reported
- Interpretation linked to business impact

## Closing

Precision is trust. Recall is coverage. F1 is balance.

F1 is not perfect. It ignores true negatives, depends on a threshold, and assumes equal importance of precision and recall. But for many imbalanced classification tasks, it is a more honest single-number summary than accuracy alone.

Use it carefully, compare it to baseline, and interpret it in business context.
