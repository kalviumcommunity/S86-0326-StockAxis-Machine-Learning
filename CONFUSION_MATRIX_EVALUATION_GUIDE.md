# Creating and Interpreting a Confusion Matrix

After training a classifier and computing Accuracy, Precision, Recall, and F1, one artifact still provides the most complete view of model behavior:

the confusion matrix.

Aggregate metrics are useful summaries, but they compress information. A strong single score can hide severe localized failure. The confusion matrix shows exactly where predictions are correct and where they fail.

## Why the Confusion Matrix Is Foundational

The confusion matrix reveals:

- what mistake types the model makes and in what proportions
- which classes are being confused with each other
- whether errors are systematic or scattered
- whether improvement should target false positives or false negatives
- whether high aggregate metrics hide failure on specific classes

Every classification evaluation should include the confusion matrix, not metrics alone.

## What Is a Confusion Matrix?

A confusion matrix cross-tabulates actual labels against predicted labels for every sample in the evaluation set.

Binary structure:

| Actual \\ Predicted | Predicted Positive | Predicted Negative |
| ------------------- | ------------------ | ------------------ |
| Actual Positive     | TP                 | FN                 |
| Actual Negative     | FP                 | TN                 |

Where:

- TP (true positive): predicted positive, actually positive
- TN (true negative): predicted negative, actually negative
- FP (false positive): predicted positive, actually negative
- FN (false negative): predicted negative, actually positive

This is the most direct answer to: where is the model right, and where is it wrong?

## Interpreting the Four Cells

### True Positives (TP)

Correct detections of the positive class. Usually the outcomes you are trying to maximize.

### True Negatives (TN)

Correct rejections of the negative class. In imbalanced data, TN is often very large and can inflate accuracy.

### False Positives (FP)

False alarms: the model predicted positive but should not have.

Typical costs:

- customer friction and support load
- wasted manual review capacity
- reputational damage

### False Negatives (FN)

Missed cases: the model predicted negative but should have flagged positive.

Typical costs:

- untreated risk or harm
- direct financial loss
- delayed intervention

In many high-stakes domains, FN is more costly than FP. The confusion matrix forces this trade-off into view.

## The Source of All Major Classification Metrics

Key metrics are derived from confusion-matrix cells:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
Precision = \frac{TP}{TP + FP}
$$

$$
Recall = \frac{TP}{TP + FN}
$$

$$
F1 = \frac{2TP}{2TP + FP + FN}
$$

F1 excludes TN, which is why it is often more informative on imbalanced tasks focused on positive-class detection.

## Worked Example

Suppose on 200 transactions:

| Actual \\ Predicted | Predicted Fraud | Predicted Legitimate |
| ------------------- | --------------- | -------------------- |
| Actual Fraud        | 40              | 10                   |
| Actual Legitimate   | 5               | 145                  |

So:

- TP = 40
- FN = 10
- FP = 5
- TN = 145

Derived metrics:

$$
Accuracy = \frac{40 + 145}{200} = 0.925
$$

$$
Precision = \frac{40}{40 + 5} = 0.889
$$

$$
Recall = \frac{40}{40 + 10} = 0.800
$$

$$
F1 = \frac{2 \cdot 0.889 \cdot 0.800}{0.889 + 0.800} \approx 0.842
$$

This tells a richer story than one score: false alarms are low, but 1 in 5 fraud cases are still missed.

## Build It in Scikit-Learn

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
```

Binary layout in scikit-learn:

```text
[[TN FP]
 [FN TP]]
```

Rows are actual labels and columns are predicted labels.

Extract cells directly:

```python
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
```

## Visualize the Matrix

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Legitimate", "Fraud"],
    cmap="Blues"
)
plt.title("Confusion Matrix - Fraud Detection")
plt.tight_layout()
plt.show()
```

Normalized view (row-wise, recall per class):

```python
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Legitimate", "Fraud"],
    normalize="true",
    cmap="Blues"
)
plt.title("Normalized Confusion Matrix")
plt.show()
```

Normalization is especially important for imbalanced datasets.

## Imbalanced Data Interpretation

Example: 95% negative, 5% positive, model predicts all negatives:

```text
[[190   0]
 [ 10   0]]
```

- Accuracy = 95%
- Precision = 0 (no positive predictions)
- Recall = 0
- F1 = 0

The matrix makes the structural failure obvious: the model never predicts the positive class.

## Multi-Class Confusion Matrix

For $K$ classes, shape is $K \times K$.

- row $i$: actual class $i$
- column $j$: predicted class $j$
- diagonal: correct predictions
- off-diagonal: specific class confusions

Use off-diagonal concentration to prioritize feature engineering or data collection on class pairs that are frequently confused.

## Threshold Effects on the Matrix

Changing threshold reshapes all four cells.

Lower threshold:

- TP up
- FP up
- FN down
- recall usually up, precision usually down

Higher threshold:

- TP down
- FP down
- FN up
- precision usually up, recall usually down

```python
import numpy as np
from sklearn.metrics import confusion_matrix

y_prob = model.predict_proba(X_test)[:, 1]

for t in [0.3, 0.5, 0.7]:
    y_hat = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
    print(f"Threshold {t:.1f} | TP:{tp:4d} FP:{fp:4d} FN:{fn:4d} TN:{tn:4d}")
```

This converts threshold tuning into a concrete business trade-off discussion.

## Common Mistakes

- reporting metrics without looking at confusion-matrix counts
- misreading row and column orientation
- focusing only on diagonal values and ignoring off-diagonals
- skipping normalized views on imbalanced data
- comparing raw counts across different test sizes
- skipping baseline confusion-matrix comparison

## Practical Checklist Before Reporting

- confusion matrix computed on held-out test data
- row/column mapping verified (TN, FP, FN, TP)
- minority-class row inspected explicitly
- normalized matrix reviewed for class-wise recall
- matrix linked back to precision, recall, and F1
- threshold effects reviewed if threshold tuning is relevant
- baseline matrix compared to model matrix

## Closing

Aggregate metrics are lossy compressions of the confusion matrix.

The confusion matrix is the evidence behind every classification score. It shows how the model behaves, where it fails, and what kind of improvements are needed for real-world deployment.
