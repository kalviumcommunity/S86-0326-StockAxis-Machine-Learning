# Accuracy Evaluation Guide for Classification

Accuracy answers a simple question: how often is the predicted class exactly correct?

## Formula

- Accuracy = (number of correct predictions) / (total predictions)
- Binary confusion matrix form: (TP + TN) / (TP + TN + FP + FN)

## Why Accuracy Is Useful

Accuracy is intuitive and easy to communicate.

- `0.92` means the model is correct 92 out of 100 predictions.
- It works well when class distribution is close to balanced and error costs are similar.

## Why Accuracy Can Mislead

On imbalanced datasets, accuracy can look high while the model misses the minority class entirely.

Example:

- 95% class 0, 5% class 1
- Model predicts class 0 for every sample
- Accuracy = 95%, but recall for class 1 = 0

In that situation, standard accuracy hides model failure.

## Project Implementation

Classification evaluation in this project now includes:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1
- ROC-AUC (when available)

All are saved in `reports/metrics.json` as baseline and model metrics.

## Balanced Accuracy

Balanced Accuracy is the average recall across classes.

- Binary form: (Sensitivity + Specificity) / 2
- Helps when one class dominates the dataset

In imbalanced settings, use Balanced Accuracy as the primary metric and standard Accuracy as supporting context.

## Baseline Comparison (Required)

Always compare model accuracy to a naive baseline.

This project uses `DummyClassifier` in classification mode.

A model should beat baseline meaningfully, not just by tiny random variation.

## Confusion Matrix Requirement

Do not report accuracy alone.

This project exports confusion matrix values to:

- `reports/confusion_matrix.csv`

Also inspect:

- `reports/classification_report.txt`

These outputs reveal whether high accuracy is coming from majority-class dominance.

## Cross-Validation for Stability

Single split accuracy can be noisy.

This project reports CV statistics for classification mode:

- Accuracy mean/std and per-fold values
- Balanced Accuracy mean/std and per-fold values
- F1 mean/std and per-fold values
- ROC-AUC mean/std and per-fold values

Use both central tendency and spread to judge reliability.

## Practical Reporting Checklist

- Compare to majority-class baseline
- Report Accuracy and Balanced Accuracy together
- Include confusion matrix and classification report
- Report CV mean and std, not single split only
- Confirm gains are meaningful for the use case

## When to Use Accuracy as Primary Metric

Use Accuracy as primary only when:

- Classes are reasonably balanced
- False positives and false negatives have similar cost
- Overall correctness is the main objective

Otherwise, prefer Balanced Accuracy, Recall, F1, and ROC-AUC.
