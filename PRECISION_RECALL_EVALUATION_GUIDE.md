# Precision and Recall Evaluation Guide for Classification

Accuracy shows overall correctness, but it hides error asymmetry.

Precision and Recall expose what accuracy cannot:

- Precision: quality of positive predictions
- Recall: completeness of positive detection

## Confusion Matrix Mapping

Binary outcomes:

- True Positive (TP): predicted positive and actually positive
- False Positive (FP): predicted positive but actually negative
- False Negative (FN): predicted negative but actually positive
- True Negative (TN): predicted negative and actually negative

Precision and Recall formulas:

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

## Interpretation

- High Precision means few false alarms.
- High Recall means few missed positives.
- They usually trade off as threshold changes.

## Project Metrics in Classification Mode

This project now reports:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1
- ROC-AUC (when available)

Plus cross-validated stability metrics:

- CV Precision mean/std and per-fold values
- CV Recall mean/std and per-fold values
- CV F1 mean/std and per-fold values
- CV Accuracy and Balanced Accuracy mean/std and per-fold values
- CV ROC-AUC mean/std and per-fold values

All metrics are written to `reports/metrics.json`.

## Baseline Comparison

Use the majority-class baseline (`DummyClassifier`) as the floor.

A useful model should exceed baseline precision/recall meaningfully, especially minority-class recall.

## Threshold Adjustment Support

Default class prediction uses threshold 0.5, but this is not always optimal.

You can tune threshold behavior with `src/config.py`:

- `classification_threshold`: threshold used for thresholded binary predictions
- `target_recall`: optional target recall to select the best precision-achieving threshold from PR curve

When enabled for binary classification, the pipeline writes:

- Thresholded precision/recall/F1/accuracy/balanced accuracy to `reports/metrics.json`
- Thresholded predictions in `reports/predictions.csv` (`prediction_thresholded` column)
- Precision-recall curve points to `reports/precision_recall_curve.csv`

## Practical Guidance

Use Precision-first when false positives are costly:

- Compliance alerting
- Spam filtering for critical inboxes
- High-confidence moderation queues

Use Recall-first when false negatives are costly:

- Disease screening
- Fraud detection
- Breach/intrusion detection

If priorities are mixed, track both and optimize F1 (or task-specific F-beta).

## Checklist Before Reporting

- Report both Precision and Recall, not accuracy alone
- Compare against baseline precision/recall
- Inspect confusion matrix and classification report
- Validate CV mean and variance for Precision and Recall
- Evaluate threshold suitability for business risk tolerance
- Document trade-off rationale in plain language
