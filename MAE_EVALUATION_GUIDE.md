# Evaluating Regression Models Using MAE

Mean Absolute Error is one of the most interpretable regression metrics. It answers one direct question:

How far off are predictions on average?

## MAE Definition

$$
\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

- $y_i$: true value
- $\hat{y}_i$: predicted value
- $N$: number of samples

Because MAE uses absolute values and no squaring:

- over-predictions and under-predictions are treated equally
- large errors are penalized linearly, not quadratically
- the metric stays in the same units as the target

## Why Teams Prefer MAE

MAE is easy to explain to non-technical stakeholders. If your target is in lakhs and MAE is 3.5, the model is off by 3.5 lakhs on average.

That makes MAE suitable for business reporting, planning, and model acceptance criteria.

## MAE vs MSE vs RMSE

- MAE: linear penalty, robust to outliers compared with squared-error metrics
- MSE: squares error, strongly penalizes large misses
- RMSE: square root of MSE, same units as target but still outlier-sensitive

Use MAE as the primary communication metric when average absolute error is what matters most.

## Project Workflow

This repository evaluates MAE with a leakage-safe process:

1. Split train/test before fitting
2. Fit baseline and model on training data
3. Evaluate both on the same test set
4. Report MAE alongside RMSE and R2
5. Add CV MAE for stability across folds
6. Export residuals for bias checks

Implementation files:

- `main.py`
- `src/evaluate.py`
- `src/baseline.py`

## Baseline Comparison (Required)

Always compare MAE to a mean baseline:

$$
\text{MAE Improvement} = \text{Baseline MAE} - \text{Model MAE}
$$

Also track percentage improvement:

$$
\text{Improvement \%} = \frac{\text{Baseline MAE} - \text{Model MAE}}{\text{Baseline MAE}} \times 100
$$

A model MAE without baseline context is not enough for decision-making.

## Scale Context

Interpret MAE relative to target scale:

$$
\text{MAE \% of Mean Target} = \frac{\text{MAE}}{|\overline{y}|} \times 100
$$

This helps answer whether the average error is acceptable in domain terms.

## Cross-Validation with MAE

Scikit-learn scoring uses negated MAE for maximization conventions. Convert back before reporting:

- scoring value: neg_mean_absolute_error
- reported MAE: negative of returned score

Report both:

- Mean CV MAE
- Std CV MAE

A low mean with high standard deviation signals unstable performance.

## Residual Checks

MAE does not show directional bias. Use residual analysis:

- residual = predicted - actual
- inspect residual spread and trends
- look for systematic over- or under-prediction

This project exports residual diagnostics to `reports/residuals.csv`.

## Practical Reporting Checklist

- MAE computed on held-out test data
- MAE compared against baseline MAE on same split
- MAE contextualized as percentage of mean target
- CV MAE mean and std reported
- Residuals checked for bias
- MAE reported with RMSE and R2

MAE is simple, but rigorous interpretation makes it trustworthy.
