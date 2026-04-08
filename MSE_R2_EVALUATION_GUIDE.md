# Evaluating Regression Models Using MSE and R2

After training a regression model, evaluate it from two complementary angles:

- How large are prediction errors in squared terms?
- How much variance is explained relative to the mean baseline?

MSE and R2 answer these questions together.

## Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

Key behavior:

- all errors become non-negative after squaring
- large errors are amplified quadratically
- units are squared target units

Because units are squared, MSE is usually paired with RMSE for reporting:

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

## Coefficient of Determination (R2)

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

Interpretation:

- 1.0 means perfect predictions
- 0.0 means equivalent to predicting the mean baseline
- less than 0 means worse than the mean baseline

R2 is a relative metric. It measures improvement over the mean baseline, not absolute error magnitude.

## Why Use Both Together

- MSE or RMSE provides absolute error scale
- R2 provides relative explanatory power

Common patterns:

- low RMSE but low R2: target variance is small and baseline already does well
- high RMSE but high R2: target variance is large and model still improves strongly over baseline

Neither metric should be reported alone.

## Project Implementation

This repository computes:

- test metrics: MSE, RMSE, MAE, R2
- baseline metrics with DummyRegressor(mean)
- improvement metrics versus baseline
- CV R2 metrics (mean, std, per fold)
- CV MSE metrics (mean, std, per fold)
- CV RMSE metrics (mean, std, per fold)

Core implementation:

- `main.py`
- `src/evaluate.py`
- `src/baseline.py`

## Cross-Validation Notes

For MSE scoring in scikit-learn:

- scoring uses neg_mean_squared_error
- convert back via sign flip before reporting

This project stores positive CV MSE and CV RMSE in metrics output for direct interpretation.

## Baseline Comparison Rule

Always compare on the same test split and same metric:

$$
\text{MSE Improvement} = \text{Baseline MSE} - \text{Model MSE}
$$

$$
\text{R2 Improvement} = \text{Model R2} - \text{Baseline R2}
$$

This keeps conclusions grounded in a meaningful reference point.

## Practical Checklist

- evaluate on held-out test data, not train data
- report baseline and model side by side
- include RMSE for interpretability
- include CV mean and std for stability
- inspect for negative R2 on test set or folds
- interpret absolute errors against business tolerance

MSE measures magnitude of squared error. R2 measures relative explained variance. Use both to make reliable regression claims.
