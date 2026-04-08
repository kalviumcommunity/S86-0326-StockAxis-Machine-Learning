# Training a Linear Regression Model

After a baseline is established, the next model for regression tasks should usually be Linear Regression. It is simple, fast, interpretable, and gives a strong reference point for more complex models.

## What Linear Regression Learns

Linear Regression predicts a continuous target using a weighted sum of features:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n
$$

Where:

- $\hat{y}$: predicted value
- $x_i$: input features
- $\beta_i$: learned coefficients
- $\beta_0$: intercept

Training minimizes mean squared error:

$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

## Leakage-Safe Workflow

Use this order every time:

1. Split data into train/test
2. Fit preprocessing only on train
3. Fit baseline on train and evaluate on test
4. Fit Linear Regression on train and evaluate on test
5. Compare baseline vs model on the same test split
6. Run cross-validation for stability

## Project Implementation

This repository implements the workflow in:

- `main.py`
- `src/train.py`
- `src/evaluate.py`
- `src/baseline.py`

The pipeline includes:

- `DummyRegressor(strategy="mean")` baseline
- `LinearRegression` model
- RMSE, MAE, MSE, and R2 metrics
- 5-fold CV R2 reporting
- Learned coefficient export to `reports/coefficients.csv`

## Metric Interpretation

Recommended primary metrics for this project:

- RMSE: error magnitude in target units
- MAE: robust average error magnitude
- R2: explained variance relative to mean predictor

Improvement direction:

- Lower is better: MSE, RMSE, MAE
- Higher is better: R2

## Coefficient Interpretation

Each coefficient represents the expected target change for a one-unit increase in that feature, holding others constant.

Interpret coefficients carefully:

- Use consistent scaling if comparing magnitudes
- Watch for multicollinearity (correlated features can destabilize coefficients)
- Validate assumptions with residual plots

## Practical Checklist

Before calling the model successful:

- Model beats baseline on RMSE and MAE
- Model has meaningfully better R2 than baseline
- CV R2 is stable across folds
- Coefficients are directionally sensible
- Residual behavior does not show obvious structure

Linear Regression is often both the first serious regression model and the most interpretable one in production workflows.
