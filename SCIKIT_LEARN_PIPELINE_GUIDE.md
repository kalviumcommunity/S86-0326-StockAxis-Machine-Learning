# Building a Scikit-Learn Pipeline for Preprocessing and Modeling

As models become more sophisticated, the workflow becomes more complex.

You are no longer just fitting a model. You are scaling numerical features, encoding categorical variables, handling missing values, possibly performing feature selection, and then training and evaluating a model.

Applied manually and separately, those steps introduce failure modes that are easy to miss and hard to debug:

- Data leakage from fitting preprocessing on the full dataset
- Inconsistent transformations between train and test data
- Fragile, non-reproducible code
- Invalid cross-validation scores caused by preprocessing that already saw the validation fold

This is why scikit-learn pipelines exist.

A pipeline bundles preprocessing and modeling steps into one object that is fitted, validated, tuned, serialized, and deployed as a unit. Every step is applied in the correct order, fitted only on training data, and reused consistently at prediction time.

## What Is a Pipeline?

A pipeline is a sequential chain of transformers followed by a final estimator. It implements the same `fit`, `predict`, and `score` interface as any scikit-learn model, which makes it compatible with cross-validation, GridSearchCV, and serialization.

Raw Data -> Step 1 -> Step 2 -> Final Estimator -> Prediction

Each intermediate step must implement:

- `fit(X, y)` to learn parameters from data
- `transform(X)` to apply the learned transformation

The final step implements:

- `fit(X, y)` to train the model
- `predict(X)` to generate predictions

When you call `pipeline.fit(X_train, y_train)`, each transformer is fitted only on the training data. When you call `pipeline.predict(X_test)`, the stored transformations are reused without refitting.

## Why Pipelines Prevent Data Leakage

The key leakage risk is fitting preprocessing before cross-validation.

```python
# WRONG: preprocessing sees all of X_train before CV
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
```

In that example, the scaler has already seen every training sample, including the samples that will later become validation folds. The validation fold is not truly unseen.

The pipeline solution refits preprocessing inside each cross-validation split:

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])

scores = cross_val_score(pipeline, X_train, y_train, cv=5)
```

Inside each fold, the scaler is fitted on that fold's training portion only. The validation portion is transformed using only statistics learned from the fold's training data.

## Simple Pipeline: Scaling + Logistic Regression

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression(max_iter=1000)),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
```

The pipeline becomes the model. You interact with it like any other scikit-learn estimator.

## Mixed Data Types: ColumnTransformer

Real datasets usually contain numerical features that need scaling and categorical features that need encoding.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_features = ["Age", "Income", "Tenure"]
cat_features = ["Gender", "City", "ContractType"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ],
    remainder="drop",
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])
```

Use `handle_unknown="ignore"` so production data containing unseen categories does not crash the encoder. Use `remainder="drop"` so accidental columns do not flow into the model.

## Handling Missing Values with Nested Pipelines

For datasets with missing values, nest sub-pipelines inside the `ColumnTransformer`.

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])
```

This is the production-grade pattern. The imputer learns fill values only from the training fold, and the same fitted logic is reused at prediction time.

## Inspecting the Pipeline

```python
print(pipeline)

preprocessor_step = pipeline.named_steps["preprocessor"]

fitted_num = (
    pipeline.named_steps["preprocessor"]
    .named_transformers_["num"]
    .named_steps["scaler"]
)
print("Learned means:", fitted_num.mean_)
print("Learned scales:", fitted_num.scale_)

feature_names_out = pipeline.named_steps["preprocessor"].get_feature_names_out()
print("Output features:", feature_names_out)
```

Inspecting the fitted pipeline is useful for debugging and for understanding exactly what the model sees after preprocessing.

## Pipelines with Cross-Validation

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="f1",
)

print(f"CV F1 scores: {cv_scores.round(3)}")
print(f"Mean CV F1:   {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
```

Inside each fold, the pipeline refits the preprocessing steps from scratch on the fold's training data. That is the correct evaluation methodology.

## Pipelines with GridSearchCV

The double-underscore notation lets you tune any hyperparameter anywhere in the pipeline.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__max_depth": [4, 6, 8, 10],
    "model__min_samples_leaf": [1, 5, 10],
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    return_train_score=True,
    n_jobs=-1,
)

grid.fit(X_train, y_train)

print(f"Best Parameters: {grid.best_params_}")
print(f"Best CV F1:      {grid.best_score_:.3f}")

y_pred = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
```

You can also tune preprocessing choices alongside model hyperparameters.

```python
param_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "model__max_depth": [4, 6, 8],
}
```

## Saving and Loading Pipelines for Deployment

The pipeline is a single deployable artifact.

```python
import joblib

joblib.dump(pipeline, "churn_model_pipeline.pkl")
loaded_pipeline = joblib.load("churn_model_pipeline.pkl")

new_predictions = loaded_pipeline.predict(new_raw_data)
```

This avoids reproducing preprocessing logic separately in production.

## A Complete Production-Grade Example

```python
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_features = ["Age", "Income", "Tenure"]
cat_features = ["Gender", "ContractType", "PaymentMethod"]

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42)),
])

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
print(f"Baseline Accuracy: {accuracy_score(y_test, baseline.predict(X_test)):.3f}")

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [4, 6, 8],
    "model__min_samples_leaf": [1, 5],
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    return_train_score=True,
    n_jobs=-1,
)

grid.fit(X_train, y_train)

print(f"Best Parameters: {grid.best_params_}")
print(f"Best CV F1:      {grid.best_score_:.3f}")

y_pred = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(grid.best_estimator_, "production_pipeline.pkl")
```

This is the correct end-to-end workflow. Every preprocessing step is protected from leakage, and the deployed artifact is identical to what was evaluated.

## Common Mistakes to Avoid

- Scaling before the train-test split
- Applying preprocessing outside the pipeline before cross-validation
- Forgetting `handle_unknown="ignore"` in `OneHotEncoder`
- Saving only the model and not the full pipeline
- Using `remainder="passthrough"` without intention
- Tuning hyperparameters outside the pipeline

## Practical Checklist Before Finalizing

- Train/test split performed before any fitting
- All preprocessing inside the pipeline
- `ColumnTransformer` used for mixed feature types
- `handle_unknown="ignore"` set on `OneHotEncoder`
- `remainder` explicitly set
- Cross-validation applied to the full pipeline
- Hyperparameter tuning run on the full pipeline
- Full pipeline saved with `joblib.dump`
- Test set evaluated once using raw, unpreprocessed data

## Closing Reflection

Pipelines are a correctness guarantee.

Without them, leakage during cross-validation is easy to introduce and hard to detect. With them, preprocessing and modeling stay aligned from training through deployment.

For serious machine learning work, pipelines are the default structure, not an optional refinement.
