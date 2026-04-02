# Feature and Target Definition Guide for StockAxis

## Quick Reference: Before You Code Anything Else

1. **Identify your TARGET** → What exactly am I predicting?
2. **List valid FEATURES** → What information do I have at prediction time?
3. **Exclude invalid columns** → IDs, leakage, future data, raw timestamps
4. **Separate X and y** → Keep them distinct
5. **Split before preprocessing** → Train set only for fitting
6. **Validate for leakage** → Test your assumptions

---

## Part 1: Define Your Target (REQUIRED)

### Step 1.1: Identify Target Column

**Question**: What column in your dataset represents the outcome you want to predict?

**For StockAxis**: What are you predicting? Examples:

- Stock price direction (UP/DOWN)
- Market regime (BULL/NEUTRAL/BEAR)
- Daily return (continuous value)
- Price crossing threshold

**Answer**: `TARGET_COLUMN = 'your_column_name'`

### Step 1.2: Understand Target Values

**Question**: What are the possible values of your target?

```python
import pandas as pd

df = pd.read_csv('data/raw/dataset.csv')

# Examine target
print(df['TARGET'].dtype)
print(df['TARGET'].unique())
print(df['TARGET'].value_counts())
```

**For StockAxis**:

- If binary classification: 2 values (e.g., [0, 1] or ['UP', 'DOWN'])
- If multi-class: 3+ values (e.g., ['BULL', 'NEUTRAL', 'BEAR'])
- If regression: continuous values (e.g., [0.02, -0.015, 0.034])

### Step 1.3: Document Business Meaning

**For each value, write what it means in business terms**:

Example for binary:

- 1 = "Price will increase more than 1% tomorrow"
- 0 = "Price will not increase more than 1% tomorrow"

Example for multi-class:

- 'BULL' = "Uptrend conditions with strong buying pressure"
- 'NEUTRAL' = "Sideways movement without clear direction"
- 'BEAR' = "Downtrend conditions with selling pressure"

---

## Part 2: Define Your Features

### Step 2.1: Create Feature Lists

**For your StockAxis data, categorize each potential column**:

```python
# Numerical features (continuous or integer values)
NUMERICAL_FEATURES = [
    'OpenPrice',      # Current day opening
    'ClosePrice',     # Current day closing
    'Volume',         # Trading volume
    'RSI_14',         # Technical indicator
    'MACD',           # Technical indicator
]

# Categorical features (discrete categories)
CATEGORICAL_FEATURES = [
    'DayOfWeek',      # Monday-Friday representation
    'Sector',         # Stock sector
]

# Ordinal features (if any - with natural ordering)
ORDINAL_FEATURES = {
    'TrendStrength': ['Weak', 'Moderate', 'Strong']  # Example
}

# Excluded columns (with reasons)
EXCLUDED_COLUMNS = [
    'Date',                    # Reason: Raw timestamp (will derive temporal features)
    'Ticker',                  # Reason: Unique identifier
    'ClosePrice_Tomorrow',     # Reason: FUTURE DATA - DATA LEAKAGE
    'ActualPriceChange',       # Reason: Derived from target
]

# Combine all features
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + list(ORDINAL_FEATURES.keys())
```

### Step 2.2: Justify Each Feature

**For every feature, ask**: "Would I have this information at prediction time?"

Fill in this table for your project:

| Feature             | Available at Prediction Time? | Evidence                          | Include? |
| ------------------- | ----------------------------- | --------------------------------- | -------- |
| OpenPrice           | Yes                           | Known at market open/close        | ✓        |
| ClosePrice          | Yes                           | Known at end of trading day       | ✓        |
| RSI_14              | Yes                           | Calculated from historical prices | ✓        |
| ClosePrice_Tomorrow | No                            | Doesn't exist yet                 | ✗        |
| ActualReturn        | No                            | Only known after trading day      | ✗        |

### Step 2.3: Handle Raw Timestamps

**Never use raw timestamps directly**. Derive meaningful features:

```python
import pandas as pd

df['Date'] = pd.to_datetime(df['Date'])

# Derive temporal features
df['DayOfWeek'] = df['Date'].dt.dayofweek          # 0=Monday, 4=Friday
df['MonthOfYear'] = df['Date'].dt.month             # 1-12
df['DayOfMonth'] = df['Date'].dt.day                # 1-31
df['IsWeekend'] = (df['DayOfWeek'] >= 4).astype(int)
df['DaysSinceYearStart'] = df['Date'].dt.dayofyear
df['QuarterOfYear'] = df['Date'].dt.quarter

# Then EXCLUDE raw 'Date' column
EXCLUDED_COLUMNS.append('Date')
```

### Step 2.4: Validate All Features

```python
# Create validation function
def validate_features(df, target_col, feature_cols, excluded_cols):
    errors = []

    # Check 1: Target exists
    if target_col not in df.columns:
        errors.append(f"Target '{target_col}' not found in dataset")

    # Check 2: Target not in features
    if target_col in feature_cols:
        errors.append(f"Target '{target_col}' found in feature list")

    # Check 3: Excluded not in features
    overlap = set(excluded_cols) & set(feature_cols)
    if overlap:
        errors.append(f"Excluded columns in features: {overlap}")

    # Check 4: All features exist
    missing = set(feature_cols) - set(df.columns)
    if missing:
        errors.append(f"Features not found: {missing}")

    # Check 5: Watch for high correlations (possible leakage)
    numeric_features = [col for col in feature_cols
                        if df[col].dtype in ['float64', 'int64']]
    for col in numeric_features:
        if df[target_col].dtype in ['float64', 'int64']:  # Both numeric
            corr = df[col].corr(df[target_col])
            if abs(corr) > 0.95:
                errors.append(f"Feature '{col}' has {corr:.2f} correlation with target (LEAKAGE?)")

    if errors:
        print("❌ VALIDATION ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✓ Feature validation passed")
        return True

# Run validation
validate_features(df, TARGET_COLUMN, ALL_FEATURES, EXCLUDED_COLUMNS)
```

---

## Part 3: Separate Features and Target

### Step 3.1: Explicit Separation

```python
import pandas as pd

# Load data
df = pd.read_csv('data/raw/dataset.csv')

# Configuration
TARGET_COLUMN = 'YourTarget'
NUMERICAL_FEATURES = [...]
CATEGORICAL_FEATURES = [...]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Separate
X = df[ALL_FEATURES]
y = df[TARGET_COLUMN]

# Verify
print(f"Features shape: {X.shape}")  # Should be (n_rows, n_features)
print(f"Target shape: {y.shape}")    # Should be (n_rows,)
print(f"Target unique values: {y.unique()}")
print(f"Target distribution:\n{y.value_counts()}")
```

### Step 3.2: Never Violate Separation

```python
# WRONG - DON'T DO THIS
X = df  # All columns mixed
# Target is still in X!

y = df['target']
# Now target appears in both X and y

# Model learns by copying target from X to y → 100% training accuracy
# But in production, y doesn't exist → catastrophic failure
```

```python
# RIGHT - DO THIS
X = df[[col for col in df.columns if col != 'target']]
y = df['target']
# Or better yet:
X = df[FEATURE_LIST]
y = df[TARGET_COLUMN]
# Now X and y are truly separate
```

---

## Part 4: Train/Test Split (BEFORE Preprocessing)

### Step 4.1: Correct Sequence

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# STEP 1: Separate X and y (if not already done)
X = df[ALL_FEATURES]
y = df[TARGET_COLUMN]

# STEP 2: SPLIT FIRST (this is critical!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 80% training, 20% test
    random_state=42,         # For reproducibility
    stratify=y if len(y.unique()) < 10 else None  # Balance classes in train/test
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# STEP 3: Fit preprocessing on TRAINING DATA ONLY
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# STEP 4: Apply same transformation to TEST DATA
X_test_scaled = scaler.transform(X_test)

# Now you can train your model
# model.fit(X_train_scaled, y_train)
```

### Step 4.2: Why Split Before Fitting?

**WRONG sequence** (leakage):

```python
scaler.fit_transform(X)  # Fit on ALL data
# Scaler learns mean/std from train + test

X_train, X_test = split(X_scaled)
# Test set influenced training (leakage)
```

**RIGHT sequence** (no leakage):

```python
X_train, X_test = split(X)  # Split first
scaler.fit(X_train)         # Learn from train only
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Test set remains truly unseen
```

---

## Part 5: Centralized Configuration

### Step 5.1: Create config.py

Create `src/config_features.py` in your StockAxis project:

```python
# src/config_features.py
"""
Feature and target definitions for StockAxis model
"""

# ============================================================================
# TARGET VARIABLE
# ============================================================================

TARGET_COLUMN = 'PriceDirection'  # Change to your actual target
TARGET_TYPE = 'binary_classification'  # or 'regression' or 'multi_class'

# For classification, what does each value mean?
TARGET_MEANING = {
    0: 'Price will decrease (or stay flat)',
    1: 'Price will increase more than threshold_X%'
}

# ============================================================================
# FEATURES
# ============================================================================

# Numerical features (continuous or discrete numbers)
NUMERICAL_FEATURES = [
    'OpenPrice',
    'ClosePrice',
    'Volume',
    'RSI_14',      # Relative Strength Index
    'MACD',        # MACD indicator
    'SMA_14',      # 14-day Simple Moving Average
    'DaysSinceSignal',  # Time since last technical signal
]

# Categorical features (discrete categories)
CATEGORICAL_FEATURES = [
    'DayOfWeek',          # 0-4 (Monday-Friday)
    'MarketTrend',        # 'Uptrend', 'Downtrend', 'Sideways'
    'Sector',             # Stock sector
]

# Ordinal features (categories with ordering)
ORDINAL_FEATURES = {
    # 'FeatureName': ['Level1', 'Level2', 'Level3']
}

# Combined feature list
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + list(ORDINAL_FEATURES.keys())

# ============================================================================
# EXCLUDED COLUMNS (INELIGIBLE FOR FEATURES)
# ============================================================================

EXCLUDED_COLUMNS = {
    'Ticker': 'Unique identifier per stock',
    'Date': 'Raw timestamp (temporal features derived instead)',
    'ClosePrice_Tomorrow': 'Future data - temporal leakage',
    'ActualPriceChange': 'Derived from target - target leakage',
    'PriceChangedUp': 'The target variable itself',
}

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config():
    """Validate feature configuration before training"""

    # Check no overlap between features and excluded
    excluded_names = set(EXCLUDED_COLUMNS.keys())
    feature_names = set(ALL_FEATURES + [TARGET_COLUMN])

    overlap = excluded_names & feature_names
    if overlap:
        raise ValueError(f"Excluded columns in feature list: {overlap}")

    print(f"✓ Configuration valid")
    print(f"  - Target: {TARGET_COLUMN}")
    print(f"  - Numerical features: {len(NUMERICAL_FEATURES)}")
    print(f"  - Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"  - Excluded columns: {len(EXCLUDED_COLUMNS)}")
    print(f"  - Total features: {len(ALL_FEATURES)}")

# Run validation when imported
if __name__ == '__main__':
    validate_config()
```

### Step 5.2: Use Configuration in Your Code

```python
# src/data_preprocessing.py
import pandas as pd
from config_features import (
    TARGET_COLUMN,
    ALL_FEATURES,
    EXCLUDED_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

def prepare_data(df):
    """
    Separate features and target using centralized config
    """

    # Validate all features exist
    missing = set(ALL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Validate target exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target '{TARGET_COLUMN}' not found")

    # Separate
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    # Validate no target in features
    assert TARGET_COLUMN not in X.columns, "Target leaked into features!"

    return X, y

# In main.py or notebooks
if __name__ == '__main__':
    df = pd.read_csv('data/raw/dataset.csv')
    X, y = prepare_data(df)

    print(f"✓ Data prepared")
    print(f"  - Features: {X.shape}")
    print(f"  - Target: {y.shape}")
```

---

## Part 6: Checklist Before Training

```
BEFORE YOU TRAIN YOUR FIRST MODEL:

Target Definition:
  ☐ Target column identified
  ☐ Target values documented
  ☐ Business meaning documented for each value
  ☐ Problem type confirmed (classification/regression)
  ☐ Target values verified in dataset

Features:
  ☐ Numerical features listed
  ☐ Categorical features listed
  ☐ For each feature: documented why it's available at prediction time
  ☐ Raw timestamps transformed into derived features
  ☐ Text data transformed into numerical features
  ☐ No post-outcome data included
  ☐ No target-derived columns included

Exclusions:
  ☐ Identifiers excluded (CustomerID, Ticker, etc.)
  ☐ Post-outcome columns excluded
  ☐ Leakage sources identified and excluded
  ☐ Reasons documented for each exclusion

Code:
  ☐ X = features, y = target separated explicitly
  ☐ TARGET not in X verified
  ☐ EXCLUDED columns not in X verified
  ☐ All features exist in dataset verified
  ☐ Centralized configuration created
  ☐ Validation function written and passes

Train/Test:
  ☐ Data split BEFORE preprocessing
  ☐ No preprocessing fit on test set
  ☐ No test data used during training

Ready to proceed to preprocessing and training!
```

---

## Troubleshooting

### "I'm not sure if this feature creates leakage"

**Test**: Ask yourself - "At the exact moment I need to make a prediction in production, would this column exist in my database?"

If NO → It's leakage, exclude it.  
If YES → It might be valid, verify it's not derived from target.

### "My training accuracy is 95% but test accuracy is 40%"

**Likely cause**: Data leakage during feature definition

**Action**:

1. Check for target-derived columns
2. Check for post-outcome data
3. Check for future information
4. Verify train/test split happened before preprocessing

### "I have a timestamp column, what should I do?"

**Never use raw timestamp directly**.

```python
# WRONG
df['Date']  # 2024-03-15 14:23:17

# RIGHT
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['HourOfDay'] = df['Date'].dt.hour
df['MonthOfYear'] = df['Date'].dt.month
df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days
```

### "Should I include ID columns like CustomerID or Ticker?"

**Almost never**.

**Why**: IDs uniquely identify training samples. A model that memorizes IDs learns nothing generalizable and overf its.

**Exception**: If ID encodes meaningful information (like ProductID contains category info), derive a feature instead:

```python
# WRONG
X = df[['ProductID', 'Price']]

# RIGHT
df['ProductCategory'] = df['ProductID'].map(product_mapping)  # Derived feature
X = df[['ProductCategory', 'Price']]
```

---

## Next Steps

1. **Fill in the template** - Answer all "Your answer" sections
2. **Create config_features.py** - Centralize your definitions
3. **Implement validation** - Test your assumptions
4. **Document everything** - Explain your choices
5. **Proceed to preprocessing** - You're now ready for the next step

**Remember**: Get feature and target definition right. Everything else depends on this foundation.
