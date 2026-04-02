# Defining Feature Columns and Target Variables - Implementation Guide

## What You Have

After this lesson, your StockAxis project includes:

1. **Jupyter Notebook**: `notebooks/defining_features_and_targets.ipynb`
   - Complete interactive lesson with code examples
   - Demonstrations of common mistakes
   - Validation patterns
   - Hands-on exercises

2. **Implementation Guide**: `FEATURE_TARGET_DEFINITION_GUIDE.md`
   - Step-by-step instructions
   - Code templates
   - Troubleshooting section
   - Checklist before training

3. **Configuration Template**: `src/config_features.py`
   - Centralized feature/target definitions
   - Validation functions
   - Ready to fill in with your project values

4. **This Document**: Quick-start reference

---

## Quick Start (5 Minutes)

### Step 1: Open the Notebook

```bash
cd S86-0326-StockAxis-Machine-Learning
jupyter notebook notebooks/defining_features_and_targets.ipynb
```

Run all cells to understand the concepts and see examples in action.

### Step 2: Answer the Key Questions

Before you touch any code, answer these in writing:

1. **What is your target variable?**
   - Column name?
   - What does it represent?
   - Classification or regression?

2. **What information do you have at prediction time?**
   - List valid features
   - For each: explain why it's available now

3. **What columns are excluded and why?**
   - Identifiers? Leakage? Post-outcome? Raw timestamps?

4. **Document your answers in:** `FEATURE_TARGET_DEFINITION_GUIDE.md` (Part 1-3)

### Step 3: Update Configuration

Open `src/config_features.py` and fill in:

```python
# TODO sections - fill these in with YOUR project values

TARGET_COLUMN = 'your_target_name'
TARGET_TYPE = 'binary_classification'  # or 'regression', 'multi_class'

NUMERICAL_FEATURES = ['feature1', 'feature2', ...]
CATEGORICAL_FEATURES = ['feature3', 'feature4', ...]
EXCLUDED_COLUMNS = {'id_col': 'reason', ...}
```

### Step 4: Test Your Configuration

Run this in Python:

```python
from src.config_features import validate_configuration
validate_configuration()
```

If it passes: ✓ Ready to proceed  
If it fails: Fix issues shown in error messages

---

## Complete Workflow

### Phase 1: Definition (Do First)

**Time**: 30-60 minutes

```
1. Read FEATURE_TARGET_DEFINITION_GUIDE.md (Part 1-3)
2. Answer all questions in writing
3. Fill in src/config_features.py
4. Run validate_configuration()
5. If issues: Fix based on error messages
```

**Deliverable**: Completed `src/config_features.py` with no errors

### Phase 2: Implementation (Build the Pipeline)

**Time**: 1-2 hours

```python
# src/data_preprocessing.py (example)

import pandas as pd
from src.config_features import (
    TARGET_COLUMN,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    validate_configuration
)

# Validate on import
validate_configuration()

def prepare_data(df):
    """
    Separate features and target using centralized config.
    No leakage. No mistakes.
    """

    # Separate X and y
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    # Verify no target in features
    assert TARGET_COLUMN not in X.columns, "ERROR: Target leaked into features!"

    return X, y

def split_and_preprocess(X, y):
    """
    Correct sequence: Split BEFORE preprocessing
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # STEP 1: Split first (critical!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 2: Fit preprocessing on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # STEP 3: Apply to test
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
```

### Phase 3: Validation (Prevent Mistakes)

**Before training any model:**

```
☐ TARGET column identified and documented
☐ TARGET values make sense in dataset
☐ All FEATURES can be computed at prediction time
☐ No TARGET-DERIVED columns in features
☐ No FUTURE DATA in features
☐ No POST-OUTCOME data in features
☐ X and y separated explicitly
☐ X and y have same number of rows
☐ Train/test split done BEFORE preprocessing
☐ Preprocessing fit only on training data
☐ Configuration passes validate_configuration()
```

---

## Common Patterns for StockAxis

### Pattern 1: Stock Price Direction (Binary Classification)

```python
# config_features.py

TARGET_COLUMN = 'price_direction_tomorrow'
TARGET_TYPE = 'binary_classification'

TARGET_MEANING = {
    0: 'Price will not increase by 1% tomorrow',
    1: 'Price will increase by 1% or more tomorrow'
}

NUMERICAL_FEATURES = [
    'open_price',      # Available at market open
    'close_price',     # Available at market close
    'volume',          # Available at market close
    'rsi_14',          # Technical indicator (historical)
    'macd',            # Technical indicator (historical)
]

CATEGORICAL_FEATURES = [
    'day_of_week',     # 0=Monday, 4=Friday
]

EXCLUDED_COLUMNS = {
    'ticker': 'Stock identifier',
    'date': 'Raw timestamp (use derived temporal features)',
    'close_price_tomorrow': 'Future data - LEAKAGE',
    'price_change_tomorrow': 'Derived from target',
}
```

### Pattern 2: Market Regime (Multi-Class Classification)

```python
TARGET_COLUMN = 'market_regime'
TARGET_TYPE = 'multi_class'

TARGET_MEANING = {
    'bull': 'Strong uptrend with buying pressure',
    'neutral': 'Sideways movement without clear direction',
    'bear': 'Strong downtrend with selling pressure'
}

NUMERICAL_FEATURES = [...]  # Similar to above

CATEGORICAL_FEATURES = [
    'day_of_week',
    'market_condition',  # e.g., 'high_volume', 'low_volume'
]
```

### Pattern 3: Return Prediction (Regression)

```python
TARGET_COLUMN = 'daily_return_pct'
TARGET_TYPE = 'regression'

# No TARGET_MEANING for regression (continuous values)

NUMERICAL_FEATURES = [
    'open_price',
    'close_price_prev_day',
    'volume',
    'volatility_20d',
]

# For regression, be especially careful about leakage!
EXCLUDED_COLUMNS = {
    'close_price_today': 'Not available until end of day - future data',
    'actual_return': 'This IS the target',
}
```

---

## Debugging Checklist

### "My model achieves 95% training accuracy but 50% on test"

**Likely cause**: Data leakage

**Debug steps**:

1. Check if TARGET appears in X: `assert TARGET_COLUMN not in X.columns`
2. Check for target-derived features: `df_train.corr()[TARGET_COLUMN].sort_values(ascending=False)`
3. Verify preprocessing split: Did you fit on ALL data instead of train-only?
4. Check temporal leakage: Do any features use future information?

### "Validation fails: Feature not in dataset"

**Solution**: Run this to see what's in your dataset:

```python
print(df.columns.tolist())
```

Compare with `ALL_FEATURES` in config and fix the name mismatches.

### "I'm not sure if this feature creates leakage"

**Test**: At the moment you need to make a prediction in production, does this information exist in your database?

- YES → Likely valid
- NO → Leakage, exclude it

### "I have a raw timestamp column"

**Action**: Derive temporal features instead:

```python
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
# Then exclude 'date' from features
```

---

## Files Reference

| File                                            | Purpose                | When to Use                         |
| ----------------------------------------------- | ---------------------- | ----------------------------------- |
| `notebooks/defining_features_and_targets.ipynb` | Interactive lesson     | Learning concepts, running examples |
| `FEATURE_TARGET_DEFINITION_GUIDE.md`            | Implementation guide   | Step-by-step instructions           |
| `src/config_features.py`                        | Configuration template | Fill in with your project values    |
| This document                                   | Quick reference        | Quick lookup, common patterns       |

---

## Next Steps After This Lesson

1. ✅ **Complete this lesson**
   - Read notebook
   - Answer questions
   - Fill in config_features.py

2. **Preprocessing** (next lesson)
   - Handle missing values
   - Encode categorical features
   - Scale numerical features
   - All using YOUR features from this lesson

3. **Feature Engineering** (next)
   - Create new features
   - Select best features
   - Validate no new leakage

4. **Training** (next)
   - Train models
   - Evaluate with appropriate metrics
   - Deploy with confidence

---

## One Golden Rule

**Define features and targets correctly first. Everything else follows from that.**

If features are wrong:

- ❌ Preprocessing fixes nothing
- ❌ Algorithm selection fixes nothing
- ❌ Hyperparameter tuning fixes nothing
- ❌ Only deployment drama when model fails

If features are right:

- ✅ Preprocessing is straightforward
- ✅ Algorithm selection becomes clear
- ✅ Evaluation makes sense
- ✅ Deployment works first time

---

## Questions?

Refer to:

1. **Conceptual questions**: See `notebooks/defining_features_and_targets.ipynb`
2. **Implementation questions**: See `FEATURE_TARGET_DEFINITION_GUIDE.md`
3. **Configuration questions**: See `src/config_features.py` comments
4. **Troubleshooting**: See "Debugging Checklist" section in this document

---

## Remember

Getting feature and target definition right is:

- Not optional
- Not negotiable
- The foundation of everything
- Worth the time invested

Focus here. Build on solid ground. Success follows.
