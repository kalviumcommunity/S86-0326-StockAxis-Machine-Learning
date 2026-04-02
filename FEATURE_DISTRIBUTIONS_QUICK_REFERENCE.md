# Feature Distributions Quick Reference

**Cheat Sheet for Distribution Inspection**

---

## Numerical Features: At-a-Glance

```python
# The 30-second inspection
df.describe()                                   # 1. Summary stats
df.isna().sum()                                 # 2. Missing values
plt.hist(df[col], bins=30); plt.show()         # 3. See shape
from scipy.stats import skew; skew(df[col])    # 4. Skewness value

# Decision: Transform needed?
# If |skewness| < 0.5 → Skip transformation
# If 0.5 < |skewness| < 1.5 → Log transform
# If |skewness| > 1.5 → Box-Cox
```

---

## Categorical Features: At-a-Glance

```python
# The 30-second inspection
df[col].value_counts()                         # 1. See all categories
df[col].nunique()                              # 2. How many?
df[col].isna().sum()                           # 3. Missing
(df[col].value_counts() / len(df) * 100 < 5).sum()  # 4. Rare categories?

# Decision: Encoding needed?
# If 2 categories → Binary encode (0/1)
# If 3-10 categories → One-hot encode
# If 10+ categories → Group rare, then encode
```

---

## The 5-Minute Inspection Protocol

For **ANY** dataset:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

df = load_your_data()

# Step 1: Get size & types (10 seconds)
print(df.shape)
print(df.dtypes)

# Step 2: Summary & missing (20 seconds)
print(df.describe())
print(df.isna().sum())

# Step 3: Visual inspection (120 seconds)
df.hist(figsize=(15, 10), bins=30)
plt.show()

# Step 4: Skewness check (60 seconds)
for col in df.select_dtypes(include=[int, float]).columns:
    s = skew(df[col].dropna())
    print(f"{col}: {s:.2f}", "← LOG TRANSFORM" if abs(s) > 1 else "")

# Step 5: Categorical check (60 seconds)
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col}: {df[col].nunique()} unique")
    print(df[col].value_counts().head())
```

---

## Transformation Lookup

| Problem         | Solution          | Code                                                       |
| --------------- | ----------------- | ---------------------------------------------------------- |
| Right-skewed    | Log transform     | `np.log1p(df[col])`                                        |
| Left-skewed     | Reflect + log     | `np.log1p(df[col].max() - df[col])`                        |
| Outliers        | Cap at IQR bounds | `df[col].clip(Q1-1.5*IQR, Q3+1.5*IQR)`                     |
| Needs scaling   | StandardScalers   | `(df[col] - mean) / std`                                   |
| 0-1 scale       | MinMaxScaler      | `(df[col] - min) / (max - min)`                            |
| Many categories | Group rare        | `df[col].apply(lambda x: x if x in top_cats else 'Other')` |

---

## Red Flags Checklist

- [ ] More than 20% missing values → Investigate why
- [ ] Skewness > 2.0 → Likely needs transformation
- [ ] Outliers > 10% → Check if real or corrupted
- [ ] Cardinality > 100 → May cause memory issues
- [ ] Class imbalance > 10:1 → Need stratified split + class weights
- [ ] Feature max >>> mean (e.g., max < mean) → Exponential dist, use log
- [ ] All same value → Remove (no variance, no predictive power)
- [ ] Missing completely random? → Investigate, may be informative

---

## The Three Questions

Ask for **every feature**:

1. **Does this feature vary?** (std > 0, nunique > 1)
   - If NO → Remove it
2. **Is this feature related to the target?** (correlation, interaction)
   - If NO → Investigate if it's redundant
3. **Is this feature in reasonable form for my model?** (right scale, right type)
   - If NO → Transform it

---

## When to Use Each Visualization

| Visualization   | Best For                            | Command                              |
| --------------- | ----------------------------------- | ------------------------------------ |
| **Histogram**   | Seeing distribution shape           | `plt.hist()`                         |
| **Boxplot**     | Spotting outliers, quartiles        | `plt.boxplot()`                      |
| **Violin Plot** | Full distribution shape + quartiles | `sns.violinplot()`                   |
| **QQ-Plot**     | Normality testing                   | `stats.probplot()`                   |
| **Heatmap**     | Correlations between features       | `sns.heatmap(df.corr())`             |
| **Bar Chart**   | Categorical counts                  | `df.value_counts().plot(kind='bar')` |

---

## Model-Specific Preprocessing

### Tree-Based Models (RandomForest, XGBoost)

```
✓ Skip scaling
✓ Skip normality transforms (they don't care)
✓ Handle outliers carefully (they become important splits)
✓ Can handle high cardinality (just slower)
✓ Missing values: Impute with median or create 'Missing'
```

### Linear Models (Logistic Regression, Linear Regression)

```
✓ MUST skew < 0.5 (log transform if needed)
✓ MUST scale (StandardScaler or MinMaxScaler)
✓ Outliers can break model (cap or remove)
✓ Missing values: Impute with mean
✓ Categorical: Must be one-hot encoded
```

### Distance-Based (KNN, SVM)

```
✓ MUST scale (very sensitive to feature scales)
✓ Outliers affect distances (cap or remove)
✓ Missing values: Impute with KNN or mean
✓ Categorical: One-hot encode
✓ High dimensions: Slow (feature selection helps)
```

### Neural Networks

```
✓ MUST scale to [0,1] or standardize to mean=0
✓ Outliers can destabilize training (cap)
✓ Missing values: Impute or create flag
✓ Categorical: Embedding or one-hot encode
✓ Skewness: Log transform helps convergence
```

---

## Distribution Names (For Stock/Financial Data)

**Prices/Returns**: Often **log-normal** (right-skewed)  
→ Solution: Log transform

**Volumes/Counts**: Often **Poisson** or **Gamma** (discrete or right-skewed)  
→ Solution: Keep as-is for trees, log for linear

**Time-to-event**: Often **Exponential** (heavily right-skewed)  
→ Solution: Log or sqrt transform

**Proportions**: Often **Beta** (bounded 0-1)  
→ Solution: Logit transform or leave as-is for trees

**Presence/Absence**: **Bernoulli** (0/1)  
→ Solution: Use as-is, no transform needed

---

## Common Mistakes

❌ **Fitting scaler on test data before removing outliers**  
✅ **Remove outliers → Fit scaler on train → Apply to test**

❌ **Transforming features based on train+test combined distribution**  
✅ **Fit all transformations on train ONLY, apply to test**

❌ **Log-transforming negative values**  
✅ **Use log1p (log1p(x)) or reflect first: log(max-x)**

❌ **Removing "outliers" without understanding why they exist**  
✅ **Investigate first, decide if real signal or noise**

❌ **One-hot encoding high-cardinality features (1000+ categories)**  
✅ **Group rare categories or use target-encoding**

---

## Stock Market Specifics

For **StockAxis** and similar stock/market data:

| Feature Type         | Typical Distribution      | Action            |
| -------------------- | ------------------------- | ----------------- |
| Price                | Log-normal (right-skewed) | Log transform     |
| Returns              | Approximately normal      | Minimal transform |
| Volume               | Heavily right-skewed      | Log transform     |
| Volatility           | Right-skewed              | Sqrt or log       |
| Technical indicators | Varies                    | Inspect each      |
| Time between events  | Exponential               | Log transform     |
| Sequential counts    | Poisson-like              | Keep or log       |

**Golden Rule:** Stock data is almost always right-skewed. Log-transform first, ask questions later.

---
