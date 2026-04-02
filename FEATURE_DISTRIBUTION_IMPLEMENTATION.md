# Feature Distribution Inspection: Implementation Guide

**Objective:** Learn how to inspect feature distributions to make informed preprocessing decisions before modeling.

**Time Estimate:** 30-45 minutes of implementation work  
**Difficulty:** Intermediate  
**Prerequisites:** Understanding of features/targets (Lesson 2), basic pandas/matplotlib

---

## Quick Navigation

- [Part 1: Numerical Feature Inspection](#part-1-numerical-feature-inspection) (5 min read)
- [Part 2: Categorical Feature Inspection](#part-2-categorical-feature-inspection) (5 min read)
- [Part 3: Target-Feature Relationships](#part-3-target-feature-relationships) (5 min read)
- [Part 4: Creating Inspection Scripts](#part-4-creating-inspection-scripts) (10-15 min implementation)
- [Part 5: Decision Trees](#part-5-decision-trees) (Reference)
- [Troubleshooting](#troubleshooting)

---

## Part 1: Numerical Feature Inspection

### Summary Statistics (Baseline)

**What to do first:**

```python
import pandas as pd

# Load your data
df = pd.read_csv('data/processed/your_data.csv')

# First look
print(df.describe())
print(df.dtypes)
```

**What to look for:**

- **std >> mean**: High variability, likely needs scaling
- **min = 0, max >>> 0**: Right-skewed, needs log transform
- **25%, 50%, 75% clustered**: Distribution is skewed or has boundary
- **Missing count != total**: NaN values present

### Visualizing Distributions

**Histograms show the shape:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Single feature
plt.hist(df['your_feature'], bins=30, edgecolor='black')
plt.title('Distribution of your_feature')
plt.show()

# All numerical features at once
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

fig, axes = plt.subplots(len(numerical_features) // 3 + 1, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(numerical_features):
    axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(col)
    axes[idx].axvline(df[col].median(), color='red', linestyle='--', label='Median')
    axes[idx].legend()

plt.suptitle('Numerical Features Distribution Grid')
plt.tight_layout()
plt.show()
```

**Shape Interpretation:**
| Pattern | Meaning | Action |
|---------|---------|--------|
| Bell curve | Normal distribution | ✓ OK for linear models |
| Right tail | Right-skewed | Log or sqrt transform |
| Left tail | Left-skewed | 1/x or reflect+log |
| Flat | Uniform/capped | Check boundaries |
| Two peaks | Bimodal | Separate by category |

### Detecting Outliers

**IQR Method (Standard Approach):**

```python
from scipy import stats

def find_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    print(f"Feature: {column}")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")

    return outliers

# Find outliers in each numerical column
for col in numerical_features:
    outliers = find_outliers_iqr(df, col)
```

**Visualizing Outliers:**

```python
# Boxplots show outliers as dots
fig, axes = plt.subplots(1, len(numerical_features), figsize=(4*len(numerical_features), 4))
if len(numerical_features) == 1:
    axes = [axes]

for idx, col in enumerate(numerical_features):
    bp = axes[idx].boxplot(df[col], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[idx].set_title(col)
    axes[idx].set_ylabel('Value')

plt.suptitle('Outlier Detection: Boxplots')
plt.tight_layout()
plt.show()
```

**Deciding What to Do with Outliers:**

```
Is the outlier:
├─ Measurement error (data entry mistake)?
│  └─ REMOVE or CORRECT
├─ Valid but extreme event?
│  ├─ Important for prediction?
│  │  └─ KEEP (but may need robust scaling)
│  └─ Noise/rare?
│     └─ CAP at threshold using IQR bounds
└─ Natural variation?
   └─ KEEP (especially if it's correlated with target)
```

### Skewness & Kurtosis

**Calculate and interpret:**

```python
from scipy.stats import skew, kurtosis

for col in numerical_features:
    skew_val = skew(df[col])
    kurt_val = kurtosis(df[col])

    print(f"\n{col}:")
    print(f"  Skewness: {skew_val:.3f}", end="")
    if abs(skew_val) < 0.5:
        print(" → Approximately Normal")
    elif skew_val > 0.5:
        print(" → RIGHT-SKEWED (>0.5)")
    else:
        print(" → LEFT-SKEWED (<-0.5)")

    print(f"  Kurtosis: {kurt_val:.3f}", end="")
    if kurt_val > 3:
        print(" → Heavy-tailed (fat tails, outlier risk)")
    else:
        print(" → Normal or light-tailed")
```

**Transformation Recommendations:**

| Skewness    | Kurtosis | Recommendation                     |
| ----------- | -------- | ---------------------------------- |
| < -0.5      | Any      | Inverse transform (1/x) or reflect |
| -0.5 to 0.5 | < 3      | No transform needed                |
| -0.5 to 0.5 | > 3      | Scaling sufficient, watch outliers |
| 0.5 to 1.0  | Any      | Log transform or sqrt              |
| > 1.0       | > 3      | Box-Cox transform or log           |

---

## Part 2: Categorical Feature Inspection

### Value Counts & Cardinality

**Basic inspection:**

```python
categorical_features = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_features:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Missing: {df[col].isna().sum()}")
    print("\n  Value counts:")
    vc = df[col].value_counts()
    pct = (vc / len(df) * 100).round(1)
    for category, count in vc.items():
        p = pct[category]
        marker = " ⚠️ RARE" if p < 5 else ""
        print(f"    {category}: {count} ({p}%){marker}")
```

### Class Balance Visualization

```python
fig, axes = plt.subplots(1, len(categorical_features), figsize=(5*len(categorical_features), 4))
if len(categorical_features) == 1:
    axes = [axes]

for idx, col in enumerate(categorical_features):
    vc = df[col].value_counts()
    axes[idx].bar(range(len(vc)), vc.values, edgecolor='black', alpha=0.7)
    axes[idx].set_xticks(range(len(vc)))
    axes[idx].set_xticklabels(vc.index, rotation=45, ha='right')
    axes[idx].set_title(col)
    axes[idx].set_ylabel('Count')
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Categorical Feature Distributions')
plt.tight_layout()
plt.show()
```

### Handling Rare Categories

```python
def consolidate_rare_categories(df, column, threshold_pct=5):
    """
    Group categories with < threshold% into 'Other'
    """
    pct = df[column].value_counts(normalize=True) * 100
    rare_categories = pct[pct < threshold_pct].index

    df_copy = df.copy()
    df_copy[column] = df_copy[column].apply(
        lambda x: 'Other' if x in rare_categories else x
    )

    print(f"Consolidated {len(rare_categories)} rare categories in {column}")
    print(f"New distribution:\n{df_copy[column].value_counts()}")

    return df_copy

# Example: Consolidate rare categories
df_consolidated = consolidate_rare_categories(df, 'your_categorical_col', threshold_pct=5)
```

---

## Part 3: Target-Feature Relationships

### Understanding Class Balance (Classification)

```python
from sklearn.model_selection import train_test_split

# Check target distribution
print("TARGET DISTRIBUTION:")
vc = df[TARGET_COLUMN].value_counts()
pct = (vc / len(df) * 100).round(1)
for val, count in vc.items():
    p = pct[val]
    print(f"  Class {val}: {count} ({p}%)")

# Visualize
plt.figure(figsize=(8, 4))
vc.plot(kind='bar', edgecolor='black', alpha=0.7)
plt.title('Target Class Distribution')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Check if stratification is needed
imbalance_ratio = vc.max() / vc.min()
print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
if imbalance_ratio > 3:
    print("⚠️ Severe imbalance detected! Use stratified_split and class_weights")
```

### Feature-Target Separability

**For Numerical Features:**

```python
# Compare distributions by target class
fig, axes = plt.subplots(1, len(numerical_features), figsize=(5*len(numerical_features), 4))
if len(numerical_features) == 1:
    axes = [axes]

for idx, col in enumerate(numerical_features):
    for target_val in df[TARGET_COLUMN].unique():
        data = df[df[TARGET_COLUMN] == target_val][col]
        axes[idx].hist(data, bins=20, alpha=0.6, label=f'{TARGET_COLUMN}={target_val}', edgecolor='black')

    axes[idx].set_title(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Numerical Features - Do They Separate Classes?')
plt.tight_layout()
plt.show()
```

**For Categorical Features:**

```python
# Cross-tabulation
for col in categorical_features:
    print(f"\n{col} vs {TARGET_COLUMN}:")
    crosstab = pd.crosstab(df[col], df[TARGET_COLUMN], normalize='index') * 100
    print(crosstab.round(1))

    # Visualize
    pd.crosstab(df[col], df[TARGET_COLUMN]).plot(kind='bar', stacked=False,
                                                    edgecolor='black', alpha=0.7)
    plt.title(f'{col} vs {TARGET_COLUMN}')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=TARGET_COLUMN)
    plt.tight_layout()
    plt.show()
```

---

## Part 4: Creating Inspection Scripts

### Create a Reusable Inspection Module

Save this as `src/distribution_inspection.py`:

```python
"""
Feature distribution inspection utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.stats import boxcox

class FeatureInspector:
    """Comprehensive feature distribution inspection."""

    def __init__(self, df, target_column=None):
        self.df = df
        self.target_column = target_column
        self.numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns

    def inspect_numerical(self, column):
        """Detailed numerical feature inspection."""
        data = self.df[column].dropna()
        stats_dict = {
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q25': data.quantile(0.25),
            'median': data.quantile(0.50),
            'q75': data.quantile(0.75),
            'skewness': skew(data),
            'kurtosis': kurtosis(data),
            'missing_pct': self.df[column].isna().sum() / len(self.df) * 100
        }

        # Outliers
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        outlier_bounds = (q1 - 1.5*iqr, q3 + 1.5*iqr)
        outliers = data[(data < outlier_bounds[0]) | (data > outlier_bounds[1])]
        stats_dict['outlier_count'] = len(outliers)
        stats_dict['outlier_pct'] = len(outliers) / len(data) * 100

        return stats_dict

    def inspect_categorical(self, column):
        """Detailed categorical feature inspection."""
        vc = self.df[column].value_counts()
        pct = (vc / len(self.df) * 100)

        rare_threshold = 5
        rare_count = (pct < rare_threshold).sum()

        stats_dict = {
            'unique_count': self.df[column].nunique(),
            'top_category': vc.index[0],
            'top_category_pct': pct.iloc[0],
            'rare_categories': rare_count,
            'missing_pct': self.df[column].isna().sum() / len(self.df) * 100
        }

        return stats_dict

    def plot_all_distributions(self):
        """Generate inspection plots for all features."""
        # Numerical
        fig, axes = plt.subplots((len(self.numerical_features) + 2) // 3, 3,
                                 figsize=(15, 5 * ((len(self.numerical_features) + 2) // 3)))
        axes = axes.flatten()

        for idx, col in enumerate(self.numerical_features):
            axes[idx].hist(self.df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            skew_val = skew(self.df[col].dropna())
            axes[idx].set_title(f'{col} (Skew: {skew_val:.2f})')
            axes[idx].grid(axis='y', alpha=0.3)

        # Hide empty subplots
        for idx in range(len(self.numerical_features), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Numerical Feature Distributions')
        plt.tight_layout()
        plt.show()

        # Categorical
        fig, axes = plt.subplots((len(self.categorical_features) + 2) // 3, 3,
                                 figsize=(15, 5 * ((len(self.categorical_features) + 2) // 3)))
        axes = axes.flatten()

        for idx, col in enumerate(self.categorical_features):
            vc = self.df[col].value_counts()
            axes[idx].bar(range(len(vc)), vc.values, edgecolor='black', alpha=0.7)
            axes[idx].set_xticklabels(vc.index, rotation=45, ha='right')
            axes[idx].set_title(col)
            axes[idx].grid(axis='y', alpha=0.3)

        # Hide empty subplots
        for idx in range(len(self.categorical_features), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Categorical Feature Distributions')
        plt.tight_layout()
        plt.show()

# Usage Example:
def main():
    df = pd.read_csv('data/processed/your_data.csv')

    inspector = FeatureInspector(df, target_column='target')

    # Inspect specific features
    for col in inspector.numerical_features:
        stats = inspector.inspect_numerical(col)
        print(f"\n{col}: Skewness={stats['skewness']:.3f}, "
              f"Outliers={stats['outlier_pct']:.1f}%")

    # Plot all
    inspector.plot_all_distributions()

if __name__ == '__main__':
    main()
```

### Use in Your Pipeline

```python
# In your data preprocessing script
from src.distribution_inspection import FeatureInspector

df = load_data()
inspector = FeatureInspector(df, target_column='target')

# Generate report
print("=" * 80)
print("FEATURE DISTRIBUTION INSPECTION REPORT")
print("=" * 80)

for col in inspector.numerical_features:
    stats = inspector.inspect_numerical(col)
    print(f"\n{col}:")
    print(f"  Skewness: {stats['skewness']:.3f}")
    print(f"  Outliers: {stats['outlier_pct']:.1f}%")

for col in inspector.categorical_features:
    stats = inspector.inspect_categorical(col)
    print(f"\n{col}:")
    print(f"  Unique: {stats['unique_count']}")
    print(f"  Rare categories: {stats['rare_categories']}")

# Generate visualizations
inspector.plot_all_distributions()
```

---

## Part 5: Decision Trees

### Numerical Feature Decision Tree

```
NUMERICAL FEATURE PROCESSING:

1. Check missing values
   ├─ < 5%: Impute with mean/median
   └─ > 5%: Investigate why, create 'missing' flag, then impute

2. Detect and handle outliers
   ├─ Use IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
   ├─ Percentage outliers?
   │  ├─ < 1%: Remove or cap
   │  ├─ 1-5%: Cap at bounds
   │  └─ > 5%: Keep (likely natural variation)
   └─ For financial/stock data: Often valid, cap rather than remove

3. Check skewness
   ├─ |skewness| < 0.5: ✓ OK, proceed to step 4
   ├─ 0.5 < skewness < 1.5: Log transform
   ├─ skewness > 1.5: Box-Cox transform
   └─ Negative skew: Reflect then log

4. Choose scaling based on model
   ├─ Tree-based (RandomForest, Gradient Boosting): Skip scaling
   ├─ Linear Models (Logistic Regression, Ridge): StandardScaler after transform
   ├─ Distance-based (KNN, SVM): StandardScaler after transform
   └─ Neural Networks: MinMaxScaler [0,1] or StandardScaler

5. Ready for modeling!
```

### Categorical Feature Decision Tree

```
CATEGORICAL FEATURE PROCESSING:

1. Check missing values
   ├─ If < 5%: Create 'Missing' category or mode-impute
   └─ If > 5%: Investigate, may be informative, keep separate

2. Check cardinality
   ├─ 2 categories: Binary, encode as 0/1
   ├─ 3-10 categories: One-hot encode
   ├─ 10-100 categories: Group rare (< 5%) into 'Other', then encode
   └─ > 100 categories: Likely needs aggregation or embedding

3. Check if ordinal
   ├─ YES (e.g., size: Small < Medium < Large): Use ordinal encoding
   └─ NO (e.g., color, region): Use one-hot encoding

4. Check class imbalance (for target variable)
   ├─ Balanced (hand-waving): Proceed normally
   ├─ Imbalanced > 3:1: Use stratified split + class weight
   └─ Severely imbalanced: Consider resampling or threshold tuning

5. Ready for modeling!
```

---

## Troubleshooting

### Problem: Histogram shows all zeros then spike

**Cause:** Highly skewed feature with many near-zero values  
**Solution:** Log transform or check for capped/bounded data

```python
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(df['feature'], bins=50)
plt.title('Original (Skewed)')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(df['feature']), bins=50)
plt.title('After Log Transform')
plt.show()
```

### Problem: Boxplot has many outlier dots

**Cause:** Heavy-tailed distribution or actually extreme values  
**Solution:** Investigate, don't blindly remove

```python
# Are they real or noise?
outliers_Q1 = df[df[col] < Q1 - 1.5*IQR]
outliers_Q3 = df[df[col] > Q3 + 1.5*IQR]

print(f"Low outliers: {len(outliers_Q1)}")
print(f"High outliers: {len(outliers_Q3)}")
print(f"\nSample outlier values:")
print(df[col][pd.concat([outliers_Q1.index, outliers_Q3.index])].sort_values().head(10))

# If they're meaningful, cap instead of remove
df[col + '_capped'] = df[col].clip(lower=lower_bound, upper=upper_bound)
```

### Problem: Most categorical values are rare

**Cause:** High cardinality feature with long-tail distribution  
**Solution:** Aggressive grouping or embedding

```python
# Consolidate to top categories + 'Other'
top_n = 10
top_categories = df[col].value_counts().head(top_n).index
df[col + '_grouped'] = df[col].apply(
    lambda x: x if x in top_categories else 'Other'
)

print(f"Before: {df[col].nunique()} unique")
print(f"After: {df[col + '_grouped'].nunique()} unique")
print(f"\nNew distribution:\n{df[col + '_grouped'].value_counts()}")
```

---

## Next Steps

1. **Run the notebook:** Execute all cells in `notebooks/inspecting_feature_distributions.ipynb`
2. **Apply to your data:** Use `FeatureInspector` on your actual dataset
3. **Document findings:** Record skewness, outliers, imbalances for each feature
4. **Make preprocessing decisions:** Use decision trees above
5. **Implement in preprocessing:** Update `src/data_preprocessing.py` with transformations
6. **Verify results:** Re-inspect after transformations
7. **Learn causally:** Play with different transforms, see impact on distributions

---
