"""
Distribution Inspection Utilities

Provides automated feature distribution analysis, outlier detection,
and visualization tools for understanding data before modeling.

Usage:
    from src.distribution_inspection import FeatureInspector, TransformationAdvisor
    
    inspector = FeatureInspector(df, target_column='target')
    inspector.generate_report()
    inspector.plot_all_distributions()
    
    advisor = TransformationAdvisor(df)
    recommendations = advisor.get_recommendations()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, boxcox
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class FeatureInspector:
    """
    Comprehensive feature distribution inspector.
    
    Analyzes numerical and categorical features to understand:
    - Shape (normal, skewed, bimodal, etc.)
    - Outliers (using IQR method)
    - Missing values
    - Class balance (for target variable)
    - Feature-target relationships
    """
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """
        Initialize inspector.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        target_column : str, optional
            Name of target variable for supervised learning analysis
        """
        self.df = df
        self.target_column = target_column
        self.numerical_features = df.select_dtypes(
            include=['int64', 'int32', 'float64', 'float32']
        ).columns.tolist()
        self.categorical_features = df.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()
        
        # Remove target from feature lists
        if target_column:
            if target_column in self.numerical_features:
                self.numerical_features.remove(target_column)
            if target_column in self.categorical_features:
                self.categorical_features.remove(target_column)
    
    def inspect_numerical(self, column: str) -> Dict:
        """
        Detailed inspection of a numerical feature.
        
        Parameters:
        -----------
        column : str
            Feature name
            
        Returns:
        --------
        dict with keys: mean, std, min, max, q25, median, q75, skewness, 
                       kurtosis, missing_pct, outlier_count, outlier_pct,
                       outlier_bounds
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        data = self.df[column].dropna()
        
        # Calculate quartiles and IQR
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        # Outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        # Calculate statistics
        stats = {
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'q25': float(q1),
            'median': float(data.median()),
            'q75': float(q3),
            'iqr': float(iqr),
            'skewness': float(skew(data)),
            'kurtosis': float(kurtosis(data)),
            'missing_count': int(self.df[column].isna().sum()),
            'missing_pct': float(self.df[column].isna().sum() / len(self.df) * 100),
            'outlier_count': int(len(outliers)),
            'outlier_pct': float(len(outliers) / len(data) * 100),
            'outlier_bounds': (float(lower_bound), float(upper_bound)),
            'data_type': str(self.df[column].dtype)
        }
        
        return stats
    
    def inspect_categorical(self, column: str) -> Dict:
        """
        Detailed inspection of a categorical feature.
        
        Parameters:
        -----------
        column : str
            Feature name
            
        Returns:
        --------
        dict with keys: unique_count, top_category, top_category_pct,
                       rare_categories (< 5%), rare_categories_pct,
                       missing_count, missing_pct, entropy, data_type
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        value_counts = self.df[column].value_counts()
        total = len(self.df)
        
        # Percentages
        pct = (value_counts / total * 100)
        
        # Rare categories (< 5%)
        rare_threshold = 5
        rare_categories = (pct < rare_threshold).sum()
        
        # Calculate entropy (measure of disorder/diversity)
        probabilities = value_counts / len(self.df[column].dropna())
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        stats = {
            'unique_count': int(self.df[column].nunique()),
            'top_category': str(value_counts.index[0]) if len(value_counts) > 0 else None,
            'top_category_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'top_category_pct': float(pct.iloc[0]) if len(value_counts) > 0 else 0,
            'rare_categories_count': int(rare_categories),
            'rare_categories_pct': float(rare_categories / len(pct) * 100) if len(pct) > 0 else 0,
            'missing_count': int(self.df[column].isna().sum()),
            'missing_pct': float(self.df[column].isna().sum() / len(self.df) * 100),
            'entropy': float(entropy),
            'data_type': str(self.df[column].dtype)
        }
        
        return stats
    
    def generate_report(self) -> None:
        """Generate and print comprehensive inspection report."""
        print("\n" + "=" * 80)
        print("FEATURE DISTRIBUTION INSPECTION REPORT")
        print("=" * 80)
        
        # Dataset overview
        print(f"\nDataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Numerical Features: {len(self.numerical_features)}")
        print(f"Categorical Features: {len(self.categorical_features)}")
        
        # Target info
        if self.target_column:
            print(f"\nTarget Column: {self.target_column} ({self.df[self.target_column].dtype})")
            if self.target_column in self.categorical_features or \
               self.df[self.target_column].dtype == 'object':
                target_stats = self.inspect_categorical(self.target_column)
                print(f"  Classes: {target_stats['unique_count']}")
                print(f"  Majority class: {target_stats['top_category']} "
                      f"({target_stats['top_category_pct']:.1f}%)")
        
        # Numerical features
        print("\n" + "-" * 80)
        print("NUMERICAL FEATURES")
        print("-" * 80)
        
        for col in self.numerical_features:
            stats = self.inspect_numerical(col)
            print(f"\n{col}:")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Median: {stats['median']:.3f}")
            print(f"  Skewness: {stats['skewness']:.3f}", end="")
            
            if abs(stats['skewness']) < 0.5:
                print(" (approximately normal)")
            elif stats['skewness'] > 0.5:
                print(" (right-skewed → consider log transform)")
            else:
                print(" (left-skewed → consider inverse/reflection)")
            
            print(f"  Outliers: {stats['outlier_count']} ({stats['outlier_pct']:.1f}%)")
            if stats['missing_pct'] > 0:
                print(f"  ⚠️  Missing: {stats['missing_count']} ({stats['missing_pct']:.1f}%)")
        
        # Categorical features
        print("\n" + "-" * 80)
        print("CATEGORICAL FEATURES")
        print("-" * 80)
        
        for col in self.categorical_features:
            stats = self.inspect_categorical(col)
            print(f"\n{col}:")
            print(f"  Unique values: {stats['unique_count']}")
            print(f"  Top category: {stats['top_category']} "
                  f"({stats['top_category_pct']:.1f}%)")
            print(f"  Rare categories: {stats['rare_categories_count']}")
            if stats['missing_pct'] > 0:
                print(f"  ⚠️  Missing: {stats['missing_count']} ({stats['missing_pct']:.1f}%)")
        
        print("\n" + "=" * 80)
    
    def plot_all_distributions(self, figsize: Tuple = None) -> None:
        """
        Generate distribution plots for all features.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height)
        """
        # Numerical features
        if self.numerical_features:
            n_features = len(self.numerical_features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (15, 5 * n_rows))
            if n_features == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, col in enumerate(self.numerical_features):
                data = self.df[col].dropna()
                axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                
                # Add statistics to title
                skew_val = skew(data)
                axes[idx].set_title(f'{col}\n(Skew: {skew_val:.2f})', fontweight='bold')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(axis='y', alpha=0.3)
                
                # Add median line
                median_val = data.median()
                axes[idx].axvline(median_val, color='red', linestyle='--', 
                                 linewidth=2, alpha=0.7, label=f'Median: {median_val:.2f}')
                axes[idx].legend()
            
            # Hide empty subplots
            for idx in range(len(self.numerical_features), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('Numerical Feature Distributions', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        # Categorical features
        if self.categorical_features:
            n_features = len(self.categorical_features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (15, 5 * n_rows))
            if n_features == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, col in enumerate(self.categorical_features):
                value_counts = self.df[col].value_counts()
                axes[idx].bar(range(len(value_counts)), value_counts.values, 
                            edgecolor='black', alpha=0.7, color='steelblue')
                axes[idx].set_xticks(range(len(value_counts)))
                axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
                axes[idx].set_title(f'{col} ({len(value_counts)} unique)', fontweight='bold')
                axes[idx].set_ylabel('Count')
                axes[idx].grid(axis='y', alpha=0.3)
            
            # Hide empty subplots
            for idx in range(len(self.categorical_features), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('Categorical Feature Distributions', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()


class TransformationAdvisor:
    """
    Provides transformation recommendations for features.
    
    Recommends appropriate transformations (log, sqrt, Box-Cox, scaling)
    based on distribution characteristics.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize advisor."""
        self.df = df
        self.inspector = FeatureInspector(df)
    
    def get_recommendations(self) -> Dict[str, Dict]:
        """
        Get transformation recommendations for all features.
        
        Returns:
        --------
        dict mapping feature names to recommendation dicts with keys:
            - transform_type: (none, log, sqrt, boxcox, standard_scale, minmax_scale)
            - reason: explanation
            - priority: (low, medium, high)
        """
        recommendations = {}
        
        # Numerical features
        for col in self.inspector.numerical_features:
            stats = self.inspector.inspect_numerical(col)
            recommendations[col] = self._recommend_numerical_transform(col, stats)
        
        # Categorical features
        for col in self.inspector.categorical_features:
            stats = self.inspector.inspect_categorical(col)
            recommendations[col] = self._recommend_categorical_transform(col, stats)
        
        return recommendations
    
    def _recommend_numerical_transform(self, column: str, stats: Dict) -> Dict:
        """Recommend transformation for numerical feature."""
        skew_val = stats['skewness']
        kurtosis_val = stats['kurtosis']
        outlier_pct = stats['outlier_pct']
        
        recommendation = {
            'feature': column,
            'type': 'numerical',
            'transform_type': 'none',
            'reason': [],
            'priority': 'low'
        }
        
        # Check skewness
        if abs(skew_val) > 2.0:
            recommendation['transform_type'] = 'boxcox'
            recommendation['reason'].append(f"Highly skewed (|skew|={abs(skew_val):.2f})")
            recommendation['priority'] = 'high'
        elif abs(skew_val) > 1.0:
            recommendation['transform_type'] = 'log' if skew_val > 0 else 'reflect_log'
            recommendation['reason'].append(f"Moderately skewed (|skew|={abs(skew_val):.2f})")
            recommendation['priority'] = 'high'
        elif abs(skew_val) > 0.5:
            recommendation['transform_type'] = 'log' if skew_val > 0 else 'reflect_log'
            recommendation['reason'].append(f"Slightly skewed (|skew|={abs(skew_val):.2f})")
            recommendation['priority'] = 'medium'
        
        # Check outliers
        if outlier_pct > 5:
            recommendation['reason'].append(f"High outlier %: {outlier_pct:.1f}%")
            recommendation['priority'] = 'high'
            if recommendation['transform_type'] == 'none':
                recommendation['transform_type'] = 'outlier_handling'
        
        # Check kurtosis (fat tails)
        if kurtosis_val > 5:
            recommendation['reason'].append("Heavy-tailed (fat tails)")
            recommendation['priority'] = 'high'
        
        # Format reason
        if recommendation['reason']:
            recommendation['reason'] = '; '.join(recommendation['reason'])
        else:
            recommendation['reason'] = "Feature appears normally distributed"
        
        return recommendation
    
    def _recommend_categorical_transform(self, column: str, stats: Dict) -> Dict:
        """Recommend transformation for categorical feature."""
        unique_count = stats['unique_count']
        rare_pct = stats['rare_categories_pct']
        
        recommendation = {
            'feature': column,
            'type': 'categorical',
            'transform_type': 'none',
            'reason': [],
            'priority': 'low'
        }
        
        # Check cardinality
        if unique_count > 100:
            recommendation['transform_type'] = 'group_rare'
            recommendation['reason'].append(f"High cardinality: {unique_count} categories")
            recommendation['priority'] = 'high'
        elif unique_count > 20:
            recommendation['transform_type'] = 'group_rare'
            recommendation['reason'].append(f"Moderate cardinality: {unique_count} categories")
            recommendation['priority'] = 'medium'
        elif unique_count > 10:
            recommendation['transform_type'] = 'onehot'
            recommendation['reason'].append(f"Moderate cardinality: {unique_count} categories")
            recommendation['priority'] = 'medium'
        elif unique_count == 2:
            recommendation['transform_type'] = 'binary_encode'
            recommendation['reason'].append("Binary feature")
        else:
            recommendation['transform_type'] = 'onehot'
            recommendation['reason'].append(f"Categorical with {unique_count} levels")
        
        # Check for rare categories
        if rare_pct > 10:
            recommendation['reason'].append(f"Many rare categories: {rare_pct:.1f}%")
            if 'group_rare' not in recommendation['transform_type']:
                recommendation['transform_type'] = 'group_rare'
            recommendation['priority'] = 'high'
        
        # Format reason
        if recommendation['reason']:
            recommendation['reason'] = '; '.join(recommendation['reason'])
        else:
            recommendation['reason'] = "Standard categorical encoding recommended"
        
        return recommendation
    
    def print_recommendations(self) -> None:
        """Print formatted recommendations."""
        recommendations = self.get_recommendations()
        
        print("\n" + "=" * 80)
        print("TRANSFORMATION RECOMMENDATIONS")
        print("=" * 80)
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_recs = sorted(recommendations.items(), 
                            key=lambda x: priority_order.get(x[1]['priority'], 3))
        
        for feature, rec in sorted_recs:
            priority_symbol = "🔴" if rec['priority'] == 'high' else \
                            "🟡" if rec['priority'] == 'medium' else "🟢"
            
            print(f"\n{priority_symbol} {feature} ({rec['type']})")
            print(f"  Transform: {rec['transform_type']}")
            print(f"  Reason: {rec['reason']}")
            print(f"  Priority: {rec['priority'].upper()}")


def consolidate_rare_categories(df: pd.DataFrame, column: str, 
                               threshold_pct: float = 5.0) -> pd.DataFrame:
    """
    Group categories with < threshold% into 'Other'.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Categorical column name
    threshold_pct : float
        Percentage threshold for rare categories (default: 5%)
        
    Returns:
    --------
    pd.DataFrame
        Modified dataframe with rare categories grouped
    """
    pct = df[column].value_counts(normalize=True) * 100
    rare_categories = pct[pct < threshold_pct].index
    
    df_copy = df.copy()
    df_copy[column] = df_copy[column].apply(
        lambda x: 'Other' if x in rare_categories else x
    )
    
    print(f"Consolidated {len(rare_categories)} rare categories in '{column}'")
    print(f"Original unique: {df[column].nunique()}, "
          f"After: {df_copy[column].nunique()}")
    
    return df_copy


# Example usage
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'age': np.random.normal(45, 15, 1000),
        'income': np.random.exponential(50000, 1000),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'status': np.random.choice(['Active', 'Inactive'], 1000),
        'target': np.random.binomial(1, 0.3, 1000)
    })
    
    # Run inspection
    inspector = FeatureInspector(sample_df, target_column='target')
    inspector.generate_report()
    inspector.plot_all_distributions()
    
    # Get transformation recommendations
    advisor = TransformationAdvisor(sample_df)
    advisor.print_recommendations()
