"""
Feature and Target Configuration for StockAxis

This module centralizes all feature and target definitions for the ML pipeline.
All other modules import from this configuration to ensure consistency and
prevent accidental data leakage or feature misconfigurations.

Before modifying this file, carefully consider:
1. Is this feature available at prediction time?
2. Is this derived from the target (leakage)?
3. Is this post-outcome information?
4. Is this a raw timestamp (should be derived)?
"""

from typing import List, Dict

# ============================================================================
# TARGET VARIABLE CONFIGURATION
# ============================================================================

TARGET_COLUMN: str = 'target'  # TODO: Change to your actual target column

TARGET_TYPE: str = 'binary_classification'  # Options: 'binary_classification', 
                                            # 'multi_class', 'regression'

# Document what each target value means in business terms
TARGET_MEANING: Dict = {
    # TODO: Fill in your actual target values
    # Example for binary:
    # 0: 'Negative outcome (e.g., stock will not increase)',
    # 1: 'Positive outcome (e.g., stock will increase)',
}

# ============================================================================
# NUMERICAL FEATURES
# ============================================================================
# Features with continuous or discrete numerical values
# These can be used directly by models but may need scaling

NUMERICAL_FEATURES: List[str] = [
    # TODO: Add your numerical features
    # Example:
    # 'feature_1',
    # 'feature_2',
]

# ============================================================================
# CATEGORICAL FEATURES
# ============================================================================
# Features with discrete categories (no intrinsic ordering)
# These must be encoded (one-hot, label, target encoding) before use

CATEGORICAL_FEATURES: List[str] = [
    # TODO: Add your categorical features
    # Example:
    # 'category_1',
    # 'category_2',
]

# ============================================================================
# ORDINAL FEATURES
# ============================================================================
# Features with discrete categories AND natural ordering
# Example: Education Level (HS < Bachelor < Master < PhD)
# Format: 'feature_name': [value_low, value_medium, value_high]

ORDINAL_FEATURES: Dict[str, List[str]] = {
    # TODO: Add ordinal features if any
    # Example:
    # 'education_level': ['High School', 'Bachelor', 'Master', 'PhD'],
}

# ============================================================================
# COMBINED FEATURE LIST
# ============================================================================
# This is used throughout the pipeline

ALL_FEATURES: List[str] = (
    NUMERICAL_FEATURES + 
    CATEGORICAL_FEATURES + 
    list(ORDINAL_FEATURES.keys())
)

# ============================================================================
# EXCLUDED COLUMNS
# ============================================================================
# Columns present in dataset but NOT used as features
# Document WHY each is excluded

EXCLUDED_COLUMNS: Dict[str, str] = {
    # TODO: Document excluded columns and reasons
    # Format: 'column_name': 'Reason for exclusion'
    # Examples:
    # 'id_column': 'Unique identifier - no predictive value',
    # 'target_derived_column': 'Calculated from target - data leakage',
    # 'raw_timestamp': 'Raw timestamp - will use derived temporal features',
}

# ============================================================================
# FEATURE AVAILABILITY DOCUMENTATION
# ============================================================================
# For each feature, document WHEN it becomes available
# This prevents temporal/future leakage

FEATURE_AVAILABILITY: Dict[str, str] = {
    # TODO: Document when each feature is available
    # Format: 'feature_name': 'When it's available'
    # Examples:
    # 'open_price': 'Available at market open (known at prediction time)',
    # 'close_price': 'Available at market close (known at prediction time)',
    # 'close_price_tomorrow': 'Not available - future data (EXCLUDE)',
}

# ============================================================================
# VALIDATION AND VERIFICATION
# ============================================================================

def validate_configuration() -> bool:
    """
    Validate feature configuration before training.
    
    Checks:
    - No overlap between features and excluded columns
    - Target not in feature list
    - All categories documented
    - No obvious naming issues
    
    Raises:
        ValueError: If configuration is invalid
        
    Returns:
        bool: True if valid (or raises exception)
    """
    
    errors = []
    
    # Check 1: Target not in features
    if TARGET_COLUMN in ALL_FEATURES:
        errors.append(f"❌ Target '{TARGET_COLUMN}' found in feature list!")
    
    # Check 2: No overlap between features and excluded
    excluded_names = set(EXCLUDED_COLUMNS.keys())
    feature_names = set(ALL_FEATURES)
    overlap = excluded_names & feature_names
    if overlap:
        errors.append(f"❌ Columns in both features and excluded: {overlap}")
    
    # Check 3: Features are not empty
    if not ALL_FEATURES:
        errors.append(f"❌ No features defined!")
    
    # Check 4: Target type is valid
    valid_types = ['binary_classification', 'multi_class', 'regression']
    if TARGET_TYPE not in valid_types:
        errors.append(f"❌ Invalid TARGET_TYPE '{TARGET_TYPE}'. Must be one of {valid_types}")
    
    # Check 5: For classification, TARGET_MEANING should be filled
    if 'classification' in TARGET_TYPE and not TARGET_MEANING:
        errors.append(f"⚠️  WARNING: TARGET_MEANING is empty for classification task")
    
    # Check 6: Feature categories don't overlap
    num_set = set(NUMERICAL_FEATURES)
    cat_set = set(CATEGORICAL_FEATURES)
    ord_set = set(ORDINAL_FEATURES.keys())
    
    num_cat_overlap = num_set & cat_set
    num_ord_overlap = num_set & ord_set
    cat_ord_overlap = cat_set & ord_set
    
    if num_cat_overlap or num_ord_overlap or cat_ord_overlap:
        errors.append(f"❌ Feature categories overlap: num-cat {num_cat_overlap}, "
                     f"num-ord {num_ord_overlap}, cat-ord {cat_ord_overlap}")
    
    # If there are errors, raise
    if errors:
        print("❌ CONFIGURATION VALIDATION FAILED:")
        for error in errors:
            print(f"  {error}")
        raise ValueError("Configuration invalid. See errors above.")
    
    # Print success message
    print("✓ Configuration Validation: PASSED")
    print(f"  - Target: {TARGET_COLUMN} ({TARGET_TYPE})")
    print(f"  - Numerical features: {len(NUMERICAL_FEATURES)}")
    print(f"  - Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"  - Ordinal features: {len(ORDINAL_FEATURES)}")
    print(f"  - Total features: {len(ALL_FEATURES)}")
    print(f"  - Excluded columns: {len(EXCLUDED_COLUMNS)}")
    
    return True


def get_feature_info() -> Dict:
    """
    Return comprehensive feature information for documentation.
    
    Returns:
        dict: Feature metadata including categories, availability, etc.
    """
    
    return {
        'target': TARGET_COLUMN,
        'target_type': TARGET_TYPE,
        'target_meaning': TARGET_MEANING,
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'ordinal_features': ORDINAL_FEATURES,
        'all_features': ALL_FEATURES,
        'excluded_columns': EXCLUDED_COLUMNS,
        'feature_availability': FEATURE_AVAILABILITY,
        'n_numerical': len(NUMERICAL_FEATURES),
        'n_categorical': len(CATEGORICAL_FEATURES),
        'n_ordinal': len(ORDINAL_FEATURES),
        'n_total_features': len(ALL_FEATURES),
        'n_excluded': len(EXCLUDED_COLUMNS),
    }


# ============================================================================
# EXECUTE VALIDATION ON IMPORT
# ============================================================================
# Uncomment the line below to validate configuration when this module is imported
# This ensures errors are caught early

# validate_configuration()  # Uncomment after filling in configuration

if __name__ == '__main__':
    """
    Template usage and testing
    """
    print("=" * 80)
    print("STOCKAXIS FEATURE CONFIGURATION")
    print("=" * 80)
    
    # Print current configuration
    info = get_feature_info()
    
    print(f"\nTarget: {info['target']} ({info['target_type']})")
    print(f"Total features: {info['n_total_features']}")
    print(f"  - Numerical: {info['n_numerical']}")
    print(f"  - Categorical: {info['n_categorical']}")
    print(f"  - Ordinal: {info['n_ordinal']}")
    print(f"Excluded columns: {info['n_excluded']}")
    
    # Validate
    try:
        validate_configuration()
    except ValueError as e:
        print(f"\n⚠️  Validation failed - fill in the TODO sections above")
