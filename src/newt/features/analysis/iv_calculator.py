from typing import Dict, Union

import pandas as pd

from newt.config import BINNING

from .woe_calculator import WOEEncoder


def calculate_iv(
    df: pd.DataFrame,
    target: str,
    feature: str,
    buckets: int = BINNING.DEFAULT_BUCKETS,
    epsilon: float = BINNING.DEFAULT_EPSILON,
) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Calculate Information Value (IV) for a feature.
    High performance implementation using vectorized operations via WOEEncoder.

    Args:
        df: Input DataFrame.
        target: Target column name (binary 0/1).
        feature: Feature column name.
        buckets: Number of buckets for numerical features (if not already binned).
        epsilon: Small constant to avoid division by zero or log(0).

    Returns:
        Dict containing 'iv' (float) and 'woe_table' (pd.DataFrame).
    """
    # If feature is numeric and has many unique values, bin it
    if pd.api.types.is_numeric_dtype(df[feature]) and df[feature].nunique() > buckets:
        try:
            # Try quantile binning first
            binned = pd.qcut(df[feature], q=buckets, duplicates="drop").astype(str)
        except ValueError:
            # Fallback to equal-width binning if quantiles fail (e.g. skewed distribution)
            binned = pd.cut(df[feature], bins=buckets).astype(str)
        
        feature_data = binned
    else:
        # Categorical or low-cardinality numeric
        feature_data = df[feature]

    encoder = WOEEncoder(epsilon=epsilon)
    encoder.fit(feature_data, df[target])

    return {"iv": float(encoder.iv_), "woe_table": encoder.summary_}
