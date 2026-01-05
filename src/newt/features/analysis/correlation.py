import pandas as pd
import numpy as np  # noqa: F401
from typing import Union, List, Dict  # noqa: F401


def calculate_correlation_matrix(
    df: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """
    Calculate correlation matrix for a DataFrame.

    Args:
        df: Input DataFrame.
        method: Correlation method ('pearson', 'kendall', 'spearman').

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.corr(method=method)


def get_high_correlation_pairs(
    corr_matrix: pd.DataFrame, threshold: float = 0.8
) -> List[Dict[str, Union[str, float]]]:
    """
    Identify pairs of variables with correlation above a threshold.

    Args:
        corr_matrix: Correlation matrix.
        threshold: Absolute correlation threshold to identify high correlation.

    Returns:
        List of dictionaries with 'var1', 'var2', and 'correlation'.
    """
    high_corr_pairs = []

    # improved efficiency: use upper triangle to avoid duplicates and self-correlation
    # keeps strict upper triangle (k=1), so diagonal is excluded

    # Mask the lower triangle and diagonal
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    # Use where to get indices of upper triangle elements
    # But standard iteration might be clearer or stack?
    # Stack approach:
    upper = corr_matrix.where(mask)

    # Stack and filter
    pairs = upper.stack()
    high_corr = pairs[pairs.abs() >= threshold]

    for (var1, var2), val in high_corr.items():
        high_corr_pairs.append(
            {"var1": var1, "var2": var2, "correlation": float(val)}
        )

    # Sort by absolute correlation descending
    high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return high_corr_pairs
