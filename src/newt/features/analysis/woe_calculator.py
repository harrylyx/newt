"""
WOE (Weight of Evidence) calculation and encoding.

Provides WOE transformation for binned/categorical features.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class WOEEncoder:
    """
    Weight of Evidence (WoE) Encoder.

    Encodes features using WoE values based on a binary target.
    Designed to work with already binned data (from Binner) or categorical features.

    For numeric data, first use Binner to bin the data, then apply WOEEncoder.

    Examples
    --------
    >>> # With already binned data
    >>> woe = WOEEncoder()
    >>> woe.fit(X_binned, y)
    >>> X_woe = woe.transform(X_binned)
    >>>
    >>> # With categorical data
    >>> woe = WOEEncoder()
    >>> woe.fit(df['category_col'], df['target'])
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize WOEEncoder.

        Parameters
        ----------
        epsilon : float
            Smoothing factor to avoid division by zero. Default 1e-8.
        """
        self.epsilon = epsilon
        self.woe_map_: Dict[Any, float] = {}
        self.iv_: float = 0.0
        self.summary_: pd.DataFrame = pd.DataFrame()
        self.is_fitted_: bool = False

    def fit(self, X: pd.Series, y: pd.Series) -> "WOEEncoder":
        """
        Fit the WoE encoder to the data.

        Parameters
        ----------
        X : pd.Series
            Feature data - can be binned numeric (codes or intervals) or categorical.
        y : pd.Series
            Target data (binary 0/1).

        Returns
        -------
        WOEEncoder
            Fitted instance.
        """
        X = X.copy()
        y = y.copy()

        # Convert to string for consistent mapping
        # This handles: integers, floats, categories, intervals, strings
        X_str = X.astype(str)

        # Create temporary dataframe for aggregation
        df = pd.DataFrame({"bin": X_str, "target": y})

        # Calculate Good/Bad stats
        grouped = df.groupby("bin", observed=True)["target"].agg(["count", "sum"])
        grouped = grouped.rename(columns={"count": "total", "sum": "bad"})
        grouped["good"] = grouped["total"] - grouped["bad"]

        total_bad = grouped["bad"].sum()
        total_good = grouped["good"].sum()

        if total_bad == 0 or total_good == 0:
            # Degenerate case, set everything to 0
            self.iv_ = 0.0
            self.woe_map_ = {k: 0.0 for k in grouped.index}
            self.summary_ = grouped.copy()
            self.is_fitted_ = True
            return self

        # Distributions with smoothing
        dist_bad = (grouped["bad"] / total_bad).clip(lower=self.epsilon)
        dist_good = (grouped["good"] / total_good).clip(lower=self.epsilon)

        # WoE and IV
        woe = np.log(dist_good / dist_bad)
        iv_contrib = (dist_good - dist_bad) * woe

        # Store results
        self.woe_map_ = woe.to_dict()
        self.iv_ = float(iv_contrib.sum())

        # Create summary table
        self.summary_ = grouped.copy()
        self.summary_["dist_good"] = dist_good
        self.summary_["dist_bad"] = dist_bad
        self.summary_["woe"] = woe
        self.summary_["iv_contribution"] = iv_contrib

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """
        Transform X using the learned WoE mapping.

        Parameters
        ----------
        X : pd.Series
            Feature data to transform.

        Returns
        -------
        pd.Series
            Transformed data (WoE values).
        """
        if not self.is_fitted_:
            raise ValueError("Encoder is not fitted. Call fit() first.")

        X = X.copy()

        # Convert to string for consistent mapping
        X_str = X.astype(str)

        # Map values
        # Note: If X_str contains categories not in woe_map_, map returns NaN.
        # We fill with 0 (neutral WoE) as standard fallback.
        mapped = X_str.map(self.woe_map_)

        return mapped.fillna(0.0).astype(float)

    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def get_iv(self) -> float:
        """Get the Information Value."""
        if not self.is_fitted_:
            raise ValueError("Encoder is not fitted. Call fit() first.")
        return self.iv_

    def get_woe_map(self) -> Dict[Any, float]:
        """Get the WOE mapping dictionary."""
        if not self.is_fitted_:
            raise ValueError("Encoder is not fitted. Call fit() first.")
        return self.woe_map_.copy()

    def get_summary(self) -> pd.DataFrame:
        """Get the WOE summary table."""
        if not self.is_fitted_:
            raise ValueError("Encoder is not fitted. Call fit() first.")
        return self.summary_.copy()


# Backward compatibility functions (wrappers)
def calculate_woe_mapping(
    df: pd.DataFrame,
    target: str,
    feature: str,
    epsilon: float = 1e-8,
) -> Dict[Any, float]:
    """
    Wrapper using WOEEncoder for backward compatibility.

    Note: This function assumes the feature is already binned or categorical.
    """
    encoder = WOEEncoder(epsilon=epsilon)
    encoder.fit(df[feature], df[target])
    return encoder.woe_map_


def apply_woe_transform(
    df: pd.DataFrame,
    feature: str,
    woe_map: Dict[Any, float],
    new_col_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply WoE transform using a pre-computed WoE mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature : str
        Feature column name.
    woe_map : Dict[Any, float]
        WoE mapping dictionary.
    new_col_name : str, optional
        Name for the new WoE column. Default: "{feature}_woe".

    Returns
    -------
    pd.DataFrame
        DataFrame with new WoE column.
    """
    if new_col_name is None:
        new_col_name = f"{feature}_woe"

    df_out = df.copy()
    mapped = df_out[feature].astype(str).map(woe_map)
    df_out[new_col_name] = mapped.fillna(0.0).astype(float)
    return df_out
