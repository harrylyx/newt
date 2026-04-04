"""
WOE (Weight of Evidence) calculation and encoding.

Provides WOE transformation for binned/categorical features.
"""

from typing import Any, Dict

import pandas as pd

from newt.config import BINNING
from newt.utils.decorators import requires_fit

from .iv_math import build_iv_summary


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

    def __init__(self, epsilon: float = BINNING.DEFAULT_EPSILON):
        """
        Initialize WOEEncoder.

        Parameters
        ----------
        epsilon : float
            Retained for backward compatibility. IV smoothing follows
            toad-compatible defaults.
        """
        # Kept for API compatibility; IV smoothing now follows toad semantics.
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
        summary, iv_value = build_iv_summary(X.copy(), y.copy())
        self.summary_ = summary
        self.iv_ = float(iv_value)
        if "woe" in summary.columns:
            self.woe_map_ = summary["woe"].to_dict()
        else:
            self.woe_map_ = {}

        self.is_fitted_ = True
        return self

    @requires_fit()
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

    @requires_fit()
    def get_iv(self) -> float:
        """Get the Information Value."""
        return self.iv_

    @requires_fit()
    def get_woe_map(self) -> Dict[Any, float]:
        """Get the WOE mapping dictionary."""
        return self.woe_map_.copy()

    @requires_fit()
    def get_summary(self) -> pd.DataFrame:
        """Get the WOE summary table."""
        return self.summary_.copy()
