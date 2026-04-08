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
    """Weight of Evidence (WOE) Encoder and Information Value (IV) calculator.

    The WOEEncoder transforms discrete bins or categorical features into numerical
    Weight of Evidence values, which represent the log-odds of the event rate
    in each bin relative to the overall population. It also calculates the
    Information Value (IV) to measure feature predictive power.

    Attributes:
        woe_map_ (Dict[Any, float]): Mapping of bin labels to WOE values.
        iv_ (float): Total Information Value for the feature.
        summary_ (pd.DataFrame): Detailed table showing bin counts, event rates,
            WOE, and IV contribution.

    Examples:
        >>> from newt.features.analysis import WOEEncoder
        >>> encoder = WOEEncoder()
        >>> # Fit on categorical column 'city' with target 'default'
        >>> encoder.fit(df['city'], df['default'])
        >>> print(f"IV: {encoder.iv_:.4f}")
        >>> # Transform to numerical values
        >>> df['city_woe'] = encoder.transform(df['city'])
    """

    def __init__(self, epsilon: float = BINNING.DEFAULT_EPSILON):
        """Initialize the WOEEncoder.

        Args:
            epsilon: Small constant for smoothing (to avoid log(0)).
                Maintained for API compatibility; internal math follows
                distribution-aware smoothing.
        """
        # Kept for API compatibility; IV smoothing now follows
        # toad-compatible semantics.
        self.epsilon = epsilon
        self.woe_map_: Dict[Any, float] = {}
        self.iv_: float = 0.0
        self.summary_: pd.DataFrame = pd.DataFrame()
        self.is_fitted_: bool = False

    def fit(self, X: pd.Series, y: pd.Series) -> "WOEEncoder":
        """Fit the WOE encoder by calculating distributions and log-odds.

        Args:
            X: Input feature series (categorical or binned).
            y: Binary target series (0/1).

        Returns:
            WOEEncoder: The fitted encoder instance.
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
        """Apply learned WOE transformations to new data.

        Args:
            X: Input feature series.

        Returns:
            pd.Series: Transformed series with float WOE values.
        """
        X_copy = X.copy()

        # Convert to string for consistent mapping
        X_str = X_copy.astype(str)

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
