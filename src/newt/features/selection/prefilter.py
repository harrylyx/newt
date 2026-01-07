"""
PreFilter module for variable pre-filtering.

Provides filtering based on IV, missing rate, and correlation.
"""

from typing import Dict, List, Tuple


import numpy as np
import pandas as pd

from newt.config import BINNING, FILTERING
from newt.features.analysis.correlation import (
    calculate_correlation_matrix,
    get_high_correlation_pairs,
)
from newt.features.binning import Binner
from newt.utils.decorators import requires_fit


class PreFilter:
    """
    Variable pre-filter based on IV, missing rate, and correlation.

    This filter performs initial variable screening before binning, using:
    - Information Value (IV) threshold
    - Missing rate threshold
    - Correlation threshold (removes highly correlated variables)

    Examples
    --------
    >>> prefilter = PreFilter(iv_threshold=0.02, missing_threshold=0.9)
    >>> X_filtered = prefilter.fit_transform(X, y)
    >>> print(prefilter.report())
    """

    def __init__(
        self,
        iv_threshold: float = FILTERING.DEFAULT_IV_THRESHOLD,
        missing_threshold: float = FILTERING.DEFAULT_MISSING_THRESHOLD,
        corr_threshold: float = FILTERING.DEFAULT_CORR_THRESHOLD,
        iv_bins: int = BINNING.DEFAULT_BUCKETS,
        corr_method: str = "pearson",
    ):
        """
        Initialize PreFilter.

        Parameters
        ----------
        iv_threshold : float
            Minimum IV value to keep a variable. Default 0.02.
        missing_threshold : float
            Maximum missing rate to keep a variable. Default 0.9 (90%).
        corr_threshold : float
            Maximum correlation allowed between variables. Default 0.8.
        iv_bins : int
            Number of bins for IV calculation. Default 10.
        corr_method : str
            Correlation method ('pearson', 'kendall', 'spearman'). Default 'pearson'.
        """
        self.iv_threshold = iv_threshold
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.iv_bins = iv_bins
        self.corr_method = corr_method

        # Results
        self.selected_features_: List[str] = []
        self.removed_features_: Dict[str, str] = {}  # feature -> reason
        self.iv_dict_: Dict[str, float] = {}
        self.missing_dict_: Dict[str, float] = {}
        self.corr_removed_: List[Tuple[str, str, float]] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PreFilter":
        """
        Fit the pre-filter to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.
        y : pd.Series
            Target variable (binary 0/1).

        Returns
        -------
        PreFilter
            Fitted instance.
        """
        X = X.copy()
        y = y.copy()

        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Reset results
        self.iv_dict_ = {}
        self.missing_dict_ = {}
        self.removed_features_ = {}
        self.corr_removed_ = []
        candidates = []

        # Step 1: Calculate missing rate and filter
        for col in numeric_cols:
            missing_rate = X[col].isna().mean()
            self.missing_dict_[col] = missing_rate

            if missing_rate > self.missing_threshold:
                self.removed_features_[col] = f"missing_rate={missing_rate:.3f}"
            else:
                candidates.append(col)

        # Step 2: Calculate IV using equal-frequency binning and filter
        iv_candidates = []
        for col in candidates:
            try:
                iv = self._calculate_iv(X[col], y)
                self.iv_dict_[col] = iv

                if iv < self.iv_threshold:
                    self.removed_features_[col] = f"iv={iv:.4f}"
                else:
                    iv_candidates.append(col)
            except Exception as e:
                # If IV calculation fails, exclude the variable
                self.removed_features_[col] = f"iv_error={str(e)}"

        # Step 3: Remove highly correlated variables
        if len(iv_candidates) > 1:
            self.selected_features_ = self._remove_correlated(
                X[iv_candidates], iv_candidates
            )
        else:
            self.selected_features_ = iv_candidates

        self.is_fitted_ = True
        return self

    @requires_fit()
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filter columns based on fitted selection.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Filtered data with only selected features.
        """
        # Keep only selected features that exist in X
        cols_to_keep = [c for c in self.selected_features_ if c in X.columns]
        return X[cols_to_keep]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    @requires_fit()
    def report(self) -> pd.DataFrame:
        """
        Generate a filtering report.

        Returns
        -------
        pd.DataFrame
            Report with columns: feature, missing_rate, iv, status, reason
        """
        all_features = list(self.missing_dict_.keys())
        records = []

        for feat in all_features:
            missing_rate = self.missing_dict_.get(feat, np.nan)
            iv = self.iv_dict_.get(feat, np.nan)

            if feat in self.selected_features_:
                status = "selected"
                reason = ""
            else:
                status = "removed"
                reason = self.removed_features_.get(feat, "corr_filter")

            records.append(
                {
                    "feature": feat,
                    "missing_rate": missing_rate,
                    "iv": iv,
                    "status": status,
                    "reason": reason,
                }
            )

        df = pd.DataFrame(records)
        # Sort by IV descending for selected, then by name for removed
        df = df.sort_values(["status", "iv"], ascending=[True, False]).reset_index(
            drop=True
        )
        return df

    def _calculate_iv(self, X: pd.Series, y: pd.Series) -> float:
        """
        Calculate IV using equal-frequency binning.

        Parameters
        ----------
        X : pd.Series
            Feature data.
        y : pd.Series
            Target variable.

        Returns
        -------
        float
            Information Value.
        """
        # Use equal-frequency binning
        binner = Binner()
        binner.fit(
            pd.DataFrame({X.name or "x": X}),
            y,
            method="quantile",
            n_bins=self.iv_bins,
        )

        # Get binned data
        X_binned = binner.transform(pd.DataFrame({X.name or "x": X}))
        col_name = X.name or "x"
        binned_series = X_binned[col_name]

        # Calculate IV
        df = pd.DataFrame({"bin": binned_series, "target": y})
        grouped = df.groupby("bin", observed=True)["target"].agg(["count", "sum"])
        grouped.columns = ["total", "bad"]
        grouped["good"] = grouped["total"] - grouped["bad"]

        total_bad = grouped["bad"].sum()
        total_good = grouped["good"].sum()

        if total_bad == 0 or total_good == 0:
            return 0.0

        epsilon = 1e-8
        dist_bad = (grouped["bad"] / total_bad).clip(lower=epsilon)
        dist_good = (grouped["good"] / total_good).clip(lower=epsilon)

        woe = np.log(dist_good / dist_bad)
        iv_contrib = (dist_good - dist_bad) * woe

        return float(iv_contrib.sum())

    def _remove_correlated(self, X: pd.DataFrame, candidates: List[str]) -> List[str]:
        """
        Remove highly correlated variables, keeping the one with higher IV.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.
        candidates : List[str]
            List of candidate features.

        Returns
        -------
        List[str]
            Features after removing highly correlated ones.
        """
        corr_matrix = calculate_correlation_matrix(X, method=self.corr_method)
        high_corr_pairs = get_high_correlation_pairs(corr_matrix, self.corr_threshold)

        # Track features to remove
        to_remove = set()

        for pair in high_corr_pairs:
            var1 = pair["var1"]
            var2 = pair["var2"]
            corr = pair["correlation"]

            # Skip if either already removed
            if var1 in to_remove or var2 in to_remove:
                continue

            # Keep the one with higher IV
            iv1 = self.iv_dict_.get(var1, 0)
            iv2 = self.iv_dict_.get(var2, 0)

            if iv1 >= iv2:
                to_remove.add(var2)
                self.corr_removed_.append((var2, var1, corr))
                self.removed_features_[var2] = f"corr_with_{var1}={corr:.3f}"
            else:
                to_remove.add(var1)
                self.corr_removed_.append((var1, var2, corr))
                self.removed_features_[var1] = f"corr_with_{var2}={corr:.3f}"

        # Return remaining candidates
        return [c for c in candidates if c not in to_remove]
