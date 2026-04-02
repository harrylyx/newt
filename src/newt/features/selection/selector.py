"""
FeatureSelector module for variable analysis and selection.

Combines Exploratory Data Analysis (EDA) and variable filtering into a unified workflow.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from newt.config import BINNING, FILTERING
from newt.features.analysis.correlation import (
    calculate_correlation_matrix,
    get_high_correlation_pairs,
)
from newt.features.analysis.iv_calculator import calculate_iv
from newt.metrics.ks import calculate_ks
from newt.metrics.lift import calculate_lift_at_k
from newt.utils.decorators import requires_fit

# Default metrics that do not require a label
BASIC_METRICS = frozenset(
    [
        "mean",
        "median",
        "q1",
        "q3",
        "variance",
        "std",
        "skewness",
        "kurtosis",
        "mode_ratio",
        "nonzero_ratio",
        "nonnull_nonneg_ratio",
        "unique_count",
        "unique_ratio",
        "missing_rate",
        "min",
        "max",
    ]
)

# Metrics that require a label (y)
LABEL_METRICS = frozenset(
    [
        "correlation",
        "ks",
        "iv",
        "lift_10",
    ]
)

ALL_METRICS = BASIC_METRICS | LABEL_METRICS


class FeatureSelector:
    """
    Unified feature analysis and selection tool.

    Workflow:
    1. Initialize with desired metrics.
    2. Call `fit(X, y)` to calculate statistics (EDA).
    3. Call `report()` to view the analysis results.
    4. Call `select()` with thresholds to filter features.
    5. Call `transform(X)` to apply the selection.

    Notes
    -----
    - The 'correlation' metric calculates correlation between each
      feature (X) and target (y).
    - Feature-to-feature correlation matrix is computed in `fit()`
      and exposed via the `corr_matrix` property.
    - In `select()`, `corr_threshold` removes highly correlated
      feature pairs, keeping the one with higher IV.

    Examples
    --------
    >>> fs = FeatureSelector(metrics=['iv', 'missing_rate', 'ks'])
    >>> fs.fit(X, y)
    >>> print(fs.report())
    >>> print(fs.corr_matrix)  # View feature-to-feature correlations
    >>> fs.select(iv_threshold=0.02, missing_threshold=0.9, corr_threshold=0.8)
    >>> X_filtered = fs.transform(X)
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        iv_bins: int = BINNING.DEFAULT_BUCKETS,
        lift_k: float = 0.1,
        corr_method: str = "pearson",
    ):
        """
        Initialize FeatureSelector.

        Args:
            metrics: List of metrics to compute. If None, computes all
                available metrics.
            iv_bins: Number of bins for IV calculation.
            lift_k: Fraction for Lift@k calculation.
            corr_method: Correlation method for pairwise removal
                ('pearson', 'kendall', 'spearman').
        """
        self.iv_bins = iv_bins
        self.lift_k = lift_k
        self.corr_method = corr_method

        # Validate and set metrics
        if metrics is None:
            self.metrics: Set[str] = set(ALL_METRICS)
        else:
            invalid = set(metrics) - ALL_METRICS
            if invalid:
                raise ValueError(
                    f"Invalid metrics: {invalid}. "
                    f"Available metrics: {sorted(ALL_METRICS)}"
                )
            self.metrics = set(metrics)

        # Ensure mandatory metrics for filtering are present if they might be used
        # We don't force them here, but `select` will complain if they are missing

        # Analysis Results
        self.eda_summary_: pd.DataFrame = pd.DataFrame()

        # Selection Results
        self.selected_features_: List[str] = []
        self.removed_features_: Dict[str, str] = {}
        self.corr_removed_: List[Tuple[str, str, float]] = []
        self.is_fitted_: bool = False
        self.is_selected_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureSelector":
        """
        Calculate feature statistics (EDA).

        Args:
            X: Feature dataframe.
            y: Target variable (optional, but required for IV/KS/Correlation).

        Returns:
            self
        """
        self._analyze_dataframe(X, y)
        self.is_fitted_ = True

        # Keep all analyzed features selected until select() applies thresholds.
        self.selected_features_ = (
            list(self.eda_summary_["feature"]) if not self.eda_summary_.empty else []
        )
        self.removed_features_ = {}
        self.corr_removed_ = []
        self.is_selected_ = False

        return self

    def _validate_selection_metrics(self) -> None:
        """Ensure metrics required by select() are present."""
        available_cols = self.eda_summary_.columns
        if "missing_rate" not in available_cols:
            raise ValueError(
                "Metric 'missing_rate' was not calculated. "
                "Cannot filter by missing rate."
            )
        if "iv" not in available_cols:
            raise ValueError("Metric 'iv' was not calculated. Cannot filter by IV.")

    def _filter_candidates(
        self,
        df: pd.DataFrame,
        iv_threshold: float,
        missing_threshold: float,
    ) -> List[str]:
        """Filter features by missing rate and IV thresholds."""
        candidates = []
        for feature in df.index:
            missing_rate = df.loc[feature, "missing_rate"]
            iv = df.loc[feature, "iv"]

            if pd.isna(iv):
                self.removed_features_[feature] = "iv_nan"
                continue

            if missing_rate > missing_threshold:
                self.removed_features_[feature] = f"missing_rate={missing_rate:.3f}"
            elif iv < iv_threshold:
                self.removed_features_[feature] = f"iv={iv:.4f}"
            else:
                candidates.append(feature)

        return candidates

    def select(
        self,
        iv_threshold: float = FILTERING.DEFAULT_IV_THRESHOLD,
        missing_threshold: float = FILTERING.DEFAULT_MISSING_THRESHOLD,
        corr_threshold: float = FILTERING.DEFAULT_CORR_THRESHOLD,
    ) -> "FeatureSelector":
        """
        Apply filtering based on calculated statistics.

        Args:
            iv_threshold: Minimum IV to keep a feature.
            missing_threshold: Maximum missing rate to keep a feature.
            corr_threshold: Maximum absolute correlation allowed between features.
                When two features have correlation >= threshold, the one with
                lower IV is removed.

        Returns:
            self
        """
        if not self.is_fitted_:
            raise ValueError("FeatureSelector is not fitted. Call fit() first.")

        self._validate_selection_metrics()
        df = self.eda_summary_.set_index("feature")
        self.removed_features_ = {}
        candidates = self._filter_candidates(df, iv_threshold, missing_threshold)

        self.selected_features_ = candidates
        if len(candidates) > 1:
            self.selected_features_ = self._remove_correlated(
                candidates, corr_threshold
            )

        self.is_selected_ = True
        return self

    def _analyze_dataframe(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ):
        """Internal method to perform EDA."""
        results = []
        # Calculate correlation matrix for later use in select()
        # Only for numeric columns
        numeric_df = X.select_dtypes(include=[np.number])
        self.corr_matrix_ = calculate_correlation_matrix(
            numeric_df, method=self.corr_method
        )

        cols = X.columns
        for feat in cols:
            # Analyze single feature
            res = self._analyze_single(X[feat], y, feature_name=feat)
            results.append(res)

        self.eda_summary_ = pd.DataFrame(results)
        if not self.eda_summary_.empty:
            # Reorder columns
            cols_order = ["feature", "dtype", "count"]
            other_cols = [c for c in self.eda_summary_.columns if c not in cols_order]
            self.eda_summary_ = self.eda_summary_[cols_order + sorted(other_cols)]

    def _analyze_single(
        self,
        X: pd.Series,
        y: Optional[pd.Series] = None,
        feature_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze a single feature."""
        if not isinstance(X, pd.Series):
            X = pd.Series(X)

        name = feature_name or X.name or "feature"
        result = {"feature": name}

        n_total = len(X)
        is_numeric = pd.api.types.is_numeric_dtype(X)
        result["dtype"] = "numeric" if is_numeric else "categorical"
        result["count"] = n_total

        n_missing = X.isna().sum()
        missing_rate = n_missing / n_total if n_total > 0 else 0.0
        if "missing_rate" in self.metrics:
            result["missing_rate"] = missing_rate

        X_valid = X.dropna()
        n_valid = len(X_valid)

        # Numeric Stats
        if is_numeric and n_valid > 0:
            X_num = X_valid.astype(float)
            if "min" in self.metrics:
                result["min"] = float(X_num.min())
            if "max" in self.metrics:
                result["max"] = float(X_num.max())
            if "mean" in self.metrics:
                result["mean"] = float(X_num.mean())
            if "median" in self.metrics:
                result["median"] = float(X_num.median())
            if "std" in self.metrics:
                result["std"] = float(X_num.std())
            if "skewness" in self.metrics:
                result["skewness"] = float(stats.skew(X_num, nan_policy="omit"))
            if "kurtosis" in self.metrics:
                result["kurtosis"] = float(stats.kurtosis(X_num, nan_policy="omit"))
            # ... and so on for others if needed

        # Label metrics
        if y is not None:
            mask = ~(X.isna() | y.isna())
            X_a = X[mask]
            y_a = y[mask]
            if len(X_a) > 0:
                if "ks" in self.metrics and is_numeric:
                    try:
                        result["ks"] = float(
                            calculate_ks(y_a.values, X_a.astype(float).values)
                        )
                    except Exception:
                        result["ks"] = np.nan

                if "iv" in self.metrics:
                    try:
                        iv_res = calculate_iv(
                            pd.DataFrame({"f": X_a, "t": y_a}),
                            "t",
                            "f",
                            buckets=self.iv_bins,
                        )
                        result["iv"] = float(iv_res["iv"])
                    except Exception:
                        result["iv"] = np.nan

                if "correlation" in self.metrics and is_numeric:
                    try:
                        result["correlation"] = float(
                            X_a.astype(float).corr(y_a.astype(float))
                        )
                    except Exception:
                        result["correlation"] = np.nan

                if "lift_10" in self.metrics and is_numeric:
                    try:
                        result["lift_10"] = float(
                            calculate_lift_at_k(
                                y_a.values,
                                X_a.astype(float).values,
                                k=self.lift_k,
                            )
                        )
                    except Exception:
                        result["lift_10"] = np.nan

        else:
            for m in LABEL_METRICS & self.metrics:
                result[m] = np.nan

        return result

    def _remove_correlated(self, candidates: List[str], threshold: float) -> List[str]:
        """Remove correlated features using the stored correlation matrix."""
        valid_candidates = [c for c in candidates if c in self.corr_matrix_.columns]
        final_selection = [c for c in candidates if c not in self.corr_matrix_.columns]

        sub_matrix = self.corr_matrix_.loc[valid_candidates, valid_candidates]
        high_corr_pairs = get_high_correlation_pairs(sub_matrix, threshold)

        to_remove: Set[str] = set()
        corr_reasons: Dict[str, List[Tuple[str, float]]] = {}
        iv_map = self.eda_summary_.set_index("feature")["iv"].to_dict()

        for pair in high_corr_pairs:
            var1 = pair["var1"]
            var2 = pair["var2"]
            corr = pair["correlation"]

            if var1 in to_remove or var2 in to_remove:
                if var1 in to_remove:
                    corr_reasons.setdefault(var1, []).append((var2, corr))
                if var2 in to_remove:
                    corr_reasons.setdefault(var2, []).append((var1, corr))
                continue

            iv1 = iv_map.get(var1, 0)
            iv2 = iv_map.get(var2, 0)

            if iv1 >= iv2:
                to_remove.add(var2)
                self.corr_removed_.append((var2, var1, corr))
                corr_reasons.setdefault(var2, []).append((var1, corr))
            else:
                to_remove.add(var1)
                self.corr_removed_.append((var1, var2, corr))
                corr_reasons.setdefault(var1, []).append((var2, corr))

        for removed_var, related_vars in corr_reasons.items():
            related_vars.sort(key=lambda x: abs(x[1]), reverse=True)
            reason_parts = [f"{v}({c:.3f})" for v, c in related_vars]
            self.removed_features_[removed_var] = (
                f"high_corr: {', '.join(reason_parts)}"
            )

        final_selection.extend([c for c in valid_candidates if c not in to_remove])
        return final_selection

    @requires_fit()
    def report(self) -> pd.DataFrame:
        """
        Generate a report combining EDA stats and selection status.
        """
        if self.eda_summary_.empty:
            return pd.DataFrame()

        df = self.eda_summary_.copy()

        # Add status columns
        df["status"] = df["feature"].apply(
            lambda x: "selected" if x in self.selected_features_ else "removed"
        )
        df["reason"] = df["feature"].apply(lambda x: self.removed_features_.get(x, ""))

        return df

    @requires_fit()
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X with only selected features."""
        available = [c for c in self.selected_features_ if c in X.columns]
        return X[available]

    @property
    @requires_fit()
    def corr_matrix(self) -> pd.DataFrame:
        """
        Get the feature-to-feature correlation matrix.

        This matrix is computed during `fit()` for all numeric features.
        It shows pairwise correlations between features (X columns),
        NOT the correlation between features and target (y).

        Returns:
            pd.DataFrame: Correlation matrix of shape (n_features, n_features).
        """
        return self.corr_matrix_.copy()
