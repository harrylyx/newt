"""Feature analysis helpers used by the selector facade."""

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy import stats

from newt.config import BINNING
from newt.features.analysis.correlation import calculate_correlation_matrix
from newt.features.analysis.iv_calculator import calculate_iv
from newt.metrics.ks import calculate_ks
from newt.metrics.lift import calculate_lift_at_k
from newt.results import FeatureAnalysisResult

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

LABEL_METRICS = frozenset(
    [
        "correlation",
        "ks",
        "iv",
        "lift_10",
    ]
)

ALL_METRICS = BASIC_METRICS | LABEL_METRICS


class FeatureAnalyzer:
    """Calculate feature-level analysis metrics without applying thresholds."""

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        iv_bins: int = BINNING.DEFAULT_BUCKETS,
        lift_k: float = 0.1,
        corr_method: str = "pearson",
    ):
        self.iv_bins = iv_bins
        self.lift_k = lift_k
        self.corr_method = corr_method

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

    def analyze(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> FeatureAnalysisResult:
        """Analyze a dataframe and return stable result objects."""
        numeric_df = X.select_dtypes(include=[np.number])
        corr_matrix = calculate_correlation_matrix(
            numeric_df,
            method=self.corr_method,
        )

        results = [
            self._analyze_single(X[column], y, feature_name=column)
            for column in X.columns
        ]
        summary = pd.DataFrame(results)
        if not summary.empty:
            first_columns = ["feature", "dtype", "count"]
            other_columns = [c for c in summary.columns if c not in first_columns]
            summary = summary[first_columns + sorted(other_columns)]

        return FeatureAnalysisResult(
            summary=summary,
            corr_matrix=corr_matrix,
            metrics=frozenset(self.metrics),
        )

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

        if y is None:
            for metric in LABEL_METRICS & self.metrics:
                result[metric] = np.nan
            return result

        mask = ~(X.isna() | y.isna())
        X_aligned = X[mask]
        y_aligned = y[mask]

        if len(X_aligned) == 0:
            for metric in LABEL_METRICS & self.metrics:
                result[metric] = np.nan
            return result

        if "ks" in self.metrics and is_numeric:
            try:
                result["ks"] = float(
                    calculate_ks(y_aligned.values, X_aligned.astype(float).values)
                )
            except Exception:
                result["ks"] = np.nan

        if "iv" in self.metrics:
            try:
                iv_result = calculate_iv(
                    pd.DataFrame({"feature": X_aligned, "target": y_aligned}),
                    "target",
                    "feature",
                    buckets=self.iv_bins,
                    engine="rust",
                )
                result["iv"] = float(iv_result["iv"])
            except Exception:
                result["iv"] = np.nan

        if "correlation" in self.metrics and is_numeric:
            try:
                result["correlation"] = float(
                    X_aligned.astype(float).corr(y_aligned.astype(float))
                )
            except Exception:
                result["correlation"] = np.nan

        if "lift_10" in self.metrics and is_numeric:
            try:
                result["lift_10"] = float(
                    calculate_lift_at_k(
                        y_aligned.values,
                        X_aligned.astype(float).values,
                        k=self.lift_k,
                    )
                )
            except Exception:
                result["lift_10"] = np.nan

        return result
