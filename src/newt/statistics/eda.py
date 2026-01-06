"""
Exploratory Data Analysis (EDA) module for feature analysis.

Provides the EDAAnalyzer class for computing descriptive statistics
and model-related metrics for features.
"""

from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from scipy import stats

from newt.features.analysis.iv_calculator import calculate_iv
from newt.features.analysis.woe_calculator import WOEEncoder

from newt.metrics.ks import calculate_ks
from newt.metrics.lift import calculate_lift_at_k

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


class EDAAnalyzer:
    """
    Exploratory Data Analysis (EDA) analyzer for features.

    Computes descriptive statistics and model-related metrics for numerical
    and categorical features. Supports flexible metric selection and batch
    processing of multiple features.

    Args:
        metrics: List of metrics to compute. If None, all metrics are computed.
                 Available metrics: mean, median, q1, q3, variance, std, skewness,
                 kurtosis, mode_ratio, nonzero_ratio, unique_count, unique_ratio,
                 missing_rate, min, max, correlation, ks, iv, lift_10.
        iv_bins: Number of bins for IV calculation (equal-frequency binning).
        lift_k: Fraction for Lift@k calculation (default 0.1 for 10% lift).

    Examples:
        >>> analyzer = EDAAnalyzer()
        >>> result = analyzer.analyze(df["feature"], df["target"])
        >>> print(result)

        >>> # Batch analysis
        >>> summary_df = analyzer.analyze_dataframe(df, target="label")
        >>> print(summary_df)
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        iv_bins: int = 10,
        lift_k: float = 0.1,
    ):
        self.iv_bins = iv_bins
        self.lift_k = lift_k
        self._last_results: Dict[str, Dict[str, Any]] = {}

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

    def analyze(
        self,
        X: Union[pd.Series, np.ndarray, list],
        y: Optional[Union[pd.Series, np.ndarray, list]] = None,
        feature_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a single feature.

        Args:
            X: Feature data (1D array-like).
            y: Optional target labels (binary 0/1) for label-dependent metrics.
            feature_name: Optional name for the feature.

        Returns:
            Dictionary with computed metric values.
        """
        # Convert to pandas Series
        if not isinstance(X, pd.Series):
            X = pd.Series(X)

        if y is not None and not isinstance(y, pd.Series):
            y = pd.Series(y)

        result: Dict[str, Any] = {}

        # Feature name
        name = feature_name or (X.name if X.name else "feature")
        result["feature"] = name

        # Data type inference
        is_numeric = pd.api.types.is_numeric_dtype(X)
        result["dtype"] = "numeric" if is_numeric else "categorical"

        # Total count
        n_total = len(X)
        result["count"] = n_total

        # Missing rate
        n_missing = X.isna().sum()
        missing_rate = n_missing / n_total if n_total > 0 else 0.0
        if "missing_rate" in self.metrics:
            result["missing_rate"] = missing_rate

        # Non-missing data for further analysis
        X_valid = X.dropna()
        n_valid = len(X_valid)

        # Basic statistics for numeric data
        if is_numeric and n_valid > 0:
            X_numeric = X_valid.astype(float)

            if "min" in self.metrics:
                result["min"] = float(X_numeric.min())
            if "max" in self.metrics:
                result["max"] = float(X_numeric.max())
            if "mean" in self.metrics:
                result["mean"] = float(X_numeric.mean())
            if "median" in self.metrics:
                result["median"] = float(X_numeric.median())
            if "q1" in self.metrics:
                result["q1"] = float(X_numeric.quantile(0.25))
            if "q3" in self.metrics:
                result["q3"] = float(X_numeric.quantile(0.75))
            if "variance" in self.metrics:
                result["variance"] = float(X_numeric.var())
            if "std" in self.metrics:
                result["std"] = float(X_numeric.std())
            if "skewness" in self.metrics:
                result["skewness"] = float(stats.skew(X_numeric, nan_policy="omit"))
            if "kurtosis" in self.metrics:
                result["kurtosis"] = float(stats.kurtosis(X_numeric, nan_policy="omit"))
            if "nonzero_ratio" in self.metrics:
                result["nonzero_ratio"] = float((X_numeric != 0).sum() / n_valid)
            if "nonnull_nonneg_ratio" in self.metrics:
                result["nonnull_nonneg_ratio"] = (
                    float((X_numeric >= 0).sum() / n_total) if n_total > 0 else 0.0
                )

            # Outlier detection (IQR method)
            q1 = X_numeric.quantile(0.25)
            q3 = X_numeric.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            n_outliers = ((X_numeric < lower_bound) | (X_numeric > upper_bound)).sum()
            result["outlier_count"] = int(n_outliers)
            result["outlier_ratio"] = (
                float(n_outliers / n_valid) if n_valid > 0 else 0.0
            )

        elif not is_numeric and n_valid > 0:
            # Categorical data - set numeric metrics to NaN
            for metric in [
                "min",
                "max",
                "mean",
                "median",
                "q1",
                "q3",
                "variance",
                "std",
                "skewness",
                "kurtosis",
            ]:
                if metric in self.metrics:
                    result[metric] = np.nan

            if "nonzero_ratio" in self.metrics:
                # For categorical, compute non-empty/non-null ratio (already handled)
                result["nonzero_ratio"] = 1.0  # All valid values are "non-zero"

        # Unique values
        if "unique_count" in self.metrics:
            result["unique_count"] = int(X_valid.nunique())
        if "unique_ratio" in self.metrics:
            result["unique_ratio"] = (
                float(X_valid.nunique() / n_valid) if n_valid > 0 else 0.0
            )

        # Mode ratio
        if "mode_ratio" in self.metrics and n_valid > 0:
            mode_count = (
                X_valid.value_counts().iloc[0] if len(X_valid.value_counts()) > 0 else 0
            )
            result["mode_ratio"] = float(mode_count / n_valid)
            result["mode_value"] = (
                X_valid.value_counts().index[0]
                if len(X_valid.value_counts()) > 0
                else None
            )

        # Label-dependent metrics
        if y is not None:
            # Align X and y, drop rows where either is NaN
            mask = ~(X.isna() | y.isna())
            X_aligned = X[mask]
            y_aligned = y[mask]
            n_aligned = len(X_aligned)

            if n_aligned > 0:
                # Correlation (for numeric features only)
                if "correlation" in self.metrics:
                    if is_numeric:
                        try:
                            corr = X_aligned.astype(float).corr(y_aligned.astype(float))
                            result["correlation"] = (
                                float(corr) if not np.isnan(corr) else 0.0
                            )
                        except Exception:
                            result["correlation"] = np.nan
                    else:
                        result["correlation"] = np.nan

                # KS statistic
                if "ks" in self.metrics:
                    if is_numeric:
                        try:
                            ks = calculate_ks(
                                y_aligned.values, X_aligned.astype(float).values
                            )
                            result["ks"] = float(ks)
                        except Exception:
                            result["ks"] = np.nan
                    else:
                        result["ks"] = np.nan

                # IV (Information Value)
                if "iv" in self.metrics:
                    try:
                        iv_res = calculate_iv(
                            pd.DataFrame({"feature": X_aligned, "target": y_aligned}),
                            target="target",
                            feature="feature",
                            buckets=self.iv_bins,
                        )
                        result["iv"] = float(iv_res["iv"])
                    except Exception:
                        result["iv"] = np.nan


                # Lift@k
                if "lift_10" in self.metrics:
                    if is_numeric:
                        try:
                            lift = calculate_lift_at_k(
                                y_aligned.values,
                                X_aligned.astype(float).values,
                                k=self.lift_k,
                            )
                            result["lift_10"] = float(lift)
                        except Exception:
                            result["lift_10"] = np.nan
                    else:
                        result["lift_10"] = np.nan
            else:
                # No valid aligned data
                for metric in LABEL_METRICS:
                    if metric in self.metrics:
                        result[metric] = np.nan
        else:
            # No label provided, skip label-dependent metrics if requested
            for metric in LABEL_METRICS & self.metrics:
                result[metric] = np.nan

        return result

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Analyze multiple features in a DataFrame.

        Args:
            df: Input DataFrame.
            target: Optional name of target column (binary 0/1).
            features: List of feature columns to analyze. If None, all columns
                     except the target are analyzed.

        Returns:
            DataFrame with one row per feature and computed metrics as columns.
        """
        if features is None:
            features = [c for c in df.columns if c != target]

        y = df[target] if target is not None else None

        results = []
        for feat in features:
            if feat not in df.columns:
                continue
            result = self.analyze(df[feat], y=y, feature_name=feat)
            results.append(result)

        self._last_results = {r["feature"]: r for r in results}

        summary_df = pd.DataFrame(results)
        # Reorder columns: feature first, then sorted metrics
        if len(summary_df) > 0:
            cols = ["feature", "dtype", "count"]
            other_cols = [c for c in summary_df.columns if c not in cols]
            summary_df = summary_df[cols + sorted(other_cols)]

        return summary_df

    def summary(self) -> pd.DataFrame:
        """
        Get the summary of the last analyze_dataframe call.

        Returns:
            DataFrame with analysis results.
        """
        if not self._last_results:
            return pd.DataFrame()
        return pd.DataFrame(list(self._last_results.values()))

    @staticmethod
    def available_metrics() -> List[str]:
        """Return list of all available metrics."""
        return sorted(ALL_METRICS)

    @staticmethod
    def basic_metrics() -> List[str]:
        """Return list of metrics that do not require a label."""
        return sorted(BASIC_METRICS)

    @staticmethod
    def label_metrics() -> List[str]:
        """Return list of metrics that require a label."""
        return sorted(LABEL_METRICS)

    @staticmethod
    def describe_metrics() -> pd.DataFrame:
        """
        Return a DataFrame describing all available metrics with bilingual descriptions.

        Returns:
            DataFrame with columns: metric, english, chinese, requires_label
        """
        descriptions = [
            # Basic statistics
            ("mean", "Mean", "均值", False),
            ("median", "Median", "中位数", False),
            ("q1", "First Quartile (25%)", "第一四分位数", False),
            ("q3", "Third Quartile (75%)", "第三四分位数", False),
            ("variance", "Variance", "方差", False),
            ("std", "Standard Deviation", "标准差", False),
            ("skewness", "Skewness", "偏度", False),
            ("kurtosis", "Kurtosis", "峰度", False),
            ("min", "Minimum", "最小值", False),
            ("max", "Maximum", "最大值", False),
            # Distribution metrics
            ("mode_ratio", "Mode Ratio", "众数占比", False),
            ("nonzero_ratio", "Non-zero Ratio", "非零值占比", False),
            (
                "nonnull_nonneg_ratio",
                "Non-null Non-negative Ratio",
                "非空非负占比",
                False,
            ),
            ("unique_count", "Unique Value Count", "唯一值数量", False),
            ("unique_ratio", "Unique Value Ratio", "唯一值占比", False),
            ("missing_rate", "Missing Rate", "缺失率", False),
            # Label-dependent metrics
            (
                "correlation",
                "Pearson Correlation with Label",
                "与标签的Pearson相关系数",
                True,
            ),
            ("ks", "Kolmogorov-Smirnov Statistic", "KS统计量", True),
            (
                "iv",
                "Information Value (equal-freq binning)",
                "信息值（等频分箱）",
                True,
            ),
            ("lift_10", "Lift at Top 10%", "前10%提升度", True),
        ]

        return pd.DataFrame(
            descriptions, columns=["metric", "english", "chinese", "requires_label"]
        )
