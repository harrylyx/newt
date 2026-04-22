"""Feature analysis helpers used by the selector facade."""

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy import stats

from newt._native import require_native_module
from newt.config import BINNING
from newt.features.analysis.batch_iv import calculate_batch_iv
from newt.features.analysis.correlation import calculate_correlation_matrix
from newt.metrics.binary_metrics import calculate_binary_metrics_batch
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
VALID_ENGINES = frozenset(["auto", "rust", "python"])


class FeatureAnalyzer:
    """Calculate feature-level analysis metrics without applying thresholds."""

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        iv_bins: int = BINNING.DEFAULT_BUCKETS,
        lift_k: float = 0.1,
        corr_method: str = "pearson",
        engine: str = "auto",
    ):
        if engine not in VALID_ENGINES:
            raise ValueError(
                f"engine must be one of {sorted(VALID_ENGINES)}, got: {engine}"
            )

        self.iv_bins = iv_bins
        self.lift_k = lift_k
        self.corr_method = corr_method
        self.engine = engine

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
        if self.engine == "rust":
            self._validate_rust_requirements()

        numeric_df = X.select_dtypes(include=[np.number])
        corr_matrix = calculate_correlation_matrix(
            numeric_df,
            method=self.corr_method,
            engine=self.engine,
        )

        results = [
            self._analyze_single(X[column], y=None, feature_name=column)
            for column in X.columns
        ]
        summary = pd.DataFrame(results)

        if y is not None and not summary.empty:
            supervised = self._analyze_supervised_batch(X, y)
            for metric in LABEL_METRICS & self.metrics:
                metric_map = supervised.get(metric, {})
                summary[metric] = summary["feature"].map(metric_map).astype(float)

        if not summary.empty:
            first_columns = ["feature", "dtype", "count"]
            other_columns = [c for c in summary.columns if c not in first_columns]
            summary = summary[first_columns + sorted(other_columns)]

        return FeatureAnalysisResult(
            summary=summary,
            corr_matrix=corr_matrix,
            metrics=frozenset(self.metrics),
        )

    def _analyze_supervised_batch(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate supervised metrics (IV/KS/Lift/target corr) in batch."""
        features = X.columns.tolist()
        metric_maps: Dict[str, Dict[str, float]] = {
            metric: {feature: np.nan for feature in features}
            for metric in (LABEL_METRICS & self.metrics)
        }

        if isinstance(y, pd.Series):
            y_series = y.reindex(X.index)
        else:
            y_series = pd.Series(np.asarray(y).ravel(), index=X.index)
        y_numeric = pd.to_numeric(y_series, errors="coerce")
        valid_binary = y_numeric.isin([0, 1])

        if "iv" in self.metrics:
            iv_df = calculate_batch_iv(
                X,
                y_numeric,
                features=features,
                bins=self.iv_bins,
                engine=self.engine,
            )
            iv_map = dict(zip(iv_df["feature"], iv_df["iv"]))
            metric_maps["iv"].update({k: float(v) for k, v in iv_map.items()})

        if "correlation" in self.metrics:
            numeric_features = [
                f for f in features if pd.api.types.is_numeric_dtype(X[f])
            ]
            if numeric_features:
                numeric_data = X[numeric_features].apply(pd.to_numeric, errors="coerce")
                corr_series = numeric_data.corrwith(y_numeric, axis=0)
                metric_maps["correlation"].update(
                    {
                        feature: (
                            float(corr_series[feature])
                            if pd.notna(corr_series[feature])
                            else np.nan
                        )
                        for feature in corr_series.index
                    }
                )

        if "ks" in self.metrics or "lift_10" in self.metrics:
            numeric_features = [
                f for f in features if pd.api.types.is_numeric_dtype(X[f])
            ]
            group_features: List[str] = []
            groups = []

            for feature in numeric_features:
                score = pd.to_numeric(X[feature], errors="coerce")
                mask = valid_binary & score.notna()
                if not mask.any():
                    continue
                group_features.append(feature)
                groups.append(
                    (
                        y_numeric.loc[mask].astype(float).to_numpy(),
                        score.loc[mask].astype(float).to_numpy(),
                    )
                )

            if groups:
                rows = calculate_binary_metrics_batch(
                    groups=groups,
                    lift_use_descending_score=True,
                    reverse_auc_label=False,
                    metrics_mode="exact",
                    lift_levels=(self.lift_k,),
                    engine=self.engine,
                )
                lift_key = f"{int(self.lift_k * 100)}%lift"

                for feature, row in zip(group_features, rows):
                    if "ks" in self.metrics:
                        metric_maps["ks"][feature] = float(row.get("KS", np.nan))
                    if "lift_10" in self.metrics:
                        metric_maps["lift_10"][feature] = float(
                            row.get(lift_key, np.nan)
                        )

        return metric_maps

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

        return result

    def _validate_rust_requirements(self) -> None:
        """Validate required Rust functions before strict Rust execution."""
        module = require_native_module()
        required = [
            "calculate_correlation_matrix_numpy",
            "extract_high_correlation_pairs_numpy",
        ]
        if "iv" in self.metrics:
            required.extend(
                ["calculate_batch_iv_numpy", "calculate_batch_categorical_iv"]
            )
        if "ks" in self.metrics or "lift_10" in self.metrics:
            required.append("calculate_binary_metrics_batch_numpy")

        missing = [
            name for name in required if not callable(getattr(module, name, None))
        ]
        if missing:
            raise RuntimeError(
                "Rust engine requires unavailable native functions: "
                + ", ".join(sorted(set(missing)))
            )
