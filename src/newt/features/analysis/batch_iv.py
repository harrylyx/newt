"""Batch IV calculation helpers with Rust-first engine support."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from newt._native import load_native_module, require_native_module
from newt.config import BINNING

from .iv_math import build_iv_summary, calculate_iv_from_counts, prepare_feature_for_iv

VALID_ENGINES = frozenset(["auto", "rust", "python"])


def calculate_batch_iv(
    X: pd.DataFrame,
    y: pd.Series,
    features: Optional[Sequence[str]] = None,
    bins: int = BINNING.DEFAULT_BUCKETS,
    epsilon: float = BINNING.DEFAULT_EPSILON,
    engine: str = "auto",
) -> pd.DataFrame:
    """Calculate IV for many features."""
    if engine not in VALID_ENGINES:
        raise ValueError(
            f"engine must be one of {sorted(VALID_ENGINES)}, got: {engine}"
        )

    feature_names = list(features) if features is not None else X.columns.tolist()
    target = pd.to_numeric(y, errors="coerce")
    valid_target = target.isin([0, 1])
    target_valid = target.loc[valid_target].astype(int)

    if engine == "python":
        values = [
            _calculate_single_iv_mixed(
                X.loc[valid_target, feature],
                target_valid,
                bins=bins,
                epsilon=epsilon,
            )
            for feature in feature_names
        ]
        return pd.DataFrame({"feature": feature_names, "iv": values})

    if engine == "rust":
        value_map = _calculate_batch_iv_rust(
            X=X,
            target=target_valid,
            feature_names=feature_names,
            bins=bins,
            epsilon=epsilon,
            strict=True,
            fallback=False,
        )
    else:
        value_map = _calculate_batch_iv_rust(
            X=X,
            target=target_valid,
            feature_names=feature_names,
            bins=bins,
            epsilon=epsilon,
            strict=False,
            fallback=True,
        )

    return pd.DataFrame(
        {
            "feature": feature_names,
            "iv": [float(value_map.get(feature, np.nan)) for feature in feature_names],
        }
    )


def _calculate_batch_iv_rust(
    X: pd.DataFrame,
    target: pd.Series,
    feature_names: Sequence[str],
    bins: int,
    epsilon: float,
    strict: bool,
    fallback: bool,
) -> Dict[str, float]:
    module = require_native_module() if strict else load_native_module()
    if module is None:
        if fallback:
            return _calculate_batch_iv_python_subset(
                X=X,
                target=target,
                feature_names=feature_names,
                bins=bins,
                epsilon=epsilon,
            )
        raise ImportError("Rust native extension is unavailable.")

    results: Dict[str, float] = {}
    numeric_features = [
        feature
        for feature in feature_names
        if pd.api.types.is_numeric_dtype(X[feature])
    ]
    categorical_features = [
        feature for feature in feature_names if feature not in numeric_features
    ]

    if numeric_features:
        try:
            numpy_fn = getattr(module, "calculate_batch_iv_numpy", None)
            if not callable(numpy_fn):
                raise RuntimeError("Rust numeric batch IV function is unavailable.")

            feature_arrays = [
                np.ascontiguousarray(
                    pd.to_numeric(
                        X.loc[target.index, feature],
                        errors="coerce",
                    ).to_numpy(dtype=np.float64)
                )
                for feature in numeric_features
            ]
            target_array = np.ascontiguousarray(target.to_numpy(dtype=np.int64))
            values = numpy_fn(feature_arrays, target_array, int(bins), float(epsilon))
            results.update(
                {
                    feature: float(value)
                    for feature, value in zip(numeric_features, values)
                }
            )
        except Exception:
            if fallback:
                results.update(
                    _calculate_batch_iv_python_subset(
                        X=X,
                        target=target,
                        feature_names=numeric_features,
                        bins=bins,
                        epsilon=epsilon,
                    )
                )
            else:
                raise

    if categorical_features:
        try:
            categorical_fn = getattr(module, "calculate_batch_categorical_iv", None)
            if not callable(categorical_fn):
                raise RuntimeError("Rust categorical batch IV function is unavailable.")

            feature_vectors = []
            for feature in categorical_features:
                series = X.loc[target.index, feature]
                vector = series.astype("object").where(series.notna(), None).tolist()
                feature_vectors.append(vector)
            values = categorical_fn(feature_vectors, target.astype(int).tolist())
            results.update(
                {
                    feature: float(value)
                    for feature, value in zip(categorical_features, values)
                }
            )
        except Exception:
            if fallback:
                results.update(
                    _calculate_batch_iv_python_subset(
                        X=X,
                        target=target,
                        feature_names=categorical_features,
                        bins=bins,
                        epsilon=epsilon,
                    )
                )
            else:
                raise

    return results


def _calculate_batch_iv_python_subset(
    X: pd.DataFrame,
    target: pd.Series,
    feature_names: Sequence[str],
    bins: int,
    epsilon: float,
) -> Dict[str, float]:
    return {
        feature: _calculate_single_iv_mixed(
            X.loc[target.index, feature],
            target,
            bins=bins,
            epsilon=epsilon,
        )
        for feature in feature_names
    }


def _calculate_single_iv_mixed(
    series: pd.Series,
    target: pd.Series,
    bins: int,
    epsilon: float,
) -> float:
    if pd.api.types.is_numeric_dtype(series):
        return _calculate_single_iv(
            pd.to_numeric(series, errors="coerce"),
            target,
            bins=bins,
            epsilon=epsilon,
        )

    prepared = prepare_feature_for_iv(series, buckets=bins)
    _, iv_value = build_iv_summary(prepared, target)
    return float(iv_value)


def _calculate_single_iv(
    series: pd.Series,
    target: pd.Series,
    bins: int,
    epsilon: float,
) -> float:
    """Python reference implementation for numeric IV."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = numeric.dropna()
    if non_missing.empty or non_missing.nunique() <= 1:
        return 0.0

    edges = _build_quantile_edges(non_missing.to_numpy(dtype=float), bins)
    good_counts = np.zeros(len(edges) - 1 + 1, dtype=float)
    bad_counts = np.zeros(len(edges) - 1 + 1, dtype=float)

    for value, label in zip(numeric.tolist(), target.tolist()):
        index = _bin_index(value, edges)
        if label == 1:
            bad_counts[index] += 1
        else:
            good_counts[index] += 1

    total_good = good_counts.sum()
    total_bad = bad_counts.sum()
    if total_good == 0 or total_bad == 0:
        return 0.0

    _ = epsilon
    return calculate_iv_from_counts(good_counts, bad_counts)


def _build_quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    unique = np.unique(values)
    if unique.size <= 1:
        return np.array([-np.inf, np.inf], dtype=float)
    sorted_values = np.sort(values.astype(float))
    unique_bins = min(bins, unique.size)
    positions = [
        int(round((len(sorted_values) - 1) * (index / unique_bins)))
        for index in range(unique_bins + 1)
    ]
    edges = np.unique(sorted_values[positions])
    if edges.size < 2:
        return np.array([-np.inf, np.inf], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_index(value: object, edges: np.ndarray) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return len(edges) - 1
    return int(np.searchsorted(edges[1:-1], float(value), side="right"))


def _load_rust_extension():
    """Compatibility helper retained for packaging tests."""
    return require_native_module()
