"""Batch IV calculation helpers with a Rust-backed engine."""

from __future__ import annotations

import importlib
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from newt.config import BINNING


def calculate_batch_iv(
    X: pd.DataFrame,
    y: pd.Series,
    features: Optional[Sequence[str]] = None,
    bins: int = BINNING.DEFAULT_BUCKETS,
    epsilon: float = BINNING.DEFAULT_EPSILON,
    engine: str = "rust",
) -> pd.DataFrame:
    """Calculate IV for many features."""
    feature_names = list(features) if features is not None else X.columns.tolist()
    target = pd.to_numeric(y, errors="coerce")
    valid_target = target.isin([0, 1])
    target_values = target.loc[valid_target].astype(int).tolist()

    if engine == "python":
        values = [
            _calculate_single_iv(
                pd.to_numeric(X.loc[valid_target, feature], errors="coerce"),
                target.loc[valid_target].astype(int),
                bins=bins,
                epsilon=epsilon,
            )
            for feature in feature_names
        ]
        return pd.DataFrame({"feature": feature_names, "iv": values})

    if engine != "rust":
        raise ValueError("engine must be 'rust' or 'python'")

    rust_module = _load_rust_extension()
    feature_vectors = [
        [
            None if pd.isna(value) else float(value)
            for value in pd.to_numeric(
                X.loc[valid_target, feature],
                errors="coerce",
            ).tolist()
        ]
        for feature in feature_names
    ]
    values = rust_module.calculate_batch_iv(
        feature_vectors,
        target_values,
        int(bins),
        float(epsilon),
    )
    return pd.DataFrame({"feature": feature_names, "iv": values})


def _calculate_single_iv(
    series: pd.Series,
    target: pd.Series,
    bins: int,
    epsilon: float,
) -> float:
    """Python reference implementation for IV."""
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

    dist_good = np.maximum(good_counts / total_good, epsilon)
    dist_bad = np.maximum(bad_counts / total_bad, epsilon)
    return float(np.sum((dist_good - dist_bad) * np.log(dist_good / dist_bad)))


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
    """Import the compiled Rust extension from the package.

    In installed environments the extension is available as
    ``newt._newt_iv_rust``.  If the extension is not present (e.g. when
    installed from a pure-Python sdist without a Rust toolchain), a clear
    ``ImportError`` is raised.  No hidden local compilation is attempted.
    """
    try:
        return importlib.import_module("newt._newt_iv_rust")
    except ImportError:
        raise ImportError(
            "The compiled Rust IV extension (newt._newt_iv_rust) is not "
            "available. Install Newt from an official wheel that includes "
            "the prebuilt Rust extension, or build from source with "
            "'maturin develop --manifest-path rust/newt_iv_rust/Cargo.toml'."
        )
