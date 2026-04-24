"""Shared metric helpers used by public metric entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, Sequence[float], pd.Series]


@dataclass(frozen=True)
class BinaryMetricInput:
    """Normalized binary-label and score arrays plus label counts."""

    total: int
    good: int
    bad: int
    bad_rate: float
    y_clean: np.ndarray
    score_clean: np.ndarray


def to_numeric_array(values: ArrayLike) -> np.ndarray:
    """Convert array-like values to a one-dimensional float array."""
    return pd.to_numeric(
        pd.Series(np.asarray(values).ravel()),
        errors="coerce",
    ).to_numpy(dtype=float)


def prepare_binary_metric_input(
    y_true: Union[pd.Series, np.ndarray, Sequence[float]],
    y_score: Union[pd.Series, np.ndarray, Sequence[float]],
) -> BinaryMetricInput:
    """Normalize binary labels and scores using the legacy filtering rules."""
    if isinstance(y_true, pd.Series):
        y_series = y_true
    else:
        y_series = pd.Series(np.asarray(y_true).ravel())

    if isinstance(y_score, pd.Series):
        s_series = y_score
    else:
        s_series = pd.Series(np.asarray(y_score).ravel())

    total = int(len(y_series))
    good = int((y_series == 0).sum())
    bad = int((y_series == 1).sum())
    binary_total = good + bad
    bad_rate = float(bad / binary_total) if binary_total else np.nan

    mask = y_series.isin([0, 1]) & pd.notna(s_series)
    y_clean = y_series.loc[mask].astype(int).to_numpy()
    score_clean = pd.to_numeric(s_series.loc[mask], errors="coerce")
    valid = pd.notna(score_clean)
    y_clean = y_clean[valid.to_numpy()]
    score_clean = score_clean.loc[valid].to_numpy(dtype=float)

    return BinaryMetricInput(
        total=total,
        good=good,
        bad=bad,
        bad_rate=bad_rate,
        y_clean=y_clean,
        score_clean=score_clean,
    )


def build_quantile_edges(values: np.ndarray, buckets: int) -> np.ndarray:
    """Build unique quantile edges with infinite outer bounds."""
    non_missing = values[~np.isnan(values)]
    if non_missing.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)

    breakpoints = np.percentile(non_missing, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints.astype(float))
    if breakpoints.size < 2:
        return np.array([-np.inf, np.inf], dtype=float)

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    return breakpoints


def build_score_edges(scores: np.ndarray, bins: int) -> np.ndarray:
    """Build score quantile edges from a numeric score array."""
    clean = scores[~np.isnan(scores)]
    unique = np.unique(clean)
    if unique.size <= 1:
        return np.array([-np.inf, np.inf], dtype=float)

    n_bins = min(bins, unique.size)
    return build_quantile_edges(clean, n_bins)


def count_values_by_edges(
    values: np.ndarray,
    edges: np.ndarray,
    include_missing_bucket: bool,
) -> np.ndarray:
    """Count values by fixed edges, optionally appending a missing-value bucket."""
    nan_mask = np.isnan(values)
    non_missing = values[~nan_mask]
    counts, _ = np.histogram(non_missing, bins=np.asarray(edges, dtype=float))
    counts = counts.astype(float)
    if include_missing_bucket:
        counts = np.append(counts, float(nan_mask.sum()))
    return counts


def psi_from_counts(
    expected_counts: np.ndarray,
    actual_counts: np.ndarray,
    epsilon: float,
) -> float:
    """Calculate PSI from aligned expected and actual bin counts."""
    expected_total = float(np.sum(expected_counts))
    actual_total = float(np.sum(actual_counts))
    if expected_total == 0.0 or actual_total == 0.0:
        return float("nan")

    expected_percents = np.maximum(expected_counts / expected_total, epsilon)
    actual_percents = np.maximum(actual_counts / actual_total, epsilon)
    psi_values = (actual_percents - expected_percents) * np.log(
        actual_percents / expected_percents
    )
    return float(np.sum(psi_values))


__all__ = [
    "ArrayLike",
    "BinaryMetricInput",
    "build_quantile_edges",
    "build_score_edges",
    "count_values_by_edges",
    "prepare_binary_metric_input",
    "psi_from_counts",
    "to_numeric_array",
]
