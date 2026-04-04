"""Shared IV helpers aligned with toad smoothing semantics."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from newt.config import BINNING

MISSING_BIN_LABEL = "Missing"


def prepare_feature_for_iv(
    series: pd.Series,
    buckets: int = BINNING.DEFAULT_BUCKETS,
) -> pd.Series:
    """Prepare a feature column for IV calculation."""
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > buckets:
        try:
            prepared = pd.qcut(series, q=buckets, duplicates="drop").astype("object")
        except ValueError:
            prepared = pd.cut(series, bins=buckets).astype("object")
    else:
        prepared = series.astype("object")

    return prepared.where(pd.notna(prepared), MISSING_BIN_LABEL).astype(str)


def calculate_iv_from_counts(
    good_counts: np.ndarray,
    bad_counts: np.ndarray,
) -> float:
    """Compute IV from per-bin counts using toad-compatible smoothing."""
    total_good_raw = float(np.sum(good_counts))
    total_bad_raw = float(np.sum(bad_counts))
    if total_good_raw <= 0.0 or total_bad_raw <= 0.0:
        return 0.0

    smoothed_good = np.maximum(good_counts.astype(float), 1.0)
    smoothed_bad = np.maximum(bad_counts.astype(float), 1.0)
    dist_good = smoothed_good / total_good_raw
    dist_bad = smoothed_bad / total_bad_raw

    woe = np.log(dist_good / dist_bad)
    iv_contribution = (dist_good - dist_bad) * woe
    return float(np.sum(iv_contribution))


def build_iv_summary(
    feature: pd.Series, target: pd.Series
) -> Tuple[pd.DataFrame, float]:
    """Build IV summary table and IV value for a pre-binned feature."""
    working = pd.DataFrame(
        {
            "bin": feature.where(feature.notna(), MISSING_BIN_LABEL).astype(str),
            "target": pd.to_numeric(target, errors="coerce"),
        }
    )
    working = working.loc[working["target"].isin([0, 1])].copy()

    empty_cols = [
        "total",
        "bad",
        "good",
        "dist_good",
        "dist_bad",
        "woe",
        "iv_contribution",
    ]
    if working.empty:
        return pd.DataFrame(columns=empty_cols), 0.0

    working["target"] = working["target"].astype(int)
    grouped = (
        working.groupby("bin", observed=True)["target"].agg(["count", "sum"]).copy()
    )
    grouped = grouped.rename(columns={"count": "total", "sum": "bad"})
    grouped["good"] = grouped["total"] - grouped["bad"]

    good_counts = grouped["good"].to_numpy(dtype=float)
    bad_counts = grouped["bad"].to_numpy(dtype=float)
    iv_value = calculate_iv_from_counts(good_counts, bad_counts)

    total_good = float(np.sum(good_counts))
    total_bad = float(np.sum(bad_counts))
    if total_good <= 0.0 or total_bad <= 0.0:
        grouped["dist_good"] = 0.0
        grouped["dist_bad"] = 0.0
        grouped["woe"] = 0.0
        grouped["iv_contribution"] = 0.0
        return grouped, 0.0

    smoothed_good = np.maximum(good_counts, 1.0)
    smoothed_bad = np.maximum(bad_counts, 1.0)
    dist_good = smoothed_good / total_good
    dist_bad = smoothed_bad / total_bad
    woe = np.log(dist_good / dist_bad)
    iv_contribution = (dist_good - dist_bad) * woe

    grouped["dist_good"] = dist_good
    grouped["dist_bad"] = dist_bad
    grouped["woe"] = woe
    grouped["iv_contribution"] = iv_contribution
    return grouped, iv_value
