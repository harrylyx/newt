"""
Binning statistics calculation.

Provides comprehensive statistics for binned features.
"""

from typing import Any, List

import numpy as np
import pandas as pd


def get_bin_boundaries(splits: List[float]) -> List[float]:
    """
    Get bin boundaries including -inf and +inf.

    Parameters
    ----------
    splits : List[float]
        Split points.

    Returns
    -------
    List[float]
        Full boundaries including -inf and +inf.
    """
    return [-np.inf] + sorted(splits) + [np.inf]


def calculate_bin_stats(
    X: pd.Series,
    y: pd.Series,
    binned: pd.Series,
    splits: List[float],
    woe_encoder: Any,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """
    Calculate comprehensive binning statistics for a feature.

    Parameters
    ----------
    X : pd.Series
        Original feature data.
    y : pd.Series
        Binary target (0/1).
    binned : pd.Series
        Binned feature data.
    splits : List[float]
        Split points used for binning.
    woe_encoder : WOEEncoder
        Fitted WOE encoder for the feature.
    epsilon : float
        Smoothing factor. Default 1e-8.

    Returns
    -------
    pd.DataFrame
        DataFrame with comprehensive statistics.
    """
    # Get WOE summary from encoder
    woe_summary = woe_encoder.summary_.copy()
    woe_summary = woe_summary.reset_index()
    woe_summary.columns = [
        "bin", "total", "bads", "goods",
        "good_prop", "bad_prop", "woe", "iv"
    ]

    # Sort by bin order
    woe_summary = woe_summary.sort_index().reset_index(drop=True)

    # Calculate min/max for each bin from splits
    boundaries = get_bin_boundaries(splits)
    bins_list = []
    for i in range(len(boundaries) - 1):
        bins_list.append({
            "bin_idx": i,
            "min": boundaries[i],
            "max": boundaries[i + 1],
        })
    bins_df = pd.DataFrame(bins_list)

    # Handle missing bin if present
    if "Missing" in woe_summary["bin"].values:
        missing_idx = len(bins_list)
        bins_df = pd.concat([
            bins_df,
            pd.DataFrame([{
                "bin_idx": missing_idx,
                "min": np.nan,
                "max": np.nan,
            }])
        ], ignore_index=True)

    # Map bin to bin_idx
    woe_summary["bin_idx"] = range(len(woe_summary))
    stats = woe_summary.merge(bins_df, on="bin_idx", how="left")

    # Totals
    total_bads = stats["bads"].sum()
    total_goods = stats["goods"].sum()
    total_all = stats["total"].sum()

    # Rates
    stats["bad_rate"] = stats["bads"] / stats["total"].clip(lower=1)
    stats["good_rate"] = stats["goods"] / stats["total"].clip(lower=1)

    # Odds (goods/bads)
    stats["odds"] = stats["goods"] / stats["bads"].clip(lower=1)

    # Total proportion
    stats["total_prop"] = stats["total"] / max(total_all, 1)

    # Cumulative
    stats["cum_bads"] = stats["bads"].cumsum()
    stats["cum_goods"] = stats["goods"].cumsum()
    stats["cum_total"] = stats["total"].cumsum()
    stats["cum_bads_prop"] = stats["cum_bads"] / max(total_bads, 1)
    stats["cum_goods_prop"] = stats["cum_goods"] / max(total_goods, 1)
    stats["cum_total_prop"] = stats["cum_total"] / max(total_all, 1)

    # KS (difference between cumulative proportions)
    stats["ks"] = abs(stats["cum_bads_prop"] - stats["cum_goods_prop"])

    # Lift = bad_rate / overall_bad_rate
    overall_bad_rate = total_bads / max(total_all, 1)
    stats["lift"] = stats["bad_rate"] / max(overall_bad_rate, epsilon)

    # Reorder columns
    result_cols = [
        "bin", "min", "max", "bads", "goods", "total",
        "bad_rate", "good_rate", "odds",
        "bad_prop", "good_prop", "total_prop",
        "cum_bads", "cum_goods", "cum_total",
        "cum_bads_prop", "cum_goods_prop", "cum_total_prop",
        "ks", "woe", "iv", "lift",
    ]

    return stats[[c for c in result_cols if c in stats.columns]]


def calculate_ks_value(
    stats: pd.DataFrame,
) -> float:
    """
    Get maximum KS value from stats.

    Parameters
    ----------
    stats : pd.DataFrame
        Binning statistics DataFrame.

    Returns
    -------
    float
        Maximum KS value.
    """
    if "ks" not in stats.columns:
        return 0.0
    return float(stats["ks"].max())
