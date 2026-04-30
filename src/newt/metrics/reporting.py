"""Reusable metrics helpers for report generation."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from newt.config import BINNING
from newt.metrics._common import (
    build_score_edges,
    count_values_by_edges,
    psi_from_counts,
)
from newt.metrics.binary_metrics import (
    calculate_binary_metrics as _unified_binary_metrics,
)
from newt.metrics.psi import calculate_grouped_psi, calculate_psi
from newt.report_sort_utils import month_sort_key as _shared_month_sort_key
from newt.report_sort_utils import ordered_month_values as _shared_ordered_month_values
from newt.report_sort_utils import sort_report_frame as _shared_sort_report_frame
from newt.report_sort_utils import tag_sort_key as _shared_tag_sort_key


def build_reference_quantile_bins(
    reference: pd.Series,
    bins: int = BINNING.DEFAULT_BUCKETS,
) -> np.ndarray:
    """Build quantile edges from a reference series."""
    numeric = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    return build_score_edges(numeric, bins)


def assign_reference_bins(values: pd.Series, edges: Sequence[float]) -> pd.Series:
    """Assign values to reference bins."""
    numeric = pd.to_numeric(values, errors="coerce")
    binned = pd.cut(
        numeric,
        bins=np.asarray(edges, dtype=float),
        include_lowest=True,
        duplicates="drop",
    )
    return binned.cat.add_categories("Missing").fillna("Missing")


def calculate_binary_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
    lift_use_descending_score: bool = True,
    reverse_auc_label: bool = False,
    metrics_mode: str = "exact",
    bins: int = 10,
) -> Dict[str, float]:
    """Calculate summary metrics for a binary label/score pair.

    Delegates to :func:`newt.metrics.binary_metrics.calculate_binary_metrics`
    which uses a single ``roc_curve`` call for both AUC and KS and a
    single ``argsort`` for all lift levels.

    Args:
        y_true: Binary labels.
        y_score: Numeric score/probability.
        lift_use_descending_score: Whether lift top-k uses higher score
            as higher risk.
        reverse_auc_label: If True, report AUC = 1 - AUC(y, s).
        metrics_mode: ``"exact"`` or ``"binned"``.
        bins: Number of bins for binned mode.
    """
    return _unified_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        lift_use_descending_score=lift_use_descending_score,
        reverse_auc_label=reverse_auc_label,
        metrics_mode=metrics_mode,
        bins=bins,
    )


def calculate_latest_month_psi(
    data: pd.DataFrame,
    tag_col: str,
    month_col: str,
    score_col: str,
    engine: str = "auto",
) -> pd.DataFrame:
    """Calculate PSI for each month against the latest month within each tag."""
    grouped = calculate_grouped_psi(
        data=data,
        group_cols=[month_col],
        score_col=score_col,
        reference_mode="latest",
        reference_col=month_col,
        partition_cols=[tag_col],
        engine=engine,
        include_stats=False,
    )
    if grouped.empty:
        return pd.DataFrame(columns=[tag_col, month_col, "latest_month_psi"])
    result = grouped.rename(columns={"psi": "latest_month_psi"})[
        [tag_col, month_col, "latest_month_psi"]
    ]
    return _sort_report_frame(result, tag_column=tag_col, month_column=month_col)


def calculate_grouped_binary_metrics(
    data: pd.DataFrame,
    group_cols: Sequence[str],
    label_col: str,
    score_col: str,
    tag_col: Optional[str] = None,
    month_col: Optional[str] = None,
    train_reference: Optional[pd.Series] = None,
    latest_month_psi: Optional[pd.DataFrame] = None,
    model_name: str = "",
) -> pd.DataFrame:
    """Calculate grouped binary metrics for reporting."""
    records: List[Dict[str, object]] = []
    # groupby with sort=True is much faster than sorting the entire 10M row DataFrame.
    for group_values, group_frame in data.groupby(
        list(group_cols), dropna=False, sort=True
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_dict = dict(zip(group_cols, group_values))
        metrics = calculate_binary_metrics(
            group_frame[label_col],
            group_frame[score_col],
        )
        record: Dict[str, object] = {
            "样本标签": label_col,
            "模型": model_name or score_col,
            "样本集": group_dict.get(tag_col, "") if tag_col else "",
            "观察点月": group_dict.get(month_col, "") if month_col else "",
            **metrics,
        }

        binary_scores = group_frame.loc[group_frame[label_col].isin([0, 1]), score_col]
        if train_reference is None:
            record["train和各集合的PSI"] = np.nan
        else:
            record["train和各集合的PSI"] = float(
                calculate_psi(train_reference, binary_scores)
            )

        latest_value = np.nan
        if latest_month_psi is not None and tag_col and month_col:
            matched = latest_month_psi.loc[
                (latest_month_psi[tag_col] == record["样本集"])
                & (latest_month_psi[month_col] == record["观察点月"])
            ]
            if not matched.empty:
                latest_value = matched["latest_month_psi"].iloc[0]
        record["近期月对比各集合PSI"] = latest_value
        records.append(record)

    return _sort_report_frame(
        pd.DataFrame(records), tag_column="样本集", month_column="观察点月"
    )


def summarize_label_distribution(
    data: pd.DataFrame,
    label_col: str,
    tag_col: str,
    month_col: str,
    channel_col: Optional[str] = None,
    include_blank_channel: bool = False,
    include_tag: bool = True,
) -> pd.DataFrame:
    """Summarize grey/good/bad counts for sheet 2."""
    group_cols: List[str] = []
    if include_tag:
        group_cols.append(tag_col)
    if channel_col and channel_col in data.columns:
        group_cols.append(channel_col)
    group_cols.append(month_col)

    rows: List[Dict[str, object]] = []
    for group_values, group_frame in data.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_map = dict(zip(group_cols, group_values))
        good = int((group_frame[label_col] == 0).sum())
        bad = int((group_frame[label_col] == 1).sum())
        grey = int((group_frame[label_col] == -1).sum())
        total = good + bad
        channel_value = (
            group_map.get(channel_col, "")
            if channel_col and channel_col in group_map
            else ""
        )
        if include_blank_channel and not channel_value:
            channel_value = ""
        row = {
            "渠道": channel_value,
            "月": group_map.get(month_col, ""),
            "标签": label_col,
            "好": good,
            "坏": bad,
            "灰": grey,
            "总数（去掉灰样本）": total,
            "坏占比（去掉灰样本）": float(bad / total) if total else np.nan,
        }
        if include_tag:
            row = {"样本集": group_map.get(tag_col, ""), **row}
        rows.append(row)
    return _sort_report_frame(
        pd.DataFrame(rows),
        tag_column="样本集" if include_tag else None,
        month_column="月",
    )


def calculate_score_correlation_matrix(
    data: pd.DataFrame,
    score_cols: Sequence[str],
) -> pd.DataFrame:
    """Calculate a score correlation matrix."""
    numeric = data.loc[:, list(score_cols)].apply(pd.to_numeric, errors="coerce")
    return numeric.corr()


def calculate_portrait_means_by_score_bin(
    data: pd.DataFrame,
    score_cols: Sequence[str],
    variable_cols: Sequence[str],
    bins: int = BINNING.DEFAULT_BUCKETS,
) -> pd.DataFrame:
    """Compare portrait means across score deciles."""
    rows: List[Dict[str, object]] = []
    for score_col in score_cols:
        edges = build_reference_quantile_bins(data[score_col], bins=bins)
        binned = assign_reference_bins(data[score_col], edges)
        for bin_label, bin_frame in data.assign(_score_bin=binned).groupby(
            "_score_bin", dropna=False
        ):
            for variable in variable_cols:
                rows.append(
                    {
                        "模型": score_col,
                        "分组": str(bin_label),
                        "画像变量": variable,
                        "均值": pd.to_numeric(
                            bin_frame[variable],
                            errors="coerce",
                        ).mean(),
                    }
                )
    return pd.DataFrame(rows)


def calculate_bin_performance_table(
    data: pd.DataFrame,
    label_col: str,
    score_col: str,
    edges: Sequence[float],
) -> pd.DataFrame:
    """Calculate bin-level model performance table."""
    frame = data.loc[data[label_col].isin([0, 1]), [label_col, score_col]].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["bin"] = assign_reference_bins(frame[score_col], edges)

    rows: List[Dict[str, object]] = []
    total_bad = int((frame[label_col] == 1).sum())
    total_all = int(len(frame))
    total_good = total_all - total_bad
    overall_bad_rate = float(total_bad / total_all) if total_all else np.nan

    for bin_label, bin_frame in frame.groupby(
        "bin", dropna=False, sort=False, observed=True
    ):
        bads = int((bin_frame[label_col] == 1).sum())
        goods = int((bin_frame[label_col] == 0).sum())
        total = bads + goods
        bad_rate = float(bads / total) if total else np.nan
        rows.append(
            {
                "bin": str(bin_label),
                "min": _extract_interval_left(bin_label),
                "max": _extract_interval_right(bin_label),
                "goods": goods,
                "bads": bads,
                "total": total,
                "total_prop": float(total / total_all) if total_all else np.nan,
                "goods_prop": float(goods / total_good) if total_good else np.nan,
                "bads_prop": float(bads / total_bad) if total_bad else np.nan,
                "bad_rate": bad_rate,
                "lift": (
                    float(bad_rate / overall_bad_rate)
                    if overall_bad_rate and not np.isnan(overall_bad_rate)
                    else np.nan
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.assign(
        _missing_order=result["bin"].eq("Missing").astype(int),
        _bin_order=result["min"].map(_bin_sort_key),
    ).sort_values(
        ["_missing_order", "_bin_order"],
        ascending=[True, True],
        kind="mergesort",
    )
    result["cum_bads"] = result["bads"].cumsum()
    result["cum_total"] = result["total"].cumsum()
    result["cum_bad_rate"] = result["cum_bads"] / result["cum_total"].clip(lower=1)
    result["cum_bads_prop"] = result["cum_bads"] / max(total_bad, 1)
    if total_all > total_bad:
        result["cum_goods_prop"] = result["goods"].cumsum() / max(
            total_all - total_bad, 1
        )
        result["ks"] = abs(result["cum_bads_prop"] - result["cum_goods_prop"])
    else:
        result["ks"] = np.nan
    result["cum_lift"] = result["cum_bad_rate"] / max(overall_bad_rate, 1e-8)

    ordered_columns = [
        "bin",
        "min",
        "max",
        "goods",
        "bads",
        "total",
        "total_prop",
        "goods_prop",
        "bads_prop",
        "bad_rate",
        "cum_bad_rate",
        "cum_bads_prop",
        "ks",
        "lift",
        "cum_lift",
    ]
    return result.reindex(columns=ordered_columns).reset_index(drop=True)


def calculate_feature_psi(
    reference: pd.Series,
    actual: pd.Series,
    edges: Sequence[float],
) -> float:
    """Calculate PSI for a feature using reference edges."""
    ref_numeric = pd.to_numeric(reference, errors="coerce")
    act_numeric = pd.to_numeric(actual, errors="coerce")
    if len(edges) < 2:
        return 0.0
    ref_counts = _count_bins(ref_numeric, edges)
    act_counts = _count_bins(act_numeric, edges)
    return psi_from_counts(ref_counts, act_counts, epsilon=BINNING.DEFAULT_EPSILON)


def _count_bins(values: pd.Series, edges: Sequence[float]) -> np.ndarray:
    numeric = values.to_numpy(dtype=float, copy=False)
    return count_values_by_edges(
        values=numeric,
        edges=np.asarray(edges, dtype=float),
        include_missing_bucket=True,
    )


def _extract_interval_left(interval_label: object) -> float:
    if hasattr(interval_label, "left"):
        return float(interval_label.left)
    if interval_label == "Missing":
        return np.nan
    text = str(interval_label).replace("[", "").replace("(", "")
    return float(text.split(",")[0].strip())


def _extract_interval_right(interval_label: object) -> float:
    if hasattr(interval_label, "right"):
        return float(interval_label.right)
    if interval_label == "Missing":
        return np.nan
    text = str(interval_label).replace("]", "").replace(")", "")
    return float(text.split(",")[1].strip())


def _sort_report_frame(
    frame: pd.DataFrame,
    tag_column: Optional[str] = None,
    month_column: Optional[str] = None,
) -> pd.DataFrame:
    return _shared_sort_report_frame(
        frame=frame,
        tag_column=tag_column,
        month_column=month_column,
        leading_columns=["样本标签", "模型", "维度列", "维度值"],
    )


def _ordered_month_values(values: pd.Series) -> List[object]:
    return _shared_ordered_month_values(values)


def _month_sort_key(value: object) -> str:
    return _shared_month_sort_key(value)


def _tag_sort_key(value: object) -> tuple[int, str]:
    return _shared_tag_sort_key(value)


def _bin_sort_key(value: object) -> float:
    if pd.isna(value):
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")
