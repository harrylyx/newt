"""Reusable metrics helpers for report generation."""

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from newt.config import BINNING
from newt.metrics.auc import calculate_auc
from newt.metrics.ks import calculate_ks
from newt.metrics.lift import calculate_lift_at_k
from newt.metrics.psi import calculate_psi


PERCENT_LEVELS = (0.10, 0.05, 0.02, 0.01)


def build_reference_quantile_bins(
    reference: pd.Series,
    bins: int = BINNING.DEFAULT_BUCKETS,
) -> np.ndarray:
    """Build quantile edges from a reference series."""
    numeric = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)

    unique = np.unique(numeric)
    if unique.size <= 1:
        return np.array([-np.inf, np.inf], dtype=float)

    quantiles = np.linspace(0, 1, min(bins, unique.size) + 1)
    edges = np.quantile(numeric, quantiles)
    edges = np.unique(edges.astype(float))
    if edges.size < 2:
        return np.array([-np.inf, np.inf], dtype=float)

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_reference_bins(values: pd.Series, edges: Sequence[float]) -> pd.Series:
    """Assign values to reference bins."""
    numeric = pd.to_numeric(values, errors="coerce")
    binned = pd.cut(
        numeric,
        bins=np.asarray(edges, dtype=float),
        include_lowest=True,
        duplicates="drop",
    )
    return binned.astype("object").where(~numeric.isna(), "Missing")


def calculate_binary_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
) -> Dict[str, float]:
    """Calculate summary metrics for a binary label/score pair."""
    mask = y_true.isin([0, 1]) & pd.notna(y_score)
    y_clean = y_true.loc[mask].astype(int)
    score_clean = pd.to_numeric(y_score.loc[mask], errors="coerce")
    valid = pd.notna(score_clean)
    y_clean = y_clean.loc[valid]
    score_clean = score_clean.loc[valid]

    total = int(len(y_true))
    good = int((y_true == 0).sum())
    bad = int((y_true == 1).sum())
    binary_total = good + bad
    bad_rate = float(bad / binary_total) if binary_total else np.nan

    if len(y_clean) == 0 or y_clean.nunique() < 2:
        metrics = {"KS": np.nan, "AUC": np.nan}
        lifts = {f"{int(level * 100)}%lift": np.nan for level in PERCENT_LEVELS}
    else:
        metrics = {
            "KS": calculate_ks(y_clean, score_clean),
            "AUC": calculate_auc(y_clean, score_clean),
        }
        lifts = {
            f"{int(level * 100)}%lift": calculate_lift_at_k(
                y_clean.to_numpy(),
                score_clean.to_numpy(),
                k=level,
            )
            for level in PERCENT_LEVELS
        }

    return {
        "总": total,
        "好": good,
        "坏": bad,
        "坏占比": bad_rate,
        **metrics,
        **lifts,
    }


def calculate_latest_month_psi(
    data: pd.DataFrame,
    tag_col: str,
    month_col: str,
    score_col: str,
) -> pd.DataFrame:
    """Calculate PSI for each month against the latest month within each tag."""
    records: List[Dict[str, object]] = []
    for tag_value, tag_frame in data.groupby(tag_col, dropna=False):
        latest_month = tag_frame[month_col].max()
        expected = tag_frame.loc[tag_frame[month_col] == latest_month, score_col]
        for month_value, month_frame in tag_frame.groupby(month_col, dropna=False):
            psi_value = 0.0
            if month_value != latest_month:
                psi_value = calculate_psi(expected, month_frame[score_col])
            records.append(
                {
                    tag_col: tag_value,
                    month_col: month_value,
                    "latest_month_psi": float(psi_value),
                }
            )
    return pd.DataFrame(records)


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
    ordered = data.sort_values(list(group_cols))

    for group_values, group_frame in ordered.groupby(list(group_cols), dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_dict = dict(zip(group_cols, group_values))
        metrics = calculate_binary_metrics(group_frame[label_col], group_frame[score_col])
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

    return pd.DataFrame(records)


def summarize_label_distribution(
    data: pd.DataFrame,
    label_col: str,
    tag_col: str,
    month_col: str,
    channel_col: Optional[str] = None,
    include_blank_channel: bool = False,
) -> pd.DataFrame:
    """Summarize grey/good/bad counts for sheet 2."""
    group_cols = [tag_col, month_col]
    if channel_col and channel_col in data.columns:
        group_cols.insert(1, channel_col)

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
            group_map.get(channel_col, "") if channel_col and channel_col in group_map else ""
        )
        if include_blank_channel and not channel_value:
            channel_value = ""
        rows.append(
            {
                "样本集": group_map.get(tag_col, ""),
                "渠道": channel_value,
                "月": group_map.get(month_col, ""),
                "标签": label_col,
                "好": good,
                "坏": bad,
                "灰": grey,
                "总数（去掉灰样本）": total,
                "坏占比（去掉灰样本）": float(bad / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


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
    frame["bin"] = assign_reference_bins(frame[score_col], edges)

    rows: List[Dict[str, object]] = []
    total_bad = int((frame[label_col] == 1).sum())
    total_all = int(len(frame))
    cumulative_bad = 0
    cumulative_total = 0

    for bin_label, bin_frame in frame.groupby("bin", dropna=False, sort=False):
        bads = int((bin_frame[label_col] == 1).sum())
        goods = int((bin_frame[label_col] == 0).sum())
        total = bads + goods
        cumulative_bad += bads
        cumulative_total += total
        bad_rate = float(bads / total) if total else np.nan
        overall_bad_rate = float(total_bad / total_all) if total_all else np.nan
        rows.append(
            {
                "bin": str(bin_label),
                "min": _extract_interval_left(bin_label),
                "max": _extract_interval_right(bin_label),
                "bads": bads,
                "goods": goods,
                "total": total,
                "bad_rate": bad_rate,
                "cum_bad_rate": float(cumulative_bad / cumulative_total)
                if cumulative_total
                else np.nan,
                "cum_bads_prop": float(cumulative_bad / total_bad) if total_bad else np.nan,
                "ks": np.nan,
                "lift": float(bad_rate / overall_bad_rate)
                if overall_bad_rate and not np.isnan(overall_bad_rate)
                else np.nan,
                "cum_lift": float((cumulative_bad / cumulative_total) / overall_bad_rate)
                if cumulative_total and overall_bad_rate and not np.isnan(overall_bad_rate)
                else np.nan,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty and total_bad and total_all > total_bad:
        result["cum_goods_prop"] = result["goods"].cumsum() / max(total_all - total_bad, 1)
        result["ks"] = abs(result["cum_bads_prop"] - result["cum_goods_prop"])
        result = result.drop(columns=["cum_goods_prop"])
    return result


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
    return _psi_from_counts(ref_counts, act_counts)


def _count_bins(values: pd.Series, edges: Sequence[float]) -> np.ndarray:
    numeric = values.to_numpy(dtype=float, copy=False)
    nan_count = int(np.isnan(numeric).sum())
    counts, _ = np.histogram(numeric[~np.isnan(numeric)], bins=np.asarray(edges, dtype=float))
    return np.append(counts, nan_count)


def _psi_from_counts(expected_counts: np.ndarray, actual_counts: np.ndarray) -> float:
    expected_total = expected_counts.sum()
    actual_total = actual_counts.sum()
    if expected_total == 0 or actual_total == 0:
        return np.nan
    epsilon = BINNING.DEFAULT_EPSILON
    expected_pct = np.maximum(expected_counts / expected_total, epsilon)
    actual_pct = np.maximum(actual_counts / actual_total, epsilon)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


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
