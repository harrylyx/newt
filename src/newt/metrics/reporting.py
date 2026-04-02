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
TAG_ORDER = {"train": 0, "test": 1, "oot": 2, "oos": 3}


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
    for tag_value, tag_frame in data.groupby(tag_col, dropna=False, sort=False):
        months = _ordered_month_values(tag_frame[month_col])
        latest_month = months[-1] if months else ""
        expected = tag_frame.loc[tag_frame[month_col] == latest_month, score_col]
        for month_value in months:
            month_frame = tag_frame.loc[tag_frame[month_col] == month_value]
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
    return _sort_report_frame(
        pd.DataFrame(records), tag_column=tag_col, month_column=month_col
    )


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
        metrics = calculate_binary_metrics(
            group_frame[label_col], group_frame[score_col]
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
            group_map.get(channel_col, "")
            if channel_col and channel_col in group_map
            else ""
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
    return _sort_report_frame(
        pd.DataFrame(rows), tag_column="样本集", month_column="月"
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
    overall_bad_rate = float(total_bad / total_all) if total_all else np.nan

    for bin_label, bin_frame in frame.groupby("bin", dropna=False, sort=False):
        bads = int((bin_frame[label_col] == 1).sum())
        goods = int((bin_frame[label_col] == 0).sum())
        total = bads + goods
        bad_rate = float(bads / total) if total else np.nan
        rows.append(
            {
                "bin": str(bin_label),
                "min": _extract_interval_left(bin_label),
                "max": _extract_interval_right(bin_label),
                "bads": bads,
                "goods": goods,
                "total": total,
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
        _bad_rate_order=result["bad_rate"].fillna(-np.inf),
        _bin_order=result["min"].map(_bin_sort_key),
    ).sort_values(
        ["_missing_order", "_bad_rate_order", "_bin_order"],
        ascending=[True, False, True],
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
    return result.drop(
        columns=[
            "_missing_order",
            "_bad_rate_order",
            "_bin_order",
            "cum_bads",
            "cum_total",
            "cum_goods_prop",
        ],
        errors="ignore",
    ).reset_index(drop=True)


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
    counts, _ = np.histogram(
        numeric[~np.isnan(numeric)], bins=np.asarray(edges, dtype=float)
    )
    return np.append(counts, nan_count)


def _psi_from_counts(expected_counts: np.ndarray, actual_counts: np.ndarray) -> float:
    expected_total = expected_counts.sum()
    actual_total = actual_counts.sum()
    if expected_total == 0 or actual_total == 0:
        return np.nan
    epsilon = BINNING.DEFAULT_EPSILON
    expected_pct = np.maximum(expected_counts / expected_total, epsilon)
    actual_pct = np.maximum(actual_counts / actual_total, epsilon)
    return float(
        np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
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
    if frame.empty:
        return frame

    ordered = frame.copy()
    sort_columns: List[str] = []
    helper_columns: List[str] = []

    if tag_column and tag_column in ordered.columns:
        ordered["_tag_order"] = ordered[tag_column].map(_tag_sort_key)
        sort_columns.append("_tag_order")
        helper_columns.append("_tag_order")

    if month_column and month_column in ordered.columns:
        ordered["_month_order"] = ordered[month_column].map(_month_sort_key)
        sort_columns.append("_month_order")
        helper_columns.append("_month_order")

    for column in ["样本标签", "模型", "维度列", "维度值"]:
        if column in ordered.columns:
            sort_columns.append(column)

    if not sort_columns:
        return ordered.reset_index(drop=True)

    ordered = ordered.sort_values(sort_columns, kind="mergesort")
    return ordered.drop(columns=helper_columns, errors="ignore").reset_index(drop=True)


def _ordered_month_values(values: pd.Series) -> List[object]:
    unique_values = pd.Series(values).drop_duplicates().tolist()
    return sorted(unique_values, key=_month_sort_key)


def _month_sort_key(value: object) -> str:
    if pd.isna(value) or value == "":
        return "999999"
    text = str(value).strip()
    if text.isdigit() and len(text) == 6:
        return text
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y%m")
    return f"999998{text}"


def _tag_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    return TAG_ORDER.get(text.lower(), len(TAG_ORDER)), text


def _bin_sort_key(value: object) -> float:
    if pd.isna(value):
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")
