"""Grouped metric computation helpers for report builders."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from newt.config import BINNING
from newt.metrics.auc import calculate_auc as _calculate_weighted_auc
from newt.metrics.binary_metrics import PERCENT_LEVELS
from newt.metrics.binary_metrics import (
    calculate_binary_metrics as _unified_binary_metrics,
)
from newt.metrics.binary_metrics import (
    calculate_binary_metrics_batch as _unified_binary_metrics_batch,
)
from newt.metrics.ks import calculate_ks as _calculate_weighted_ks
from newt.metrics.psi import calculate_psi_batch
from newt.report_sort_utils import month_sort_key as _shared_month_sort_key
from newt.report_sort_utils import ordered_month_values as _shared_ordered_month_values
from newt.report_sort_utils import ordered_tag_values as _shared_ordered_tag_values
from newt.report_sort_utils import sort_report_frame as _shared_sort_report_frame
from newt.report_sort_utils import tag_sort_key as _shared_tag_sort_key
from newt.reporting.table_context import ReportBuildContext

LOGGER = logging.getLogger("newt.reporting.tables")

_VALID_METRIC_BASIS = frozenset({"count", "amount"})

CORE_METRIC_COLUMNS: Tuple[str, ...] = (
    "总",
    "好",
    "坏",
    "坏占比",
    "KS",
    "AUC",
    "10%lift",
    "5%lift",
    "2%lift",
    "1%lift",
)

AMOUNT_METRIC_COLUMNS: Tuple[str, ...] = (
    "逾期本金",
    "放款金额",
    "金额坏占比",
    "放款金额占比",
    "逾期本金占比",
    "10%金额lift",
    "5%金额lift",
    "2%金额lift",
    "1%金额lift",
)


def _resolve_metric_basis(metric_basis: str) -> str:
    normalized = str(metric_basis).strip().lower()
    if normalized not in _VALID_METRIC_BASIS:
        raise ValueError("metric_basis must be 'count' or 'amount'")
    return normalized


def _has_amount_columns(
    prin_bal_amount_col: Optional[str],
    loan_amount_col: Optional[str],
) -> bool:
    return prin_bal_amount_col is not None and loan_amount_col is not None


def _validate_amount_columns(
    data: pd.DataFrame,
    prin_bal_amount_col: Optional[str],
    loan_amount_col: Optional[str],
) -> None:
    if (prin_bal_amount_col is None) ^ (loan_amount_col is None):
        raise ValueError(
            "prin_bal_amount_col and loan_amount_col must be provided together"
        )
    if prin_bal_amount_col is None:
        return
    missing = [
        column
        for column in [prin_bal_amount_col, loan_amount_col]
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {sorted(set(missing))}")


def _validate_metric_basis_amount_dependency(
    metric_basis: str,
    prin_bal_amount_col: Optional[str],
    loan_amount_col: Optional[str],
) -> None:
    if metric_basis == "amount" and not _has_amount_columns(
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
    ):
        raise ValueError(
            "metric_basis='amount' requires prin_bal_amount_col and loan_amount_col"
        )


def _safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _prepare_metric_vectors(
    frame: pd.DataFrame,
    label_col: str,
    score_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_numeric = pd.to_numeric(frame[score_col], errors="coerce")
    mask = frame[label_col].isin([0, 1]) & score_numeric.notna()
    labels = frame.loc[mask, label_col].astype(int).to_numpy(dtype=int)
    scores = score_numeric.loc[mask].to_numpy(dtype=float)
    return labels, scores, mask.to_numpy(dtype=bool)


def _compute_amount_lifts(
    y_score: np.ndarray,
    overdue_amount: np.ndarray,
    loan_amount: np.ndarray,
    descending: bool,
    levels: Sequence[float] = PERCENT_LEVELS,
) -> Dict[str, float]:
    if len(y_score) == 0:
        return {f"{int(level * 100)}%金额lift": np.nan for level in levels}

    global_amount_bad_rate = _safe_divide(
        float(np.nansum(overdue_amount)),
        float(np.nansum(loan_amount)),
    )
    if pd.isna(global_amount_bad_rate) or global_amount_bad_rate == 0:
        return {f"{int(level * 100)}%金额lift": np.nan for level in levels}

    sort_scores = y_score if descending else -y_score
    ranked_index = np.argsort(sort_scores)[::-1]

    results: Dict[str, float] = {}
    sample_total = len(y_score)
    for level in levels:
        top_count = max(1, int(np.ceil(sample_total * float(level))))
        top_index = ranked_index[:top_count]
        top_overdue = float(np.nansum(overdue_amount[top_index]))
        top_loan = float(np.nansum(loan_amount[top_index]))
        top_bad_rate = _safe_divide(top_overdue, top_loan)
        results[f"{int(level * 100)}%金额lift"] = _safe_divide(
            top_bad_rate,
            global_amount_bad_rate,
        )
    return results


def _compute_amount_totals(
    frame: pd.DataFrame,
    prin_bal_amount_col: str,
    loan_amount_col: str,
) -> Tuple[float, float, float]:
    total_prin = float(pd.to_numeric(frame[prin_bal_amount_col], errors="coerce").sum())
    total_loan = float(pd.to_numeric(frame[loan_amount_col], errors="coerce").sum())
    total_bad_ratio = _safe_divide(total_prin, total_loan)
    return total_prin, total_loan, total_bad_ratio


def _calculate_amount_basis_metrics(
    frame: pd.DataFrame,
    label_col: str,
    score_col: str,
    prin_bal_amount_col: str,
    loan_amount_col: str,
    reverse_auc_label: bool = False,
) -> Dict[str, float]:
    principal_amount = pd.to_numeric(
        frame[prin_bal_amount_col], errors="coerce"
    ).fillna(0)
    loan_amount = pd.to_numeric(frame[loan_amount_col], errors="coerce").fillna(0)

    binary_mask = frame[label_col].isin([0, 1]).to_numpy(dtype=bool)
    total_amount = float(np.nansum(loan_amount.to_numpy(dtype=float)[binary_mask]))
    bad_amount = float(np.nansum(principal_amount.to_numpy(dtype=float)[binary_mask]))
    good_amount = total_amount - bad_amount
    bad_rate = _safe_divide(bad_amount, total_amount)

    y_true, y_score, metric_mask = _prepare_metric_vectors(frame, label_col, score_col)
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        auc_value = np.nan
        ks_value = np.nan
        amount_lifts = {f"{int(level * 100)}%lift": np.nan for level in PERCENT_LEVELS}
    else:
        sample_weight = loan_amount.to_numpy(dtype=float)[metric_mask]
        sample_weight = np.nan_to_num(sample_weight, nan=0.0, posinf=0.0, neginf=0.0)
        sample_weight = np.clip(sample_weight, a_min=0.0, a_max=None)
        auc_value = _calculate_weighted_auc(
            y_true=y_true,
            y_prob=y_score,
            sample_weight=sample_weight,
        )
        if reverse_auc_label and not pd.isna(auc_value):
            auc_value = float(1.0 - auc_value)
        ks_value = _calculate_weighted_ks(
            y_true=y_true,
            y_prob=y_score,
            sample_weight=sample_weight,
        )
        raw_amount_lifts = _compute_amount_lifts(
            y_score=y_score,
            overdue_amount=principal_amount.to_numpy(dtype=float)[metric_mask],
            loan_amount=sample_weight,
            descending=not reverse_auc_label,
            levels=PERCENT_LEVELS,
        )
        amount_lifts = {
            f"{int(level * 100)}%lift": raw_amount_lifts.get(
                f"{int(level * 100)}%金额lift",
                np.nan,
            )
            for level in PERCENT_LEVELS
        }

    return {
        "总": total_amount,
        "好": good_amount,
        "坏": bad_amount,
        "坏占比": bad_rate,
        "KS": ks_value,
        "AUC": auc_value,
        **amount_lifts,
    }


def _build_amount_metrics(
    frame: pd.DataFrame,
    label_col: str,
    score_col: str,
    prin_bal_amount_col: str,
    loan_amount_col: str,
    total_prin: float,
    total_loan: float,
    reverse_auc_label: bool = False,
) -> Dict[str, float]:
    group_prin = float(pd.to_numeric(frame[prin_bal_amount_col], errors="coerce").sum())
    group_loan = float(pd.to_numeric(frame[loan_amount_col], errors="coerce").sum())
    amount_bad_ratio = _safe_divide(group_prin, group_loan)
    _, y_score, metric_mask = _prepare_metric_vectors(frame, label_col, score_col)
    overdue_amount = (
        pd.to_numeric(frame[prin_bal_amount_col], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float)[metric_mask]
    )
    loan_amount = (
        pd.to_numeric(frame[loan_amount_col], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float)[metric_mask]
    )
    amount_lifts = _compute_amount_lifts(
        y_score=y_score,
        overdue_amount=overdue_amount,
        loan_amount=loan_amount,
        descending=not reverse_auc_label,
        levels=PERCENT_LEVELS,
    )
    return {
        "逾期本金": group_prin,
        "放款金额": group_loan,
        "金额坏占比": amount_bad_ratio,
        "放款金额占比": _safe_divide(group_loan, total_loan),
        "逾期本金占比": _safe_divide(group_prin, total_prin),
        "10%金额lift": amount_lifts["10%金额lift"],
        "5%金额lift": amount_lifts["5%金额lift"],
        "2%金额lift": amount_lifts["2%金额lift"],
        "1%金额lift": amount_lifts["1%金额lift"],
    }


def _build_split_metrics_tables(
    data: pd.DataFrame,
    tag_col: str,
    month_col: str,
    raw_date_col: str,
    label_list: Sequence[str],
    score_col: str,
    model_name: str,
    reverse_auc_label: bool = False,
    metrics_mode: str = "exact",
    metric_basis: str = "count",
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
    build_context: Optional[ReportBuildContext] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _validate_amount_columns(data, prin_bal_amount_col, loan_amount_col)
    resolved_metric_basis = _resolve_metric_basis(metric_basis)
    _validate_metric_basis_amount_dependency(
        metric_basis=resolved_metric_basis,
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
    )
    cache_key = (
        tuple(label_list),
        score_col,
        model_name,
        bool(reverse_auc_label),
        str(metrics_mode),
        resolved_metric_basis,
        tag_col,
        month_col,
        raw_date_col,
        prin_bal_amount_col,
        loan_amount_col,
    )
    if build_context is not None:
        cached = build_context.cache_get_split_metrics(cache_key)
        if cached is not None:
            return cached

    LOGGER.debug(
        "_build_split_metrics_tables started | rows=%d label_list=%s "
        "tag_col=%s month_col=%s score_col=%s",
        len(data),
        list(label_list),
        tag_col,
        month_col,
        score_col,
    )

    tag_rows: List[pd.DataFrame] = []
    month_rows: List[pd.DataFrame] = []
    month_latest_psi = _build_latest_month_psi_map(
        data,
        month_col=month_col,
        score_col=score_col,
        build_context=build_context,
    )

    for label_col in label_list:
        train_reference = data.loc[
            (data[tag_col] == "train") & data[label_col].isin([0, 1]),
            score_col,
        ]
        tag_table = _build_group_metrics(
            data=data,
            group_cols=[tag_col],
            label_col=label_col,
            score_col=score_col,
            tag_col=tag_col,
            month_col=month_col,
            raw_date_col=raw_date_col,
            train_reference=train_reference,
            model_name=model_name,
            reverse_auc_label=reverse_auc_label,
            metrics_mode=metrics_mode,
            metric_basis=resolved_metric_basis,
            prin_bal_amount_col=prin_bal_amount_col,
            loan_amount_col=loan_amount_col,
            build_context=build_context,
        )
        tag_rows.append(tag_table)

        month_table = _build_group_metrics(
            data=data,
            group_cols=[month_col],
            label_col=label_col,
            score_col=score_col,
            tag_col=tag_col,
            month_col=month_col,
            raw_date_col=raw_date_col,
            train_reference=train_reference,
            model_name=model_name,
            reverse_auc_label=reverse_auc_label,
            metrics_mode=metrics_mode,
            metric_basis=resolved_metric_basis,
            prin_bal_amount_col=prin_bal_amount_col,
            loan_amount_col=loan_amount_col,
            build_context=build_context,
        )
        if not month_table.empty:
            month_table["近期月对比各集合PSI"] = month_table["观察点月"].map(
                month_latest_psi
            )
        month_rows.append(month_table)

    tag_result = _sort_report_table(
        pd.concat(tag_rows, ignore_index=True),
        tag_column="样本集",
        leading_columns=["样本标签", "模型"],
    )
    month_result = _sort_report_table(
        pd.concat(month_rows, ignore_index=True),
        month_column="观察点月",
        leading_columns=["样本标签", "模型"],
    )
    if build_context is not None:
        build_context.cache_set_split_metrics(cache_key, (tag_result, month_result))
    return tag_result, month_result


def _build_model_pair_comparison(
    data: pd.DataFrame,
    group_mode: str,
    label_list: Sequence[str],
    model_columns: Sequence[Tuple[str, str]],
    tag_col: str,
    month_col: str,
    raw_date_col: str,
    score_metric_options: Dict[str, Dict[str, bool]],
    metrics_mode: str = "exact",
    metric_basis: str = "count",
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
    build_context: Optional[ReportBuildContext] = None,
) -> pd.DataFrame:
    _validate_amount_columns(data, prin_bal_amount_col, loan_amount_col)
    resolved_metric_basis = _resolve_metric_basis(metric_basis)
    _validate_metric_basis_amount_dependency(
        metric_basis=resolved_metric_basis,
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
    )
    if group_mode not in {"tag", "month"}:
        raise ValueError(f"Unknown comparison mode: {group_mode}")

    score_columns = [score_col for _, score_col in model_columns]
    filtered_data = _filter_model_comparison_intersection(
        data=data,
        score_columns=score_columns,
    )

    group_cols = [tag_col] if group_mode == "tag" else [month_col]
    frames: List[pd.DataFrame] = []
    latest_map_by_model = {
        model_name: _build_latest_month_psi_map(
            filtered_data,
            month_col=month_col,
            score_col=score_col,
            build_context=build_context,
        )
        for model_name, score_col in model_columns
    }

    for model_name, score_col in model_columns:
        for label_col in label_list:
            train_reference = filtered_data.loc[
                (filtered_data[tag_col] == "train")
                & filtered_data[label_col].isin([0, 1]),
                score_col,
            ]
            table = _build_group_metrics(
                data=filtered_data,
                group_cols=group_cols,
                label_col=label_col,
                score_col=score_col,
                tag_col=tag_col,
                month_col=month_col,
                raw_date_col=raw_date_col,
                train_reference=train_reference,
                model_name=model_name,
                reverse_auc_label=_lookup_reverse_auc(score_metric_options, model_name),
                metrics_mode=metrics_mode,
                metric_basis=resolved_metric_basis,
                prin_bal_amount_col=prin_bal_amount_col,
                loan_amount_col=loan_amount_col,
                build_context=build_context,
            )
            if group_mode == "month":
                table["近期月对比各集合PSI"] = table["观察点月"].map(
                    latest_map_by_model[model_name]
                )
            frames.append(table)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    model_order = {
        model_name: index for index, (model_name, _) in enumerate(model_columns)
    }
    return _sort_pair_comparison_table(
        combined,
        model_order=model_order,
        group_mode=group_mode,
    )


def _filter_model_comparison_intersection(
    data: pd.DataFrame,
    score_columns: Sequence[str],
) -> pd.DataFrame:
    if not score_columns:
        return data
    missing = [column for column in score_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {sorted(set(missing))}")

    intersection_mask = pd.Series(True, index=data.index, dtype=bool)
    for score_col in score_columns:
        numeric = pd.to_numeric(data[score_col], errors="coerce")
        finite_mask = pd.Series(
            np.isfinite(numeric.to_numpy(dtype=float, copy=False)),
            index=data.index,
        )
        valid_mask = numeric.notna() & finite_mask & numeric.gt(0)
        intersection_mask &= valid_mask
    return data.loc[intersection_mask].copy()


def _build_group_metrics(
    data: pd.DataFrame,
    group_cols: Sequence[str],
    label_col: str,
    score_col: str,
    tag_col: str,
    month_col: str,
    raw_date_col: str,
    train_reference: Optional[pd.Series] = None,
    latest_month_psi: Optional[pd.DataFrame] = None,
    model_name: str = "",
    reverse_auc_label: bool = False,
    metrics_mode: str = "exact",
    metric_basis: str = "count",
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
    build_context: Optional[ReportBuildContext] = None,
) -> pd.DataFrame:
    _validate_amount_columns(data, prin_bal_amount_col, loan_amount_col)
    resolved_metric_basis = _resolve_metric_basis(metric_basis)
    _validate_metric_basis_amount_dependency(
        metric_basis=resolved_metric_basis,
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
    )
    include_amount_metrics = _has_amount_columns(
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
    )
    group_key = (
        id(data),
        tuple(group_cols),
        label_col,
        score_col,
        tag_col,
        month_col,
        raw_date_col,
        model_name,
        bool(reverse_auc_label),
        str(metrics_mode),
        resolved_metric_basis,
        prin_bal_amount_col,
        loan_amount_col,
        bool(train_reference is not None),
    )
    if build_context is not None:
        cached = build_context.cache_get_group_metrics(group_key)
        if cached is not None:
            return cached

    LOGGER.debug(
        "_build_group_metrics started | rows=%d group_cols=%s "
        "label_col=%s score_col=%s",
        len(data),
        list(group_cols),
        label_col,
        score_col,
    )

    records: List[Dict[str, object]] = []
    resolved_metrics_mode = (
        build_context.options.metrics_mode
        if build_context is not None
        else metrics_mode
    )
    resolved_engine = (
        build_context.options.engine if build_context is not None else "python"
    )
    _group_cols_set = set(group_cols)
    if _group_cols_set == {tag_col}:
        required_columns = [*group_cols, label_col, score_col, tag_col, raw_date_col]
    else:
        required_columns = [*group_cols, label_col, score_col, tag_col, month_col]
    if include_amount_metrics:
        required_columns.extend([str(prin_bal_amount_col), str(loan_amount_col)])
    required_columns = list(dict.fromkeys(required_columns))
    ordered = data.loc[:, required_columns].copy()
    for group_col in group_cols:
        normalized = _normalize_group_column(
            ordered[group_col],
            is_tag_column=(group_col == tag_col),
        )
        categories = [value for value in pd.unique(normalized) if pd.notna(value)]
        if not categories:
            categories = ["None" if group_col == tag_col else ""]
        ordered[group_col] = pd.Categorical(
            normalized,
            categories=categories,
            ordered=False,
        )

    psi_values: List[float] = []
    grouped_frames = list(
        ordered.groupby(list(group_cols), sort=True, dropna=False, observed=True)
    )
    metric_groups = [
        (group_frame[label_col], group_frame[score_col])
        for _, group_frame in grouped_frames
    ]
    if resolved_metric_basis == "count":
        grouped_metrics = _unified_binary_metrics_batch(
            groups=metric_groups,
            lift_use_descending_score=not reverse_auc_label,
            reverse_auc_label=reverse_auc_label,
            metrics_mode=resolved_metrics_mode,
            engine=resolved_engine,
        )
    else:
        grouped_metrics = [
            _calculate_amount_basis_metrics(
                group_frame,
                label_col=label_col,
                score_col=score_col,
                prin_bal_amount_col=str(prin_bal_amount_col),
                loan_amount_col=str(loan_amount_col),
                reverse_auc_label=reverse_auc_label,
            )
            for _, group_frame in grouped_frames
        ]

    total_prin = np.nan
    total_loan = np.nan
    if include_amount_metrics:
        total_prin, total_loan, _ = _compute_amount_totals(
            ordered,
            prin_bal_amount_col=str(prin_bal_amount_col),
            loan_amount_col=str(loan_amount_col),
        )
    if train_reference is not None:
        group_binary_scores = [
            group_frame.loc[group_frame[label_col].isin([0, 1]), score_col]
            for _, group_frame in grouped_frames
        ]
        psi_values = calculate_psi_batch(
            expected=train_reference,
            actual_groups=group_binary_scores,
            buckets=BINNING.DEFAULT_BUCKETS,
            engine=build_context.options.engine if build_context else "python",
        )

    for group_index, (group_values, group_frame) in enumerate(grouped_frames):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        metrics = grouped_metrics[group_index]
        sample_tag_value = _resolve_sample_set_label(
            group_frame=group_frame,
            group_cols=group_cols,
            tag_col=tag_col,
            month_col=month_col,
        )
        observation_month_value = _resolve_observation_window(
            group_frame=group_frame,
            group_cols=group_cols,
            tag_col=tag_col,
            month_col=month_col,
            raw_date_col=raw_date_col,
        )
        record: Dict[str, object] = {
            "样本标签": label_col,
            "模型": model_name or score_col,
            "样本集": sample_tag_value,
            "观察点月": observation_month_value,
            **metrics,
        }
        if include_amount_metrics:
            record.update(
                _build_amount_metrics(
                    group_frame,
                    label_col=label_col,
                    score_col=score_col,
                    prin_bal_amount_col=str(prin_bal_amount_col),
                    loan_amount_col=str(loan_amount_col),
                    total_prin=total_prin,
                    total_loan=total_loan,
                    reverse_auc_label=reverse_auc_label,
                )
            )

        if train_reference is None:
            record["train和各集合的PSI"] = np.nan
        else:
            record["train和各集合的PSI"] = float(psi_values[group_index])

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

    result = pd.DataFrame(records)
    if not result.empty:
        leading = ["样本标签", "模型", "样本集", "观察点月"]
        psi_cols = ["train和各集合的PSI", "近期月对比各集合PSI"]
        result = _reorder_metric_columns(
            result,
            leading_columns=leading,
            trailing_columns=psi_cols,
        )

    result = _sort_report_table(result, tag_column="样本集", month_column="观察点月")
    if build_context is not None:
        build_context.cache_set_group_metrics(group_key, result)
    return result


def _calculate_report_metrics(
    frame: pd.DataFrame,
    label_col: str,
    score_col: str,
    reverse_auc_label: bool = False,
    metrics_mode: str = "exact",
    metric_basis: str = "count",
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
) -> Dict[str, float]:
    """Calculate report metrics using the unified binary metrics path."""
    resolved_metric_basis = _resolve_metric_basis(metric_basis)
    if resolved_metric_basis == "amount":
        _validate_metric_basis_amount_dependency(
            metric_basis=resolved_metric_basis,
            prin_bal_amount_col=prin_bal_amount_col,
            loan_amount_col=loan_amount_col,
        )
        return _calculate_amount_basis_metrics(
            frame=frame,
            label_col=label_col,
            score_col=score_col,
            prin_bal_amount_col=str(prin_bal_amount_col),
            loan_amount_col=str(loan_amount_col),
            reverse_auc_label=reverse_auc_label,
        )
    return _unified_binary_metrics(
        y_true=frame[label_col],
        y_score=frame[score_col],
        lift_use_descending_score=not reverse_auc_label,
        reverse_auc_label=reverse_auc_label,
        metrics_mode=metrics_mode,
    )


def _build_latest_month_psi_map(
    data: pd.DataFrame,
    month_col: str,
    score_col: str,
    build_context: Optional[ReportBuildContext] = None,
) -> Dict[object, float]:
    cache_key = (score_col,)
    if build_context is not None:
        cached = build_context.cache_get_latest_month_psi(cache_key)
        if cached is not None:
            return cached

    LOGGER.debug(
        "_build_latest_month_psi_map started | rows=%d month_col=%s score_col=%s",
        len(data),
        month_col,
        score_col,
    )

    if data.empty:
        return {}
    month_values = _ordered_month_values(data[month_col])
    if not month_values:
        return {}
    latest_month = month_values[-1]
    reference = data.loc[data[month_col] == latest_month, score_col]
    result: Dict[object, float] = {latest_month: 0.0}
    compare_months = month_values[:-1]
    compare_groups = [
        data.loc[data[month_col] == month_value, score_col]
        for month_value in compare_months
    ]
    compare_values = calculate_psi_batch(
        expected=reference,
        actual_groups=compare_groups,
        buckets=BINNING.DEFAULT_BUCKETS,
        engine=build_context.options.engine if build_context else "python",
    )
    for month_value, psi_value in zip(compare_months, compare_values):
        result[month_value] = float(psi_value)
    if build_context is not None:
        build_context.cache_set_latest_month_psi(cache_key, result)
    return result


def _sort_pair_comparison_table(
    frame: pd.DataFrame,
    model_order: Dict[str, int],
    group_mode: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    ordered = frame.copy()
    ordered["_label_order"] = ordered["样本标签"].astype(str)
    ordered["_model_order"] = ordered["模型"].map(model_order).fillna(99)
    sort_columns = ["_label_order"]
    helper_columns = ["_label_order", "_model_order"]

    if group_mode == "tag":
        ordered["_tag_order"] = ordered["样本集"].map(_tag_sort_key)
        sort_columns.append("_tag_order")
        helper_columns.append("_tag_order")
    else:
        ordered["_month_order"] = ordered["观察点月"].map(_month_sort_key)
        sort_columns.append("_month_order")
        helper_columns.append("_month_order")

    sort_columns.append("_model_order")
    ordered = ordered.sort_values(sort_columns, kind="mergesort")
    return ordered.drop(columns=helper_columns, errors="ignore").reset_index(drop=True)


def _resolve_score_model_columns(
    primary_score_name: str,
    score_list: Sequence[str],
    report_score_columns: Dict[str, str],
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for score_name in [primary_score_name, *score_list]:
        report_column = report_score_columns.get(score_name)
        if report_column and score_name not in {name for name, _ in pairs}:
            pairs.append((score_name, report_column))
    return pairs


def _build_dimensional_comparison(
    data: pd.DataFrame,
    dim_list: Sequence[str],
    label_list: Sequence[str],
    score_model_columns: Sequence[Tuple[str, str]],
    score_metric_options: Dict[str, Dict[str, bool]],
    metrics_mode: str = "exact",
    metric_basis: str = "count",
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
) -> pd.DataFrame:
    _validate_amount_columns(data, prin_bal_amount_col, loan_amount_col)
    resolved_metric_basis = _resolve_metric_basis(metric_basis)
    _validate_metric_basis_amount_dependency(
        metric_basis=resolved_metric_basis,
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
    )
    include_amount_metrics = _has_amount_columns(
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
    )
    total_prin = np.nan
    total_loan = np.nan
    if include_amount_metrics:
        total_prin, total_loan, _ = _compute_amount_totals(
            data,
            prin_bal_amount_col=str(prin_bal_amount_col),
            loan_amount_col=str(loan_amount_col),
        )

    rows: List[Dict[str, object]] = []
    for dim_col in dim_list:
        for dim_value, dim_frame in data.groupby(dim_col, dropna=False):
            for label_col in label_list:
                for model_name, score_col in score_model_columns:
                    metrics = _calculate_report_metrics(
                        frame=dim_frame,
                        label_col=label_col,
                        score_col=score_col,
                        reverse_auc_label=_lookup_reverse_auc(
                            score_metric_options, model_name
                        ),
                        metrics_mode=metrics_mode,
                        metric_basis=resolved_metric_basis,
                        prin_bal_amount_col=prin_bal_amount_col,
                        loan_amount_col=loan_amount_col,
                    )
                    row: Dict[str, object] = {
                        "维度列": dim_col,
                        "维度值": dim_value,
                        "样本标签": label_col,
                        "模型": model_name,
                        **metrics,
                    }
                    if include_amount_metrics:
                        row.update(
                            _build_amount_metrics(
                                dim_frame,
                                label_col=label_col,
                                score_col=score_col,
                                prin_bal_amount_col=str(prin_bal_amount_col),
                                loan_amount_col=str(loan_amount_col),
                                total_prin=total_prin,
                                total_loan=total_loan,
                                reverse_auc_label=_lookup_reverse_auc(
                                    score_metric_options, model_name
                                ),
                            )
                        )
                    rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        leading = ["维度列", "维度值", "样本标签", "模型"]
        result = _reorder_metric_columns(result, leading_columns=leading)
    return result


def _resolve_sample_set_label(
    group_frame: pd.DataFrame,
    group_cols: Sequence[str],
    tag_col: str,
    month_col: str,
) -> str:
    if list(group_cols) == [tag_col]:
        first = group_frame[tag_col].iloc[0] if not group_frame.empty else ""
        return "" if pd.isna(first) else str(first)
    if list(group_cols) == [month_col]:
        return _join_present_tags(group_frame[tag_col])
    return ""


def _resolve_observation_window(
    group_frame: pd.DataFrame,
    group_cols: Sequence[str],
    tag_col: str,
    month_col: str,
    raw_date_col: str,
) -> str:
    if list(group_cols) == [tag_col] and raw_date_col in group_frame.columns:
        return _format_date_range(group_frame[raw_date_col])
    if list(group_cols) == [month_col]:
        first = group_frame[month_col].iloc[0] if not group_frame.empty else ""
        return "" if pd.isna(first) else str(first)
    return ""


def _normalize_group_column(values: pd.Series, is_tag_column: bool) -> pd.Series:
    normalized = values.astype("object").copy()
    if is_tag_column:
        text = normalized.astype(str).str.strip()
        missing_mask = normalized.isna() | text.eq("")
        normalized.loc[missing_mask] = "None"
        return normalized
    normalized.loc[normalized.isna()] = ""
    return normalized


def _format_date_range(values: pd.Series) -> str:
    truncated = values.astype(str).str.slice(stop=10)
    unique_days = pd.unique(truncated)
    unique_days = [v for v in unique_days if pd.notna(v) and str(v).lower() != "nan"]
    if not unique_days:
        return ""

    parsed = pd.to_datetime(unique_days, errors="coerce").dropna()
    if parsed.empty:
        return ""
    return f"{parsed.min().strftime('%Y%m%d')}-{parsed.max().strftime('%Y%m%d')}"


def _join_present_tags(values: pd.Series) -> str:
    tags = [
        str(value) for value in values.dropna().tolist() if str(value).strip() != ""
    ]
    if not tags:
        return ""
    ordered = sorted(set(tags), key=_tag_sort_key)
    return ",".join(ordered)


def _build_score_metric_options(
    score_direction_summary: pd.DataFrame,
) -> Dict[str, Dict[str, bool]]:
    if (
        score_direction_summary.empty
        or "分数字段" not in score_direction_summary.columns
    ):
        return {}

    options: Dict[str, Dict[str, bool]] = {}
    for _, row in score_direction_summary.iterrows():
        options[str(row["分数字段"])] = {
            "reverse_auc_label": bool(row.get("AUC反向", False)),
        }
    return options


def _lookup_reverse_auc(
    score_metric_options: Dict[str, Dict[str, bool]],
    score_name: str,
) -> bool:
    return bool(
        score_metric_options.get(str(score_name), {}).get("reverse_auc_label", False)
    )


def _sort_report_table(
    frame: pd.DataFrame,
    tag_column: Optional[str] = None,
    month_column: Optional[str] = None,
    leading_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    return _shared_sort_report_frame(
        frame=frame,
        tag_column=tag_column,
        month_column=month_column,
        leading_columns=leading_columns,
    )


def _reorder_metric_columns(
    frame: pd.DataFrame,
    leading_columns: Optional[Sequence[str]] = None,
    trailing_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Reorder report metric columns with a stable core metric order."""
    if frame.empty:
        return frame

    preferred = [
        *(leading_columns or ()),
        *CORE_METRIC_COLUMNS,
        *AMOUNT_METRIC_COLUMNS,
        *(trailing_columns or ()),
    ]
    ordered: List[str] = []
    for column in preferred:
        if column in frame.columns and column not in ordered:
            ordered.append(column)
    remaining = [column for column in frame.columns if column not in ordered]
    return frame.reindex(columns=ordered + remaining)


def _ordered_month_values(values: pd.Series) -> List[object]:
    return _shared_ordered_month_values(values)


def _ordered_tag_values(values: pd.Series) -> List[object]:
    return _shared_ordered_tag_values(values)


def _month_sort_key(value: object) -> str:
    return _shared_month_sort_key(value)


def _tag_sort_key(value: object) -> tuple[int, str]:
    return _shared_tag_sort_key(value)


build_split_metrics_tables = _build_split_metrics_tables
build_group_metrics = _build_group_metrics
build_score_metric_options = _build_score_metric_options
lookup_reverse_auc = _lookup_reverse_auc
resolve_score_model_columns = _resolve_score_model_columns
build_dimensional_comparison = _build_dimensional_comparison
build_model_pair_comparison = _build_model_pair_comparison
