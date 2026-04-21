"""Interactive reporting wrappers for use in Jupyter Notebooks."""

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from newt.metrics.reporting import (
    assign_reference_bins,
    build_reference_quantile_bins,
    calculate_bin_performance_table,
)
from newt.reporting.report import _vectorized_normalize_month
from newt.reporting.tables import (
    _build_dimensional_comparison,
    _build_model_pair_comparison,
    _build_split_metrics_tables,
)

_VALID_SCORE_TYPES = frozenset({"probability", "score"})


def _resolve_reverse_auc_label(score_type: str) -> bool:
    normalized = str(score_type).strip().lower()
    if normalized not in _VALID_SCORE_TYPES:
        raise ValueError("score_type must be 'probability' or 'score'")
    return normalized == "score"


def _validate_amount_metric_columns(
    data: pd.DataFrame,
    prin_bal_amount_col: Optional[str],
    loan_amount_col: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if (prin_bal_amount_col is None) ^ (loan_amount_col is None):
        raise ValueError(
            "prin_bal_amount_col and loan_amount_col must be provided together"
        )
    if prin_bal_amount_col is None:
        return None, None
    missing = [
        column
        for column in [prin_bal_amount_col, loan_amount_col]
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {sorted(set(missing))}")
    return prin_bal_amount_col, loan_amount_col


def _safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _safe_divide_series(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    values = numerator / denominator
    return values.replace([np.inf, -np.inf], np.nan)


def _safe_divide_scalar_series(
    numerator: pd.Series,
    denominator: float,
) -> pd.Series:
    if pd.isna(denominator) or denominator == 0:
        return pd.Series(np.nan, index=numerator.index, dtype=float)
    values = numerator / denominator
    return values.replace([np.inf, -np.inf], np.nan)


_CORE_METRIC_COLUMNS = (
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
_AMOUNT_EXTENSION_COLUMNS = (
    "放款金额",
    "逾期本金",
    "金额坏占比",
    "金额AUC",
    "金额KS",
    "10%金额lift",
    "5%金额lift",
    "2%金额lift",
    "1%金额lift",
)


def _build_amount_extension_frame(
    amount_metrics_frame: pd.DataFrame,
    key_columns: Sequence[str],
) -> pd.DataFrame:
    columns = [*key_columns, *_AMOUNT_EXTENSION_COLUMNS]
    if amount_metrics_frame.empty:
        return pd.DataFrame(columns=columns)

    output = amount_metrics_frame.loc[:, key_columns].copy()
    mapping = {
        "总": "放款金额",
        "坏": "逾期本金",
        "坏占比": "金额坏占比",
        "AUC": "金额AUC",
        "KS": "金额KS",
        "10%lift": "10%金额lift",
        "5%lift": "5%金额lift",
        "2%lift": "2%金额lift",
        "1%lift": "1%金额lift",
    }
    for source_col, target_col in mapping.items():
        if source_col in amount_metrics_frame.columns:
            output[target_col] = amount_metrics_frame[source_col]
        else:
            output[target_col] = np.nan
    return output


def _merge_amount_extension_columns(
    base_frame: pd.DataFrame,
    amount_metrics_frame: pd.DataFrame,
    key_columns: Sequence[str],
    leading_columns: Sequence[str],
) -> pd.DataFrame:
    extension = _build_amount_extension_frame(
        amount_metrics_frame=amount_metrics_frame,
        key_columns=key_columns,
    )
    merged = base_frame.merge(extension, on=list(key_columns), how="left")
    preferred_order = [
        *leading_columns,
        *_CORE_METRIC_COLUMNS,
        *_AMOUNT_EXTENSION_COLUMNS,
    ]
    ordered = [column for column in preferred_order if column in merged.columns]
    remaining = [column for column in merged.columns if column not in ordered]
    return merged.reindex(columns=ordered + remaining)


def calculate_split_metrics(
    data: pd.DataFrame,
    tag_col: str,
    date_col: str,
    label_list: Sequence[str],
    score_col: str,
    model_name: str,
    metrics_mode: str = "exact",
    score_type: str = "probability",
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate split performance metrics by tag and month.

    Args:
        data: Input DataFrame containing the predictions.
        tag_col: Column name indicating sample set (e.g., 'train', 'oot').
        date_col: Date column name; used to automatically generate month column.
        label_list: List of label column names.
        score_col: The score column name.
        model_name: Name of the model.
        metrics_mode: 'exact' or 'binned'. Default is 'exact'.
        score_type: Score semantics: 'probability' (higher=more risky) or
            'score' (higher=less risky).
        prin_bal_amount_col: Optional principal-balance amount column.
        loan_amount_col: Optional total-loan amount column.

    Returns:
        Tuple of two DataFrames:
            - tag_df: Metrics grouped by tag.
            - month_df: Metrics grouped by auto-derived month.
    """
    reverse_auc_label = _resolve_reverse_auc_label(score_type)
    amount_prin_col, amount_loan_col = _validate_amount_metric_columns(
        data=data,
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
    )

    working_data = data.copy()
    working_data["_report_month"] = _vectorized_normalize_month(working_data[date_col])

    tag_df, month_df = _build_split_metrics_tables(
        data=working_data,
        tag_col=tag_col,
        month_col="_report_month",
        raw_date_col=date_col,
        label_list=label_list,
        score_col=score_col,
        model_name=model_name,
        reverse_auc_label=reverse_auc_label,
        metrics_mode=metrics_mode,
        prin_bal_amount_col=None,
        loan_amount_col=None,
        build_context=None,
    )
    if amount_prin_col is None or amount_loan_col is None:
        return tag_df, month_df

    amount_tag_df, amount_month_df = _build_split_metrics_tables(
        data=working_data,
        tag_col=tag_col,
        month_col="_report_month",
        raw_date_col=date_col,
        label_list=label_list,
        score_col=score_col,
        model_name=model_name,
        reverse_auc_label=reverse_auc_label,
        metrics_mode=metrics_mode,
        metric_basis="amount",
        prin_bal_amount_col=amount_prin_col,
        loan_amount_col=amount_loan_col,
        build_context=None,
    )
    key_columns = ["样本标签", "模型", "样本集", "观察点月"]
    return (
        _merge_amount_extension_columns(
            base_frame=tag_df,
            amount_metrics_frame=amount_tag_df,
            key_columns=key_columns,
            leading_columns=key_columns,
        ),
        _merge_amount_extension_columns(
            base_frame=month_df,
            amount_metrics_frame=amount_month_df,
            key_columns=key_columns,
            leading_columns=key_columns,
        ),
    )


def calculate_dimensional_comparison(
    data: pd.DataFrame,
    dim_list: Sequence[str],
    label_list: Sequence[str],
    score_model_columns: Sequence[Tuple[str, str]],
    metrics_mode: str = "exact",
    score_type: str = "probability",
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate dimensional comparison metrics.

    Args:
        data: Input DataFrame.
        dim_list: List of dimension column names to split by.
        label_list: List of label column names.
        score_model_columns: List of (model_name, score_column) tuples.
        metrics_mode: 'exact' or 'binned'. Default is 'exact'.
        score_type: Score semantics: 'probability' (higher=more risky) or
            'score' (higher=less risky).
        prin_bal_amount_col: Optional principal-balance amount column.
        loan_amount_col: Optional total-loan amount column.

    Returns:
        DataFrame containing metrics grouped by dimensions.
    """
    reverse_auc_label = _resolve_reverse_auc_label(score_type)
    score_metric_options = {
        model_name: {"reverse_auc_label": reverse_auc_label}
        for model_name, _ in score_model_columns
    }
    amount_prin_col, amount_loan_col = _validate_amount_metric_columns(
        data=data,
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
    )

    dim_df = _build_dimensional_comparison(
        data=data,
        dim_list=dim_list,
        label_list=label_list,
        score_model_columns=score_model_columns,
        score_metric_options=score_metric_options,
        metrics_mode=metrics_mode,
        prin_bal_amount_col=None,
        loan_amount_col=None,
    )
    if amount_prin_col is None or amount_loan_col is None:
        return dim_df

    amount_dim_df = _build_dimensional_comparison(
        data=data,
        dim_list=dim_list,
        label_list=label_list,
        score_model_columns=score_model_columns,
        score_metric_options=score_metric_options,
        metrics_mode=metrics_mode,
        metric_basis="amount",
        prin_bal_amount_col=amount_prin_col,
        loan_amount_col=amount_loan_col,
    )
    key_columns = ["维度列", "维度值", "样本标签", "模型"]
    return _merge_amount_extension_columns(
        base_frame=dim_df,
        amount_metrics_frame=amount_dim_df,
        key_columns=key_columns,
        leading_columns=key_columns,
    )


def calculate_model_comparison(
    data: pd.DataFrame,
    tag_col: str,
    date_col: str,
    label_list: Sequence[str],
    model_columns: Sequence[Tuple[str, str]],
    group_mode: str = "month",
    metrics_mode: str = "exact",
    score_type: str = "probability",
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compare multiple models directly.

    Args:
        data: Input DataFrame.
        tag_col: Column name indicating sample set (e.g., 'train', 'oot').
        date_col: Date column name; used to generate month column.
        label_list: List of label column names.
        model_columns: List of (model_name, score_column) tuples.
        group_mode: Mode to group by, either 'month' or 'tag'. Default is 'month'.
        metrics_mode: 'exact' or 'binned'. Default is 'exact'.
        score_type: Score semantics: 'probability' (higher=more risky) or
            'score' (higher=less risky).
        prin_bal_amount_col: Optional principal-balance amount column.
        loan_amount_col: Optional total-loan amount column.

    Returns:
        DataFrame containing model comparison metrics.
    """
    reverse_auc_label = _resolve_reverse_auc_label(score_type)
    score_metric_options = {
        model_name: {"reverse_auc_label": reverse_auc_label}
        for model_name, _ in model_columns
    }
    amount_prin_col, amount_loan_col = _validate_amount_metric_columns(
        data=data,
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
    )

    working_data = data.copy()
    working_data["_report_month"] = _vectorized_normalize_month(working_data[date_col])

    comparison_df = _build_model_pair_comparison(
        data=working_data,
        group_mode=group_mode,
        label_list=label_list,
        model_columns=model_columns,
        tag_col=tag_col,
        month_col="_report_month",
        raw_date_col=date_col,
        score_metric_options=score_metric_options,
        metrics_mode=metrics_mode,
        prin_bal_amount_col=None,
        loan_amount_col=None,
        build_context=None,
    )
    if amount_prin_col is None or amount_loan_col is None:
        return comparison_df

    amount_comparison_df = _build_model_pair_comparison(
        data=working_data,
        group_mode=group_mode,
        label_list=label_list,
        model_columns=model_columns,
        tag_col=tag_col,
        month_col="_report_month",
        raw_date_col=date_col,
        score_metric_options=score_metric_options,
        metrics_mode=metrics_mode,
        metric_basis="amount",
        prin_bal_amount_col=amount_prin_col,
        loan_amount_col=amount_loan_col,
        build_context=None,
    )
    key_columns = ["样本标签", "模型", "样本集", "观察点月"]
    return _merge_amount_extension_columns(
        base_frame=comparison_df,
        amount_metrics_frame=amount_comparison_df,
        key_columns=key_columns,
        leading_columns=key_columns,
    )


def calculate_bin_metrics(
    data: pd.DataFrame,
    label_col: str,
    score_col: str,
    q: int = 10,
    bins: Optional[Sequence[float]] = None,
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate bin-level sample and optional amount metrics.

    Args:
        data: Input DataFrame.
        label_col: Binary label column name.
        score_col: Score column name.
        q: Number of quantile bins when ``bins`` is not provided.
        bins: Optional custom split edges.
        prin_bal_amount_col: Optional principal-balance amount column.
        loan_amount_col: Optional total-loan amount column.

    Returns:
        DataFrame containing per-bin sample metrics and optional amount metrics.
    """
    amount_prin_col, amount_loan_col = _validate_amount_metric_columns(
        data=data,
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
    )

    if bins is None:
        if int(q) < 2:
            raise ValueError("q must be >= 2")
        edges = build_reference_quantile_bins(data[score_col], bins=int(q))
    else:
        edge_values = np.asarray(list(bins), dtype=float)
        if edge_values.ndim != 1 or edge_values.size < 2:
            raise ValueError("bins must contain at least two edges")
        if not np.all(np.diff(edge_values) > 0):
            raise ValueError("bins must be strictly increasing")
        edges = edge_values

    result = calculate_bin_performance_table(
        data=data,
        label_col=label_col,
        score_col=score_col,
        edges=edges,
    )
    if result.empty or amount_prin_col is None or amount_loan_col is None:
        return result

    amount_frame = data.loc[
        data[label_col].isin([0, 1]),
        [score_col, amount_prin_col, amount_loan_col],
    ].copy()
    if amount_frame.empty:
        return result

    amount_frame["bin"] = assign_reference_bins(amount_frame[score_col], edges).astype(
        str
    )
    amount_frame["_逾期本金"] = pd.to_numeric(
        amount_frame[amount_prin_col], errors="coerce"
    )
    amount_frame["_放款金额"] = pd.to_numeric(
        amount_frame[amount_loan_col], errors="coerce"
    )

    amount_grouped = (
        amount_frame.groupby("bin", dropna=False, sort=False)[["_逾期本金", "_放款金额"]]
        .sum()
        .reset_index()
        .rename(columns={"_逾期本金": "逾期本金", "_放款金额": "放款金额"})
    )
    merged = result.merge(amount_grouped, on="bin", how="left")

    total_prin_bal = float(amount_frame["_逾期本金"].sum())
    total_loan = float(amount_frame["_放款金额"].sum())
    overall_amount_bad_rate = _safe_divide(total_prin_bal, total_loan)

    merged["金额坏占比"] = _safe_divide_series(merged["逾期本金"], merged["放款金额"])
    merged["放款金额占比"] = _safe_divide_scalar_series(merged["放款金额"], total_loan)
    merged["逾期本金占比"] = _safe_divide_scalar_series(merged["逾期本金"], total_prin_bal)
    if pd.isna(overall_amount_bad_rate) or overall_amount_bad_rate == 0:
        merged["金额lift"] = np.nan
    else:
        merged["金额lift"] = merged["金额坏占比"] / overall_amount_bad_rate
    merged["金额lift"] = merged["金额lift"].replace([np.inf, -np.inf], np.nan)

    ordered_columns = [
        *result.columns.tolist(),
        "逾期本金",
        "放款金额",
        "金额坏占比",
        "放款金额占比",
        "逾期本金占比",
        "金额lift",
    ]
    return merged.reindex(columns=ordered_columns)
