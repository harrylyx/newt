"""Interactive reporting wrappers for use in Jupyter Notebooks."""

from typing import Sequence, Tuple

import pandas as pd

from newt.reporting.report import _vectorized_normalize_month
from newt.reporting.tables import (
    _build_dimensional_comparison,
    _build_model_pair_comparison,
    _build_split_metrics_tables,
)


def calculate_split_metrics(
    data: pd.DataFrame,
    tag_col: str,
    date_col: str,
    label_list: Sequence[str],
    score_col: str,
    model_name: str,
    metrics_mode: str = "exact",
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

    Returns:
        Tuple of two DataFrames:
            - tag_df: Metrics grouped by tag.
            - month_df: Metrics grouped by auto-derived month.
    """
    working_data = data.copy()
    working_data["_report_month"] = _vectorized_normalize_month(working_data[date_col])

    return _build_split_metrics_tables(
        data=working_data,
        tag_col=tag_col,
        month_col="_report_month",
        raw_date_col=date_col,
        label_list=label_list,
        score_col=score_col,
        model_name=model_name,
        reverse_auc_label=False,
        metrics_mode=metrics_mode,
        build_context=None,
    )


def calculate_dimensional_comparison(
    data: pd.DataFrame,
    dim_list: Sequence[str],
    label_list: Sequence[str],
    score_model_columns: Sequence[Tuple[str, str]],
    metrics_mode: str = "exact",
) -> pd.DataFrame:
    """Calculate dimensional comparison metrics.

    Args:
        data: Input DataFrame.
        dim_list: List of dimension column names to split by.
        label_list: List of label column names.
        score_model_columns: List of (model_name, score_column) tuples.
        metrics_mode: 'exact' or 'binned'. Default is 'exact'.

    Returns:
        DataFrame containing metrics grouped by dimensions.
    """
    return _build_dimensional_comparison(
        data=data,
        dim_list=dim_list,
        label_list=label_list,
        score_model_columns=score_model_columns,
        score_metric_options={},
        metrics_mode=metrics_mode,
    )


def calculate_model_comparison(
    data: pd.DataFrame,
    tag_col: str,
    date_col: str,
    label_list: Sequence[str],
    model_columns: Sequence[Tuple[str, str]],
    group_mode: str = "month",
    metrics_mode: str = "exact",
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

    Returns:
        DataFrame containing model comparison metrics.
    """
    working_data = data.copy()
    working_data["_report_month"] = _vectorized_normalize_month(working_data[date_col])

    return _build_model_pair_comparison(
        data=working_data,
        group_mode=group_mode,
        label_list=label_list,
        model_columns=model_columns,
        tag_col=tag_col,
        month_col="_report_month",
        raw_date_col=date_col,
        score_metric_options={},
        metrics_mode=metrics_mode,
        build_context=None,
    )
