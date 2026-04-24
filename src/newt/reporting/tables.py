"""Compatibility facade for report table builders."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import pandas as pd

from newt.reporting.model_adapter import ModelAdapter
from newt.reporting.table_context import ReportBuildContext, ReportBuildOptions
from newt.results import ModelReportResult, ReportSheet

from .builders import appendix_sheets, feature_metrics, group_metrics, main_sheets
from .builders.orchestrator import build_report_result as _build_report_result_impl
from .builders.sheet_registry import (
    APPENDIX_SHEET_KEY_ORDER,
    APPENDIX_SHEET_LABEL_MAP,
    MAIN_SHEET_KEY_ORDER,
    MAIN_SHEET_OUTPUT_NAME_MAP,
    SHEET_INDEX_SELECTOR_MAP,
    SHEET_KEY_ORDER,
    SHEET_NAME_SELECTOR_MAP,
)
from .builders.sheet_registry import (
    filter_output_sheet_keys as _filter_output_sheet_keys,
)
from .builders.sheet_registry import resolve_build_keys as _resolve_build_keys
from .builders.sheet_registry import (
    resolve_optional_sheet_availability as _resolve_optional_sheet_availability,
)
from .builders.sheet_registry import (
    resolve_output_sheet_names as _resolve_output_sheet_names,
)
from .builders.sheet_registry import resolve_sheet_keys, resolve_sheet_names

# Backward-compatible patch points used by existing tests.
_build_feature_analysis_table = feature_metrics._build_feature_analysis_table
_build_feature_bin_stats = feature_metrics._build_feature_bin_stats
_build_feature_monthly_metrics = feature_metrics._build_feature_monthly_metrics
_load_feature_dictionary = feature_metrics._load_feature_dictionary
_build_split_metrics_tables = group_metrics._build_split_metrics_tables
_build_group_metrics = group_metrics._build_group_metrics

# Additional private aliases retained for compatibility.
_lookup_feature_meta = feature_metrics._lookup_feature_meta
_determine_feature_columns = feature_metrics._determine_feature_columns
_build_score_metric_options = group_metrics._build_score_metric_options
_lookup_reverse_auc = group_metrics._lookup_reverse_auc
_resolve_score_model_columns = group_metrics._resolve_score_model_columns
_build_dimensional_comparison = group_metrics._build_dimensional_comparison
_build_model_pair_comparison = group_metrics._build_model_pair_comparison

# Public/semipublic builders re-exported for compatibility.
build_overview_sheet = main_sheets.build_overview_sheet
build_model_design_sheet = main_sheets.build_model_design_sheet
build_scorecard_details_sheet = main_sheets.build_scorecard_details_sheet
build_model_performance_sheet = main_sheets.build_model_performance_sheet
build_dimensional_comparison_sheet = appendix_sheets.build_dimensional_comparison_sheet
build_model_comparison_sheet = appendix_sheets.build_model_comparison_sheet
build_portrait_sheet = appendix_sheets.build_portrait_sheet


def _sync_legacy_patch_points() -> None:
    feature_metrics._build_feature_analysis_table = _build_feature_analysis_table
    feature_metrics._build_feature_bin_stats = _build_feature_bin_stats
    feature_metrics._build_feature_monthly_metrics = _build_feature_monthly_metrics
    feature_metrics._load_feature_dictionary = _load_feature_dictionary
    feature_metrics.build_feature_analysis_table = _build_feature_analysis_table
    feature_metrics.build_feature_bin_stats = _build_feature_bin_stats
    feature_metrics.build_feature_monthly_metrics = _build_feature_monthly_metrics
    feature_metrics.load_feature_dictionary = _load_feature_dictionary

    group_metrics._build_split_metrics_tables = _build_split_metrics_tables
    group_metrics._build_group_metrics = _build_group_metrics
    group_metrics.build_split_metrics_tables = _build_split_metrics_tables
    group_metrics.build_group_metrics = _build_group_metrics


def build_report_result(
    data: pd.DataFrame,
    model_adapter: ModelAdapter,
    tag_col: str,
    month_col: str,
    raw_date_col: str,
    label_list: Sequence[str],
    score_list: Sequence[str],
    primary_score_name: str,
    report_score_columns: Dict[str, str],
    score_direction_summary: pd.DataFrame,
    dim_list: Sequence[str],
    var_list: Sequence[str],
    feature_df: Optional[pd.DataFrame],
    selected_sheets: Sequence[str],
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
    options: Optional[ReportBuildOptions] = None,
) -> ModelReportResult:
    """Build the full report result object."""
    _sync_legacy_patch_points()
    return _build_report_result_impl(
        data=data,
        model_adapter=model_adapter,
        tag_col=tag_col,
        month_col=month_col,
        raw_date_col=raw_date_col,
        label_list=label_list,
        score_list=score_list,
        primary_score_name=primary_score_name,
        report_score_columns=report_score_columns,
        score_direction_summary=score_direction_summary,
        dim_list=dim_list,
        var_list=var_list,
        feature_df=feature_df,
        selected_sheets=selected_sheets,
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
        options=options,
    )


def build_variable_analysis_sheet(
    data: pd.DataFrame,
    tag_col: str,
    month_col: str,
    primary_label: str,
    feature_cols: Sequence[str],
    feature_dict: pd.DataFrame,
    model_adapter: ModelAdapter,
    build_context: Optional[ReportBuildContext] = None,
) -> ReportSheet:
    """Build variable analysis sheet."""
    _sync_legacy_patch_points()
    return main_sheets.build_variable_analysis_sheet(
        data=data,
        tag_col=tag_col,
        month_col=month_col,
        primary_label=primary_label,
        feature_cols=feature_cols,
        feature_dict=feature_dict,
        model_adapter=model_adapter,
        build_context=build_context,
    )


__all__ = [
    "APPENDIX_SHEET_KEY_ORDER",
    "APPENDIX_SHEET_LABEL_MAP",
    "MAIN_SHEET_KEY_ORDER",
    "MAIN_SHEET_OUTPUT_NAME_MAP",
    "SHEET_INDEX_SELECTOR_MAP",
    "SHEET_KEY_ORDER",
    "SHEET_NAME_SELECTOR_MAP",
    "resolve_sheet_keys",
    "resolve_sheet_names",
    "_resolve_optional_sheet_availability",
    "_filter_output_sheet_keys",
    "_resolve_build_keys",
    "_resolve_output_sheet_names",
    "build_report_result",
    "build_overview_sheet",
    "build_model_design_sheet",
    "build_variable_analysis_sheet",
    "build_scorecard_details_sheet",
    "build_model_performance_sheet",
    "build_dimensional_comparison_sheet",
    "build_model_comparison_sheet",
    "build_portrait_sheet",
    "_build_split_metrics_tables",
    "_build_group_metrics",
    "_build_feature_analysis_table",
    "_build_feature_bin_stats",
    "_build_feature_monthly_metrics",
    "_load_feature_dictionary",
    "_lookup_feature_meta",
    "_determine_feature_columns",
    "_build_score_metric_options",
    "_lookup_reverse_auc",
    "_resolve_score_model_columns",
    "_build_dimensional_comparison",
    "_build_model_pair_comparison",
]
