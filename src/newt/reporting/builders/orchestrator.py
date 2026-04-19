"""Top-level report result orchestration."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from newt.metrics.reporting import build_reference_quantile_bins
from newt.reporting.model_adapter import ModelAdapter
from newt.reporting.table_context import ReportBuildContext, ReportBuildOptions
from newt.results import ModelReportResult, ReportSheet

from . import (
    appendix_sheets,
    feature_metrics,
    group_metrics,
    main_sheets,
    sheet_registry,
)

LOGGER = logging.getLogger("newt.reporting.tables")


def _timed_sheet_build(builder) -> Tuple[ReportSheet, float]:
    start = time.perf_counter()
    sheet = builder()
    return sheet, time.perf_counter() - start


def _log_context_stage(
    context: ReportBuildContext,
    stage_name: str,
    elapsed: float,
    extra: str = "",
) -> None:
    context.record_timing(stage_name, elapsed)
    suffix = f" | {extra}" if extra else ""
    LOGGER.debug(
        "build_report_result step finished | step=%s elapsed=%.3fs%s",
        stage_name,
        elapsed,
        suffix,
    )


def _log_top_context_timings(
    context: ReportBuildContext,
    limit: int = 5,
) -> None:
    ranked = sorted(context.timings, key=lambda item: item[1], reverse=True)
    if not ranked:
        return
    top = ", ".join(f"{name}:{elapsed:.3f}s" for name, elapsed in ranked[:limit])
    LOGGER.debug("build_report_result slowest_steps | %s", top)


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
    feature_path: Optional[str],
    selected_sheets: Sequence[str],
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
    options: Optional[ReportBuildOptions] = None,
) -> ModelReportResult:
    """Build the full report result object."""
    build_start = time.perf_counter()
    resolved_options = options or ReportBuildOptions()
    context = ReportBuildContext(
        data=data,
        tag_col=tag_col,
        month_col=month_col,
        options=resolved_options,
    )
    LOGGER.debug(
        "build_report_result started | rows=%d cols=%d selected_sheet_keys=%s "
        "engine=%s workers=%d parallel_sheets=%s memory_mode=%s",
        len(data),
        len(data.columns),
        list(selected_sheets),
        resolved_options.engine,
        resolved_options.max_workers,
        resolved_options.parallel_sheets,
        resolved_options.memory_mode,
    )

    optional_sheet_availability = sheet_registry.resolve_optional_sheet_availability(
        data=data,
        tag_col=tag_col,
        dim_list=dim_list,
        score_list=score_list,
        var_list=var_list,
        model_adapter=model_adapter,
    )
    output_sheet_keys = sheet_registry.filter_output_sheet_keys(
        requested_keys=selected_sheets,
        availability=optional_sheet_availability,
    )
    if not output_sheet_keys:
        raise ValueError("No available sheets for the current report configuration.")
    build_sheet_keys = sheet_registry.resolve_build_keys(
        output_keys=output_sheet_keys,
        availability=optional_sheet_availability,
    )

    step_start = time.perf_counter()
    score_metric_options = group_metrics._build_score_metric_options(
        score_direction_summary
    )
    _log_context_stage(
        context,
        "build_score_metric_options",
        time.perf_counter() - step_start,
        extra=f"score_count={len(score_metric_options)}",
    )

    primary_label = label_list[0]
    primary_report_score = report_score_columns[primary_score_name]

    shared_tag_metrics = pd.DataFrame()
    shared_month_metrics = pd.DataFrame()
    if any(key in build_sheet_keys for key in ["model_design", "model_performance"]):
        step_start = time.perf_counter()
        (
            shared_tag_metrics,
            shared_month_metrics,
        ) = group_metrics._build_split_metrics_tables(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            raw_date_col=raw_date_col,
            label_list=label_list,
            score_col=primary_report_score,
            model_name=primary_score_name,
            reverse_auc_label=group_metrics._lookup_reverse_auc(
                score_metric_options, primary_score_name
            ),
            metrics_mode=resolved_options.metrics_mode,
            prin_bal_amount_col=prin_bal_amount_col,
            loan_amount_col=loan_amount_col,
            build_context=context,
        )
        _log_context_stage(
            context,
            "precompute_split_metrics",
            time.perf_counter() - step_start,
            extra=(
                f"tag_rows={len(shared_tag_metrics)} "
                f"month_rows={len(shared_month_metrics)}"
            ),
        )

    feature_dict = pd.DataFrame()
    if any(key in build_sheet_keys for key in ["variable_analysis", "portrait"]):
        step_start = time.perf_counter()
        feature_dict = feature_metrics._load_feature_dictionary(feature_path)
        _log_context_stage(
            context,
            "load_feature_dictionary",
            time.perf_counter() - step_start,
            extra=f"rows={len(feature_dict)}",
        )

    feature_cols: List[str] = []
    if "variable_analysis" in build_sheet_keys:
        step_start = time.perf_counter()
        feature_cols = feature_metrics._determine_feature_columns(
            data,
            model_features=model_adapter.get_feature_names(),
            excluded=[
                tag_col,
                month_col,
                *label_list,
                primary_score_name,
                *score_list,
                *report_score_columns.values(),
                *dim_list,
                *var_list,
            ],
        )
        _log_context_stage(
            context,
            "determine_feature_columns",
            time.perf_counter() - step_start,
            extra=f"feature_count={len(feature_cols)}",
        )

    score_edges: List[float] = []
    if "model_performance" in build_sheet_keys:
        step_start = time.perf_counter()
        score_edges = build_reference_quantile_bins(
            data.loc[data[tag_col] == "train", primary_report_score],
            bins=10,
        )
        _log_context_stage(
            context,
            "build_reference_quantile_bins",
            time.perf_counter() - step_start,
            extra=f"edge_count={len(score_edges)}",
        )

    score_model_columns: List[Tuple[str, str]] = []
    if any(
        key in build_sheet_keys
        for key in ["dimensional_comparison", "model_comparison", "portrait"]
    ):
        score_model_columns = group_metrics._resolve_score_model_columns(
            primary_score_name=primary_score_name,
            score_list=score_list,
            report_score_columns=report_score_columns,
        )

    shared_oot_frame = (
        data.loc[data[tag_col] == "oot"]
        if any(
            key in build_sheet_keys
            for key in ["dimensional_comparison", "model_comparison", "portrait"]
        )
        else pd.DataFrame()
    )
    shared_primary_binary_data = (
        data.loc[data[primary_label].isin([0, 1])]
        if "model_performance" in build_sheet_keys
        else pd.DataFrame()
    )

    child_builders = {
        "model_design": lambda: main_sheets.build_model_design_sheet(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            raw_date_col=raw_date_col,
            primary_label=primary_label,
            label_list=label_list,
            score_col=primary_report_score,
            model_name=primary_score_name,
            reverse_auc_label=group_metrics._lookup_reverse_auc(
                score_metric_options, primary_score_name
            ),
            metrics_mode=resolved_options.metrics_mode,
            precomputed_tag_metrics=shared_tag_metrics,
            build_context=context,
        ),
        "variable_analysis": lambda: main_sheets.build_variable_analysis_sheet(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            primary_label=primary_label,
            feature_cols=feature_cols,
            feature_dict=feature_dict,
            model_adapter=model_adapter,
            build_context=context,
        ),
        "scorecard_details": lambda: main_sheets.build_scorecard_details_sheet(
            model_adapter=model_adapter,
        ),
        "model_performance": lambda: main_sheets.build_model_performance_sheet(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            raw_date_col=raw_date_col,
            label_list=label_list,
            primary_label=primary_label,
            score_col=primary_report_score,
            model_name=primary_score_name,
            model_adapter=model_adapter,
            score_edges=score_edges,
            reverse_auc_label=group_metrics._lookup_reverse_auc(
                score_metric_options, primary_score_name
            ),
            metrics_mode=resolved_options.metrics_mode,
            precomputed_tag_metrics=shared_tag_metrics,
            precomputed_month_metrics=shared_month_metrics,
            primary_binary_data=shared_primary_binary_data,
            prin_bal_amount_col=prin_bal_amount_col,
            loan_amount_col=loan_amount_col,
            build_context=context,
        ),
        "dimensional_comparison": (
            lambda: appendix_sheets.build_dimensional_comparison_sheet(
                data=data,
                tag_col=tag_col,
                dim_list=dim_list,
                label_list=label_list,
                score_model_columns=score_model_columns,
                score_metric_options=score_metric_options,
                oot_frame=shared_oot_frame,
                prin_bal_amount_col=prin_bal_amount_col,
                loan_amount_col=loan_amount_col,
                build_context=context,
            )
        ),
        "model_comparison": lambda: appendix_sheets.build_model_comparison_sheet(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            raw_date_col=raw_date_col,
            label_list=label_list,
            model_columns=score_model_columns,
            score_metric_options=score_metric_options,
            oot_frame=shared_oot_frame,
            prin_bal_amount_col=prin_bal_amount_col,
            loan_amount_col=loan_amount_col,
            build_context=context,
        ),
        "portrait": lambda: appendix_sheets.build_portrait_sheet(
            data=data,
            tag_col=tag_col,
            var_list=var_list,
            score_model_columns=score_model_columns,
            feature_dict=feature_dict,
            oot_frame=shared_oot_frame,
        ),
    }

    child_sheet_keys = [
        key for key in build_sheet_keys if key != "overview" and key in child_builders
    ]
    built_sheets_by_key: Dict[str, ReportSheet] = {}
    if (
        resolved_options.parallel_sheets
        and len(child_sheet_keys) > 1
        and resolved_options.max_workers > 1
    ):
        worker_count = min(resolved_options.max_workers, len(child_sheet_keys))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                sheet_key: executor.submit(
                    _timed_sheet_build, child_builders[sheet_key]
                )
                for sheet_key in child_sheet_keys
            }
            for sheet_key in child_sheet_keys:
                sheet_obj, elapsed = futures[sheet_key].result()
                built_sheets_by_key[sheet_key] = sheet_obj
                _log_context_stage(
                    context,
                    f"sheet:{sheet_key}",
                    elapsed,
                    extra=f"blocks={len(sheet_obj.blocks)}",
                )
    else:
        for sheet_key in child_sheet_keys:
            sheet_obj, elapsed = _timed_sheet_build(child_builders[sheet_key])
            built_sheets_by_key[sheet_key] = sheet_obj
            _log_context_stage(
                context,
                f"sheet:{sheet_key}",
                elapsed,
                extra=f"blocks={len(sheet_obj.blocks)}",
            )

    if "overview" in build_sheet_keys:
        overview_sheet, elapsed = _timed_sheet_build(
            lambda: main_sheets.build_overview_sheet(
                score_direction_summary=score_direction_summary,
                model_performance_sheet=built_sheets_by_key.get("model_performance"),
                dimensional_sheet=built_sheets_by_key.get("dimensional_comparison"),
                comparison_sheet=built_sheets_by_key.get("model_comparison"),
                portrait_sheet=built_sheets_by_key.get("portrait"),
            )
        )
        built_sheets_by_key["overview"] = overview_sheet
        _log_context_stage(
            context,
            "sheet:overview",
            elapsed,
            extra=f"blocks={len(overview_sheet.blocks)}",
        )

    ordered_output_keys = [
        key for key in sheet_registry.SHEET_KEY_ORDER if key in output_sheet_keys
    ]
    output_sheet_names = sheet_registry.resolve_output_sheet_names(ordered_output_keys)

    sheets: Dict[str, ReportSheet] = {}
    for sheet_key in ordered_output_keys:
        if sheet_key not in built_sheets_by_key:
            continue
        display_name = output_sheet_names[sheet_key]
        sheet_obj = built_sheets_by_key[sheet_key]
        renamed = ReportSheet(name=display_name, blocks=sheet_obj.blocks)
        sheets[display_name] = renamed

    total_elapsed = time.perf_counter() - build_start
    LOGGER.debug(
        "build_report_result completed | elapsed=%.3fs sheet_count=%d",
        total_elapsed,
        len(sheets),
    )
    _log_top_context_timings(context)

    return ModelReportResult(
        sheets=sheets,
        metadata={
            "score_col": primary_score_name,
            "label_list": list(label_list),
            "feature_columns": feature_cols,
            "score_directions": score_direction_summary.to_dict("records"),
            "report_score_columns": dict(report_score_columns),
            "sheet_output_names": output_sheet_names,
            "report_compute_options": {
                "engine": resolved_options.engine,
                "max_workers": resolved_options.max_workers,
                "parallel_sheets": resolved_options.parallel_sheets,
                "memory_mode": resolved_options.memory_mode,
                "metrics_mode": resolved_options.metrics_mode,
            },
            "report_compute_top_timings": [
                {"step": name, "elapsed": elapsed}
                for name, elapsed in sorted(
                    context.timings, key=lambda item: item[1], reverse=True
                )[:5]
            ],
        },
    )
