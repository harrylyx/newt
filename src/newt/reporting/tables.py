"""Table builders for the Excel model report."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from newt.config import BINNING
from newt.features.analysis.batch_iv import calculate_batch_iv
from newt.metrics.binary_metrics import (
    calculate_binary_metrics as _unified_binary_metrics,
)
from newt.metrics.binary_metrics import (
    calculate_binary_metrics_batch as _unified_binary_metrics_batch,
)
from newt.metrics.psi import calculate_feature_psi_pairs_batch, calculate_psi_batch
from newt.metrics.reporting import (
    build_reference_quantile_bins,
    calculate_bin_performance_table,
    calculate_portrait_means_by_score_bin,
    calculate_score_correlation_matrix,
    summarize_label_distribution,
)
from newt.report_sort_utils import month_sort_key as _shared_month_sort_key
from newt.report_sort_utils import ordered_month_values as _shared_ordered_month_values
from newt.report_sort_utils import ordered_tag_values as _shared_ordered_tag_values
from newt.report_sort_utils import sort_report_frame as _shared_sort_report_frame
from newt.report_sort_utils import tag_sort_key as _shared_tag_sort_key
from newt.reporting.model_adapter import ModelAdapter
from newt.reporting.table_context import (
    FeatureComputationArtifacts,
    ReportBuildContext,
    ReportBuildOptions,
)
from newt.results import ModelReportResult, ReportBlock, ReportChart, ReportSheet

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


MAIN_SHEET_KEY_ORDER = [
    "overview",
    "model_design",
    "variable_analysis",
    "model_performance",
]
APPENDIX_SHEET_KEY_ORDER = [
    "dimensional_comparison",
    "model_comparison",
    "portrait",
]
SHEET_KEY_ORDER = [*MAIN_SHEET_KEY_ORDER, *APPENDIX_SHEET_KEY_ORDER]

SHEET_INDEX_SELECTOR_MAP = {
    1: "overview",
    2: "model_design",
    3: "variable_analysis",
    4: "model_performance",
}

SHEET_NAME_SELECTOR_MAP = {
    "总览": "overview",
    "模型设计": "model_design",
    "变量分析": "variable_analysis",
    "模型表现": "model_performance",
    "分维度对比": "dimensional_comparison",
    "新老模型对比": "model_comparison",
    "画像变量": "portrait",
}

MAIN_SHEET_OUTPUT_NAME_MAP = {
    "overview": "总览",
    "model_design": "1.模型设计",
    "variable_analysis": "2.变量分析",
    "model_performance": "3.模型表现",
}

APPENDIX_SHEET_LABEL_MAP = {
    "dimensional_comparison": "分维度对比",
    "model_comparison": "新老模型对比",
    "portrait": "画像变量",
}


def resolve_sheet_keys(sheet_list: Optional[Sequence[object]]) -> List[str]:
    """Resolve user sheet selectors into logical sheet keys."""
    if not sheet_list:
        return list(SHEET_KEY_ORDER)

    resolved: List[str] = []
    for item in sheet_list:
        if isinstance(item, int):
            if item not in SHEET_INDEX_SELECTOR_MAP:
                raise ValueError(f"Unknown sheet index: {item}")
            sheet_key = SHEET_INDEX_SELECTOR_MAP[item]
        else:
            sheet_name = str(item)
            if sheet_name not in SHEET_NAME_SELECTOR_MAP:
                raise ValueError(f"Unknown sheet name: {sheet_name}")
            sheet_key = SHEET_NAME_SELECTOR_MAP[sheet_name]
        if sheet_key not in resolved:
            resolved.append(sheet_key)
    return resolved


def resolve_sheet_names(sheet_list: Optional[Sequence[object]]) -> List[str]:
    """Backward-compatible alias for logical sheet key resolution."""
    return resolve_sheet_keys(sheet_list)


def _resolve_optional_sheet_availability(
    data: pd.DataFrame,
    tag_col: str,
    dim_list: Sequence[str],
    score_list: Sequence[str],
    var_list: Sequence[str],
) -> Dict[str, bool]:
    has_oot = bool((data[tag_col] == "oot").any())
    return {
        "dimensional_comparison": bool(dim_list) and has_oot,
        "model_comparison": bool(score_list),
        "portrait": bool(var_list) and has_oot,
    }


def _filter_output_sheet_keys(
    requested_keys: Sequence[str],
    availability: Dict[str, bool],
) -> List[str]:
    output: List[str] = []
    for key in requested_keys:
        if key in availability and not availability[key]:
            continue
        if key not in output:
            output.append(key)
    return output


def _resolve_build_keys(
    output_keys: Sequence[str],
    availability: Dict[str, bool],
) -> List[str]:
    build_keys = set(output_keys)
    if "overview" in output_keys:
        build_keys.add("model_performance")
        for appendix_key in APPENDIX_SHEET_KEY_ORDER:
            if availability.get(appendix_key, False):
                build_keys.add(appendix_key)
    return [key for key in SHEET_KEY_ORDER if key in build_keys]


def _resolve_output_sheet_names(output_keys: Sequence[str]) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for main_key in MAIN_SHEET_KEY_ORDER:
        if main_key in output_keys:
            names[main_key] = MAIN_SHEET_OUTPUT_NAME_MAP[main_key]

    appendix_present = [
        key for key in APPENDIX_SHEET_KEY_ORDER if key in set(output_keys)
    ]
    for index, appendix_key in enumerate(appendix_present, start=1):
        names[appendix_key] = f"附{index} {APPENDIX_SHEET_LABEL_MAP[appendix_key]}"
    return names


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

    optional_sheet_availability = _resolve_optional_sheet_availability(
        data=data,
        tag_col=tag_col,
        dim_list=dim_list,
        score_list=score_list,
        var_list=var_list,
    )
    output_sheet_keys = _filter_output_sheet_keys(
        requested_keys=selected_sheets,
        availability=optional_sheet_availability,
    )
    if not output_sheet_keys:
        raise ValueError("No available sheets for the current report configuration.")
    build_sheet_keys = _resolve_build_keys(
        output_keys=output_sheet_keys,
        availability=optional_sheet_availability,
    )

    step_start = time.perf_counter()
    score_metric_options = _build_score_metric_options(score_direction_summary)
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
        shared_tag_metrics, shared_month_metrics = _build_split_metrics_tables(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            raw_date_col=raw_date_col,
            label_list=label_list,
            score_col=primary_report_score,
            model_name=primary_score_name,
            reverse_auc_label=_lookup_reverse_auc(
                score_metric_options, primary_score_name
            ),
            metrics_mode=resolved_options.metrics_mode,
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
        feature_dict = _load_feature_dictionary(feature_path)
        _log_context_stage(
            context,
            "load_feature_dictionary",
            time.perf_counter() - step_start,
            extra=f"rows={len(feature_dict)}",
        )

    feature_cols: List[str] = []
    if "variable_analysis" in build_sheet_keys:
        step_start = time.perf_counter()
        feature_cols = _determine_feature_columns(
            data,
            model_adapter,
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
        score_model_columns = _resolve_score_model_columns(
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
        "model_design": lambda: build_model_design_sheet(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            raw_date_col=raw_date_col,
            primary_label=primary_label,
            label_list=label_list,
            score_col=primary_report_score,
            model_name=primary_score_name,
            reverse_auc_label=_lookup_reverse_auc(
                score_metric_options, primary_score_name
            ),
            metrics_mode=resolved_options.metrics_mode,
            precomputed_tag_metrics=shared_tag_metrics,
            build_context=context,
        ),
        "variable_analysis": lambda: build_variable_analysis_sheet(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            primary_label=primary_label,
            feature_cols=feature_cols,
            feature_dict=feature_dict,
            model_adapter=model_adapter,
            build_context=context,
        ),
        "model_performance": lambda: build_model_performance_sheet(
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
            reverse_auc_label=_lookup_reverse_auc(
                score_metric_options, primary_score_name
            ),
            metrics_mode=resolved_options.metrics_mode,
            precomputed_tag_metrics=shared_tag_metrics,
            precomputed_month_metrics=shared_month_metrics,
            primary_binary_data=shared_primary_binary_data,
            build_context=context,
        ),
        "dimensional_comparison": lambda: build_dimensional_comparison_sheet(
            data=data,
            tag_col=tag_col,
            dim_list=dim_list,
            label_list=label_list,
            score_model_columns=score_model_columns,
            score_metric_options=score_metric_options,
            oot_frame=shared_oot_frame,
            build_context=context,
        ),
        "model_comparison": lambda: build_model_comparison_sheet(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            raw_date_col=raw_date_col,
            label_list=label_list,
            model_columns=score_model_columns,
            score_metric_options=score_metric_options,
            oot_frame=shared_oot_frame,
            build_context=context,
        ),
        "portrait": lambda: build_portrait_sheet(
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
            lambda: build_overview_sheet(
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

    ordered_output_keys = [key for key in SHEET_KEY_ORDER if key in output_sheet_keys]
    output_sheet_names = _resolve_output_sheet_names(ordered_output_keys)

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


def build_overview_sheet(
    score_direction_summary: pd.DataFrame,
    model_performance_sheet: Optional[ReportSheet] = None,
    dimensional_sheet: Optional[ReportSheet] = None,
    comparison_sheet: Optional[ReportSheet] = None,
    portrait_sheet: Optional[ReportSheet] = None,
) -> ReportSheet:
    """Build overview sheet from prebuilt child sheets."""
    section_mapping = {
        1: "一",
        2: "二",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }
    current_section = 1

    blocks = [
        ReportBlock(
            title=f"{section_mapping[current_section]}、目标与设计方案", blank_rows_after=1
        )
    ]
    blocks.append(
        ReportBlock(
            title="表格1",
            data=pd.DataFrame(columns=["迭代目标", "问题", "原因", "迭代方向"]),
        )
    )
    current_section += 1

    blocks.append(
        ReportBlock(
            title=f"{section_mapping[current_section]}、迭代效果", blank_rows_after=1
        )
    )

    if not score_direction_summary.empty:
        direction_columns = [
            column
            for column in [
                "分数字段",
                "原始方向",
                "报表计算方向",
                "判断依据",
                "原始AUC",
            ]
            if column in score_direction_summary.columns
        ]
        blocks.append(
            ReportBlock(
                title="分数字段方向说明",
                data=score_direction_summary.loc[:, direction_columns].copy(),
            )
        )

    tag_metrics = _extract_block_data(model_performance_sheet, "二、按tag模型效果")
    if not tag_metrics.empty:
        blocks.append(ReportBlock(title="按tag模型效果", data=tag_metrics))

    month_metrics = _extract_block_data(model_performance_sheet, "三、按月模型效果")
    if not month_metrics.empty:
        blocks.append(ReportBlock(title="按月模型效果", data=month_metrics))

    dimensional_blocks = _extract_numbered_data_blocks(dimensional_sheet)
    if dimensional_blocks:
        current_section += 1
        blocks.append(
            ReportBlock(
                title=f"{section_mapping[current_section]}、分维度对比", blank_rows_after=1
            )
        )
        blocks.extend(dimensional_blocks)

    if comparison_sheet is not None:
        comparison_blocks = []
        for block in comparison_sheet.blocks:
            if (
                block.title.startswith("按tag新老模型对比(")
                or block.title.startswith("按月新老模型对比(")
                or block.title == "OOT相关性矩阵"
            ):
                comparison_blocks.append(
                    ReportBlock(title=block.title, data=block.data)
                )

        if comparison_blocks:
            current_section += 1
            blocks.append(
                ReportBlock(
                    title=f"{section_mapping[current_section]}、新老模型对比",
                    blank_rows_after=1,
                )
            )
            blocks.extend(comparison_blocks)

    portrait_blocks = _extract_numbered_data_blocks(portrait_sheet)
    if portrait_blocks:
        current_section += 1
        blocks.append(
            ReportBlock(
                title=f"{section_mapping[current_section]}、画像变量", blank_rows_after=1
            )
        )
        blocks.extend(portrait_blocks)

    return ReportSheet(name="总览", blocks=blocks)


def _extract_block_data(sheet: Optional[ReportSheet], title: str) -> pd.DataFrame:
    if sheet is None:
        return pd.DataFrame()
    try:
        block = sheet.get_block(title)
    except KeyError:
        return pd.DataFrame()
    return block.data


def _extract_numbered_data_blocks(sheet: Optional[ReportSheet]) -> List[ReportBlock]:
    if sheet is None:
        return []
    blocks: List[ReportBlock] = []
    for block in sheet.blocks:
        if not block.title or block.data.empty:
            continue
        prefix = block.title.split(".", 1)[0]
        if not prefix.isdigit():
            continue
        blocks.append(
            ReportBlock(
                title=block.title,
                title_right=block.title_right,
                data=block.data,
            )
        )
    return blocks


def build_dimensional_comparison_sheet(
    data: pd.DataFrame,
    tag_col: str,
    dim_list: Sequence[str],
    label_list: Sequence[str],
    score_model_columns: Sequence[Tuple[str, str]],
    score_metric_options: Dict[str, Dict[str, bool]],
    oot_frame: Optional[pd.DataFrame] = None,
    build_context: Optional[ReportBuildContext] = None,
) -> ReportSheet:
    """Build appendix sheet for dimensional model-effect comparison."""
    oot_frame = oot_frame if oot_frame is not None else data.loc[data[tag_col] == "oot"]
    dim_table = (
        _build_dimensional_comparison(
            data=oot_frame,
            dim_list=dim_list,
            label_list=label_list,
            score_model_columns=score_model_columns,
            score_metric_options=score_metric_options,
            metrics_mode=(
                build_context.options.metrics_mode
                if build_context is not None
                else "exact"
            ),
        )
        if dim_list and not oot_frame.empty
        else pd.DataFrame()
    )
    blocks = [ReportBlock(title="一、分维度对比", blank_rows_after=1)]
    if dim_table.empty:
        return ReportSheet(name="分维度对比", blocks=blocks)

    for index, dim_col in enumerate(dim_list, start=1):
        dim_rows = dim_table.loc[dim_table["维度列"] == dim_col]
        if dim_rows.empty:
            continue
        blocks.append(
            ReportBlock(
                title=f"{index}.{dim_col}",
                data=dim_rows.drop(columns=["维度列"], errors="ignore").reset_index(
                    drop=True
                ),
            )
        )
    return ReportSheet(
        name="分维度对比",
        blocks=blocks,
    )


def build_model_comparison_sheet(
    data: pd.DataFrame,
    tag_col: str,
    month_col: str,
    raw_date_col: str,
    label_list: Sequence[str],
    model_columns: Sequence[Tuple[str, str]],
    score_metric_options: Dict[str, Dict[str, bool]],
    oot_frame: Optional[pd.DataFrame] = None,
    build_context: Optional[ReportBuildContext] = None,
) -> ReportSheet:
    """Build appendix sheet for old/new model comparison."""
    blocks = [ReportBlock(title="一、新老模型对比", blank_rows_after=1)]
    if len(model_columns) >= 2:
        for old_model_name, old_score_col in model_columns[1:]:
            pair_models = [model_columns[0], (old_model_name, old_score_col)]
            tag_compare = _build_model_pair_comparison(
                data=data,
                group_mode="tag",
                label_list=label_list,
                model_columns=pair_models,
                tag_col=tag_col,
                month_col=month_col,
                raw_date_col=raw_date_col,
                score_metric_options=score_metric_options,
                metrics_mode=(
                    build_context.options.metrics_mode
                    if build_context is not None
                    else "exact"
                ),
                build_context=build_context,
            )
            month_compare = _build_model_pair_comparison(
                data=data,
                group_mode="month",
                label_list=label_list,
                model_columns=pair_models,
                tag_col=tag_col,
                month_col=month_col,
                raw_date_col=raw_date_col,
                score_metric_options=score_metric_options,
                metrics_mode=(
                    build_context.options.metrics_mode
                    if build_context is not None
                    else "exact"
                ),
                build_context=build_context,
            )
            blocks.append(
                ReportBlock(
                    title=f"按tag新老模型对比({old_model_name})",
                    data=tag_compare,
                )
            )
            blocks.append(
                ReportBlock(
                    title=f"按月新老模型对比({old_model_name})",
                    data=month_compare,
                )
            )

    oot_frame = oot_frame if oot_frame is not None else data.loc[data[tag_col] == "oot"]
    if not oot_frame.empty and model_columns:
        display_by_column = {column: model for model, column in model_columns}
        corr = calculate_score_correlation_matrix(
            oot_frame,
            [column for _, column in model_columns],
        )
        corr = corr.round(4)
        corr = corr.rename(index=display_by_column, columns=display_by_column)
        corr.index.name = "模型"
        blocks.append(ReportBlock(title="OOT相关性矩阵", data=corr.reset_index()))
    return ReportSheet(name="新老模型对比", blocks=blocks)


def build_portrait_sheet(
    data: pd.DataFrame,
    tag_col: str,
    var_list: Sequence[str],
    score_model_columns: Sequence[Tuple[str, str]],
    feature_dict: Optional[pd.DataFrame] = None,
    oot_frame: Optional[pd.DataFrame] = None,
) -> ReportSheet:
    """Build appendix sheet for OOT portrait variable means."""
    oot_frame = oot_frame if oot_frame is not None else data.loc[data[tag_col] == "oot"]
    portrait = pd.DataFrame()
    if var_list and not oot_frame.empty and score_model_columns:
        display_by_column = {column: model for model, column in score_model_columns}
        portrait = calculate_portrait_means_by_score_bin(
            data=oot_frame,
            score_cols=[column for _, column in score_model_columns],
            variable_cols=var_list,
        )
        portrait["模型"] = portrait["模型"].map(display_by_column).fillna(portrait["模型"])
        portrait = _reshape_portrait_table(portrait)
    feature_dict = feature_dict if feature_dict is not None else pd.DataFrame()
    blocks = [ReportBlock(title="一、画像变量均值对比", blank_rows_after=1)]
    target_columns = ["模型", *[str(index) for index in range(1, 11)], "Missing"]
    for index, variable_name in enumerate(var_list, start=1):
        variable_table = portrait.loc[
            portrait["画像变量"] == variable_name,
            target_columns,
        ].copy()
        meta = _lookup_feature_meta(feature_dict, str(variable_name))
        display_name = meta.get("中文名", "")
        title_prefix = f"{index}.{variable_name}"
        full_title = f"{title_prefix} {display_name}" if display_name else title_prefix
        blocks.append(
            ReportBlock(
                title=full_title,
                data=variable_table.reset_index(drop=True),
            )
        )
    return ReportSheet(name="画像变量", blocks=blocks)


def build_model_design_sheet(
    data: pd.DataFrame,
    tag_col: str,
    month_col: str,
    raw_date_col: str,
    primary_label: str,
    label_list: Sequence[str],
    score_col: str,
    model_name: str,
    reverse_auc_label: bool = False,
    metrics_mode: str = "exact",
    precomputed_tag_metrics: Optional[pd.DataFrame] = None,
    build_context: Optional[ReportBuildContext] = None,
) -> ReportSheet:
    """Build sheet 2."""
    blocks = [
        ReportBlock(title="一、模型设计方案", blank_rows_after=2),
        ReportBlock(title="二、样本和Y定义", blank_rows_after=2),
        ReportBlock(
            title="原始样本分布表",
            data=summarize_label_distribution(
                data=data.loc[data[tag_col].notna()],
                label_col=primary_label,
                tag_col=tag_col,
                month_col=month_col,
                include_blank_channel=True,
                include_tag=False,
            ),
        ),
        ReportBlock(title="样本筛选条件", blank_rows_after=2),
        ReportBlock(
            title="开发样本分布表",
            data=summarize_label_distribution(
                data=data.loc[data[tag_col].notna()],
                label_col=primary_label,
                tag_col=tag_col,
                month_col=month_col,
                include_blank_channel=True,
                include_tag=False,
            ),
        ),
    ]

    sample_rows = []
    for label_col in label_list:
        for tag_value in _ordered_tag_values(data[tag_col]):
            tag_frame = data.loc[data[tag_col] == tag_value]
            metrics = _calculate_report_metrics(
                tag_frame[label_col],
                tag_frame[score_col],
                reverse_auc_label=reverse_auc_label,
                metrics_mode=metrics_mode,
            )
            sample_rows.append(
                {
                    "样本集": tag_value,
                    "样本标签": label_col,
                    "好样本": metrics["好"],
                    "坏样本": metrics["坏"],
                    "总量": metrics["总"],
                    "坏占比": metrics["坏占比"],
                    "备注": "",
                }
            )
    blocks.append(ReportBlock(title="建模样本分布情况表", data=pd.DataFrame(sample_rows)))

    effect_rows = []
    if precomputed_tag_metrics is not None and not precomputed_tag_metrics.empty:
        effect_source = precomputed_tag_metrics.loc[
            precomputed_tag_metrics["样本标签"].isin(label_list)
        ]
        for _, row in effect_source.iterrows():
            effect_rows.append(
                {
                    "样本集": row["样本集"],
                    "样本标签": row["样本标签"],
                    "AUC": row["AUC"],
                    "KS": row["KS"],
                }
            )
    else:
        for label_col in label_list:
            effect_table = _build_group_metrics(
                data=data,
                group_cols=[tag_col],
                label_col=label_col,
                score_col=score_col,
                tag_col=tag_col,
                month_col=month_col,
                raw_date_col=raw_date_col,
                model_name=model_name,
                reverse_auc_label=reverse_auc_label,
                metrics_mode=metrics_mode,
                build_context=build_context,
            )
            for _, row in effect_table.iterrows():
                effect_rows.append(
                    {
                        "样本集": row["样本集"],
                        "样本标签": row["样本标签"],
                        "AUC": row["AUC"],
                        "KS": row["KS"],
                    }
                )
    blocks.append(ReportBlock(title="模型效果汇总", data=pd.DataFrame(effect_rows)))
    return ReportSheet(name="模型设计", blocks=blocks)


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
    """Build sheet 3."""
    sheet_start = time.perf_counter()

    step_start = time.perf_counter()
    importance = model_adapter.get_importance_table()
    LOGGER.debug(
        "build_variable_analysis_sheet step finished | step=get_importance_table "
        "elapsed=%.3fs rows=%d",
        time.perf_counter() - step_start,
        len(importance),
    )

    step_start = time.perf_counter()
    train_frame = data.loc[
        (data[tag_col] == "train") & data[primary_label].isin([0, 1])
    ]
    oot_frame = data.loc[(data[tag_col] == "oot") & data[primary_label].isin([0, 1])]
    LOGGER.debug(
        "build_variable_analysis_sheet step finished | step=prepare_train_oot "
        "elapsed=%.3fs train_rows=%d oot_rows=%d",
        time.perf_counter() - step_start,
        len(train_frame),
        len(oot_frame),
    )

    step_start = time.perf_counter()
    feature_table, feature_artifacts = _build_feature_analysis_table(
        train_frame=train_frame,
        oot_frame=oot_frame,
        month_frame=data,
        tag_col=tag_col,
        month_col=month_col,
        label_col=primary_label,
        feature_cols=feature_cols,
        feature_dict=feature_dict,
        importance=importance,
        build_context=build_context,
    )
    LOGGER.debug(
        "build_variable_analysis_sheet step finished | "
        "step=build_feature_analysis_table elapsed=%.3fs feature_rows=%d",
        time.perf_counter() - step_start,
        len(feature_table),
    )
    top_features = (
        feature_table["vars"].head(30).tolist() if not feature_table.empty else []
    )

    summary_table, type_table = _build_feature_selection_summary(
        feature_table, feature_dict
    )
    blocks = [
        ReportBlock(title="一、变量筛选", data=summary_table),
    ]
    if not type_table.empty:
        blocks.append(ReportBlock(title="变量类型分布", data=type_table))

    blocks.append(ReportBlock(title="二、变量分析", data=feature_table))
    blocks.append(ReportBlock(title="三、单变量分析", blank_rows_after=1))

    feature_jobs = [
        (rank, str(feature)) for rank, feature in enumerate(top_features, start=1)
    ]
    use_parallel_features = (
        build_context is not None
        and build_context.options.max_workers > 1
        and len(feature_jobs) > 1
    )

    def _run_feature_job(
        job: Tuple[int, str]
    ) -> Tuple[int, str, str, str, str, pd.DataFrame, pd.DataFrame, float]:
        rank, feature_name = job
        feature_start = time.perf_counter()
        edges = feature_artifacts.edges_by_feature.get(feature_name)
        if edges is None:
            edges = build_reference_quantile_bins(train_frame[feature_name], bins=10)
        oot_bins = feature_artifacts.oot_bin_stats_by_feature.get(feature_name)
        if oot_bins is None:
            oot_bins = _build_feature_bin_stats(
                frame=oot_frame,
                feature=feature_name,
                label_col=primary_label,
                edges=edges,
            )
        monthly_table = _build_feature_monthly_metrics(
            all_data=data,
            train_frame=train_frame,
            feature=feature_name,
            label_col=primary_label,
            month_col=month_col,
            edges=edges,
            engine=build_context.options.engine if build_context else "python",
            metrics_mode=(
                build_context.options.metrics_mode
                if build_context is not None
                else "exact"
            ),
        )
        display_name = _lookup_feature_meta(feature_dict, feature_name).get("中文名", "")
        title_prefix = f"{rank}.{feature_name}"
        # Merge English and Chinese names in the main title to ensure
        # consistent font styling
        full_title = f"{title_prefix} {display_name}" if display_name else title_prefix
        # Use Chinese name for chart title if available, otherwise feature name
        chart_title = str(display_name or feature_name)
        return (
            rank,
            feature_name,
            full_title,
            chart_title,
            oot_bins,
            monthly_table,
            time.perf_counter() - feature_start,
        )

    feature_results: List[
        Tuple[int, str, str, str, pd.DataFrame, pd.DataFrame, float]
    ] = []
    if use_parallel_features:
        worker_cap = min(build_context.options.max_workers, len(feature_jobs))
        if build_context.options.memory_mode == "compact":
            worker_cap = min(worker_cap, 4)
        with ThreadPoolExecutor(max_workers=max(1, worker_cap)) as executor:
            for result in executor.map(_run_feature_job, feature_jobs):
                feature_results.append(result)
    else:
        for job in feature_jobs:
            feature_results.append(_run_feature_job(job))

    feature_timings: List[Tuple[str, float]] = []
    for (
        _,
        feature_name,
        full_title,
        chart_title,
        oot_bins,
        monthly_table,
        elapsed,
    ) in feature_results:
        blocks.append(
            ReportBlock(
                title=full_title,
                blank_rows_after=0,
            )
        )
        blocks.append(ReportBlock(title=f"{full_title} 分箱表", data=oot_bins))
        blocks.append(ReportBlock(title=f"{full_title} 按月效果", data=monthly_table))
        if not oot_bins.empty:
            blocks.append(
                ReportBlock(
                    title="",
                    chart=ReportChart(
                        chart_type="combo",
                        category_column="bin",
                        value_columns=["total_prop"],
                        secondary_value_columns=["bad_rate"],
                        title=chart_title,
                        source_block_title=f"{full_title} 分箱表",
                    ),
                    blank_rows_after=2,
                )
            )
        feature_timings.append((feature_name, elapsed))
        LOGGER.debug(
            "build_variable_analysis_sheet feature finished | feature=%s "
            "elapsed=%.3fs oot_bins=%d monthly_rows=%d",
            feature_name,
            elapsed,
            len(oot_bins),
            len(monthly_table),
        )

    if feature_timings:
        top_feature_cost = ", ".join(
            f"{name}:{elapsed:.3f}s"
            for name, elapsed in sorted(
                feature_timings, key=lambda item: item[1], reverse=True
            )[:5]
        )
        LOGGER.debug(
            "build_variable_analysis_sheet slowest_features | %s",
            top_feature_cost,
        )

    LOGGER.debug(
        "build_variable_analysis_sheet completed | elapsed=%.3fs "
        "top_features=%d blocks=%d",
        time.perf_counter() - sheet_start,
        len(top_features),
        len(blocks),
    )

    return ReportSheet(name="变量分析", blocks=blocks)


def build_model_performance_sheet(
    data: pd.DataFrame,
    tag_col: str,
    month_col: str,
    raw_date_col: str,
    label_list: Sequence[str],
    primary_label: str,
    score_col: str,
    model_name: str,
    model_adapter: ModelAdapter,
    score_edges: Sequence[float],
    reverse_auc_label: bool = False,
    metrics_mode: str = "exact",
    precomputed_tag_metrics: Optional[pd.DataFrame] = None,
    precomputed_month_metrics: Optional[pd.DataFrame] = None,
    primary_binary_data: Optional[pd.DataFrame] = None,
    build_context: Optional[ReportBuildContext] = None,
) -> ReportSheet:
    """Build sheet 4."""
    blocks = [ReportBlock(title="一、建模方法选择", data=model_adapter.get_param_table())]
    if (
        precomputed_tag_metrics is not None
        and precomputed_month_metrics is not None
        and not precomputed_tag_metrics.empty
        and not precomputed_month_metrics.empty
    ):
        tag_metrics = precomputed_tag_metrics
        month_metrics = precomputed_month_metrics
    else:
        tag_metrics, month_metrics = _build_split_metrics_tables(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            raw_date_col=raw_date_col,
            label_list=label_list,
            score_col=score_col,
            model_name=model_name,
            reverse_auc_label=reverse_auc_label,
            metrics_mode=metrics_mode,
            build_context=build_context,
        )
    blocks.append(ReportBlock(title="二、按tag模型效果", data=tag_metrics))
    blocks.append(ReportBlock(title="三、按月模型效果", data=month_metrics))

    binary_data = (
        primary_binary_data
        if primary_binary_data is not None
        else data.loc[data[primary_label].isin([0, 1])]
    )
    blocks.append(ReportBlock(title="四、模型分箱表现", blank_rows_after=1))
    for tag_value in _ordered_tag_values(binary_data[tag_col]):
        tag_frame = binary_data.loc[binary_data[tag_col] == tag_value]
        table = calculate_bin_performance_table(
            data=tag_frame,
            label_col=primary_label,
            score_col=score_col,
            edges=score_edges,
        )
        if not table.empty:
            blocks.append(
                ReportBlock(title=str(tag_value), data=table, blank_rows_after=3)
            )

    for month_value in _ordered_month_values(binary_data[month_col]):
        month_frame = binary_data.loc[binary_data[month_col] == month_value]
        table = calculate_bin_performance_table(
            data=month_frame,
            label_col=primary_label,
            score_col=score_col,
            edges=score_edges,
        )
        if not table.empty:
            blocks.append(
                ReportBlock(title=str(month_value), data=table, blank_rows_after=3)
            )
    return ReportSheet(name="模型表现", blocks=blocks)


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
    build_context: Optional[ReportBuildContext] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cache_key = (
        tuple(label_list),
        score_col,
        model_name,
        bool(reverse_auc_label),
        str(metrics_mode),
        tag_col,
        month_col,
        raw_date_col,
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
            build_context=build_context,
        )
        if not month_table.empty:
            month_table["近期月对比各集合PSI"] = month_table["观察点月"].map(month_latest_psi)
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
    build_context: Optional[ReportBuildContext] = None,
) -> pd.DataFrame:
    if group_mode not in {"tag", "month"}:
        raise ValueError(f"Unknown comparison mode: {group_mode}")

    group_cols = [tag_col] if group_mode == "tag" else [month_col]
    frames: List[pd.DataFrame] = []
    latest_map_by_model = {
        model_name: _build_latest_month_psi_map(
            data,
            month_col=month_col,
            score_col=score_col,
            build_context=build_context,
        )
        for model_name, score_col in model_columns
    }

    for model_name, score_col in model_columns:
        for label_col in label_list:
            train_reference = data.loc[
                (data[tag_col] == "train") & data[label_col].isin([0, 1]),
                score_col,
            ]
            table = _build_group_metrics(
                data=data,
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
                build_context=build_context,
            )
            if group_mode == "month":
                table["近期月对比各集合PSI"] = table["观察点月"].map(
                    latest_map_by_model[model_name]
                )
            frames.append(table)

    combined = pd.concat(frames, ignore_index=True)
    model_order = {
        model_name: index for index, (model_name, _) in enumerate(model_columns)
    }
    return _sort_pair_comparison_table(
        combined,
        model_order=model_order,
        group_mode=group_mode,
    )


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
    build_context: Optional[ReportBuildContext] = None,
) -> pd.DataFrame:
    group_key = (
        tuple(group_cols),
        label_col,
        score_col,
        tag_col,
        month_col,
        raw_date_col,
        model_name,
        bool(reverse_auc_label),
        str(metrics_mode),
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
    # Only include raw_date_col when grouping by tag (needed for date-range display).
    # Excluding it from the sort columns gives a massive speedup when raw_date_col
    # contains mixed-format datetime strings.
    _group_cols_set = set(group_cols)
    if _group_cols_set == {tag_col}:
        required_columns = list(
            dict.fromkeys([*group_cols, label_col, score_col, tag_col, raw_date_col])
        )
    else:
        required_columns = list(
            dict.fromkeys([*group_cols, label_col, score_col, tag_col, month_col])
        )
    ordered = data.loc[:, required_columns].copy()
    for group_col in group_cols:
        col = ordered[group_col]
        # Convert object-dtype columns to Categorical for faster sorting in groupby.
        # Only add "" as a category when there are actual NaN values to fill —
        # otherwise fillna("") would add a phantom "" category and create an extra
        # empty-string group (0 rows) when groupby(dropna=False).
        has_na = col.isna().any()
        if not isinstance(col.dtype, pd.CategoricalDtype):
            unique_vals = pd.unique(col)
            cats = list(unique_vals)
            if has_na and "" not in cats:
                cats.append("")
            ordered[group_col] = pd.Categorical(col, categories=cats, ordered=False)
        elif has_na and "" not in ordered[group_col].cat.categories:
            ordered[group_col] = ordered[group_col].cat.add_categories("")
        if has_na:
            ordered[group_col] = ordered[group_col].fillna("")

    psi_values: List[float] = []
    # Use observed=True to only iterate over groups that actually appear in the data.
    # This prevents creating redundant "phantom" groups for unused Categorical levels.
    grouped_frames = list(
        ordered.groupby(list(group_cols), sort=True, dropna=False, observed=True)
    )
    metric_groups = [
        (group_frame[label_col], group_frame[score_col])
        for _, group_frame in grouped_frames
    ]
    grouped_metrics = _unified_binary_metrics_batch(
        groups=metric_groups,
        lift_use_descending_score=not reverse_auc_label,
        reverse_auc_label=reverse_auc_label,
        metrics_mode=resolved_metrics_mode,
        engine=resolved_engine,
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
        # Define exact column order as requested
        # First sample/model/tag identification columns
        leading = ["样本标签", "模型", "样本集", "观察点月"]

        # Then performance metrics in the specific order
        metrics_cols = [
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
        ]

        # Then PSI columns
        psi_cols = ["train和各集合的PSI", "近期月对比各集合PSI"]

        # Assemble final column list, preserving only existing ones
        final_columns = []
        for col in leading + metrics_cols + psi_cols:
            if col in result.columns:
                final_columns.append(col)

        # Append any remaining columns that were not in the explicit list
        remaining = [c for c in result.columns if c not in final_columns]
        result = result.reindex(columns=final_columns + remaining)

    result = _sort_report_table(result, tag_column="样本集", month_column="观察点月")
    if build_context is not None:
        build_context.cache_set_group_metrics(group_key, result)
    return result


def _calculate_report_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
    reverse_auc_label: bool = False,
    metrics_mode: str = "exact",
) -> Dict[str, float]:
    """Calculate report metrics using the unified binary metrics path."""
    return _unified_binary_metrics(
        y_true=y_true,
        y_score=y_score,
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
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dim_col in dim_list:
        for dim_value, dim_frame in data.groupby(dim_col, dropna=False):
            for label_col in label_list:
                for model_name, score_col in score_model_columns:
                    metrics = _calculate_report_metrics(
                        dim_frame[label_col],
                        dim_frame[score_col],
                        reverse_auc_label=_lookup_reverse_auc(
                            score_metric_options, model_name
                        ),
                        metrics_mode=metrics_mode,
                    )
                    rows.append(
                        {
                            "维度列": dim_col,
                            "维度值": dim_value,
                            "样本标签": label_col,
                            "模型": model_name,
                            **metrics,
                        }
                    )
    result = pd.DataFrame(rows)
    if not result.empty:
        leading = ["维度列", "维度值", "样本标签", "模型"]
        metrics_cols = [
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
        ]
        final_columns = []
        for col in leading + metrics_cols:
            if col in result.columns:
                final_columns.append(col)
        remaining = [c for c in result.columns if c not in final_columns]
        result = result.reindex(columns=final_columns + remaining)
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


def _format_date_range(values: pd.Series) -> str:
    # Large scale optimization for 10M+ rows: truncate to date part first
    # to reduce cardinality before parsing. This is robust against mixed
    # formats and timestamps (seconds/milliseconds).
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


def _reshape_portrait_table(portrait: pd.DataFrame) -> pd.DataFrame:
    target_columns = [
        "画像变量",
        "模型",
        *[str(index) for index in range(1, 11)],
        "Missing",
    ]
    if portrait.empty:
        return pd.DataFrame(columns=target_columns)

    working = portrait.copy()
    working["分组"] = working["分组"].astype(str)
    mapped_frames: List[pd.DataFrame] = []

    for _, model_frame in working.groupby("模型", dropna=False, sort=False):
        non_missing_bins = sorted(
            [
                str(value)
                for value in model_frame["分组"].drop_duplicates().tolist()
                if str(value) != "Missing"
            ],
            key=_interval_left,
        )
        bin_label_map = {
            label: str(index + 1) for index, label in enumerate(non_missing_bins[:10])
        }
        mapped = model_frame.copy()
        mapped["_bin_label"] = mapped["分组"].map(bin_label_map)
        mapped.loc[mapped["分组"] == "Missing", "_bin_label"] = "Missing"
        mapped_frames.append(mapped.loc[mapped["_bin_label"].notna()])

    if not mapped_frames:
        return pd.DataFrame(columns=target_columns)
    working = pd.concat(mapped_frames, ignore_index=True)

    pivoted = (
        working.pivot_table(
            index=["画像变量", "模型"],
            columns="_bin_label",
            values="均值",
            aggfunc="mean",
        )
        .reindex(columns=[str(index) for index in range(1, 11)] + ["Missing"])
        .reset_index()
    )
    return pivoted.reindex(columns=target_columns)


def _build_feature_analysis_table(
    train_frame: pd.DataFrame,
    oot_frame: pd.DataFrame,
    month_frame: pd.DataFrame,
    tag_col: str,
    month_col: str,
    label_col: str,
    feature_cols: Sequence[str],
    feature_dict: pd.DataFrame,
    importance: pd.DataFrame,
    build_context: Optional[ReportBuildContext] = None,
) -> Tuple[pd.DataFrame, FeatureComputationArtifacts]:
    if not feature_cols:
        return pd.DataFrame(), FeatureComputationArtifacts()

    table_start = time.perf_counter()
    step_start = time.perf_counter()
    train_iv = _calculate_batch_iv_with_fallback(
        train_frame.loc[:, feature_cols],
        train_frame[label_col],
        engine=build_context.options.engine if build_context else "rust",
    )
    LOGGER.debug(
        "build_feature_analysis_table step finished | step=calculate_train_iv "
        "elapsed=%.3fs rows=%d",
        time.perf_counter() - step_start,
        len(train_iv),
    )

    step_start = time.perf_counter()
    oot_iv = (
        _calculate_batch_iv_with_fallback(
            oot_frame.loc[:, feature_cols],
            oot_frame[label_col],
            engine=build_context.options.engine if build_context else "rust",
        )
        if not oot_frame.empty
        else pd.DataFrame({"feature": feature_cols, "iv": np.nan})
    )
    LOGGER.debug(
        "build_feature_analysis_table step finished | step=calculate_oot_iv "
        "elapsed=%.3fs rows=%d",
        time.perf_counter() - step_start,
        len(oot_iv),
    )

    train_iv_lookup = (
        train_iv.set_index("feature")["iv"].to_dict() if not train_iv.empty else {}
    )
    oot_iv_lookup = (
        oot_iv.set_index("feature")["iv"].to_dict() if not oot_iv.empty else {}
    )
    importance_lookup = (
        importance.set_index("feature") if not importance.empty else pd.DataFrame()
    )
    train_missing_rate = train_frame.loc[:, feature_cols].isna().mean().to_dict()
    oot_missing_rate = (
        oot_frame.loc[:, feature_cols].isna().mean().to_dict()
        if not oot_frame.empty
        else {}
    )

    # Pre-compute all feature train-vs-oot PSI in one batch
    step_start = time.perf_counter()
    feature_psi_lookup: Dict[str, float] = {}
    psi_engine = build_context.options.engine if build_context else "python"
    if not oot_frame.empty and feature_cols:
        psi_values = calculate_feature_psi_pairs_batch(
            expected_groups=[train_frame[feature] for feature in feature_cols],
            actual_groups=[oot_frame[feature] for feature in feature_cols],
            buckets=10,
            engine=psi_engine,
        )
        feature_psi_lookup = {
            feature: float(value) for feature, value in zip(feature_cols, psi_values)
        }
    LOGGER.debug(
        "build_feature_analysis_table step finished | step=batch_feature_psi "
        "elapsed=%.3fs features=%d",
        time.perf_counter() - step_start,
        len(feature_psi_lookup),
    )

    total_features = len(feature_cols)
    worker_count = 1
    if (
        build_context is not None
        and total_features > 1
        and build_context.options.max_workers > 1
    ):
        worker_count = min(build_context.options.max_workers, total_features)
        if build_context.options.memory_mode == "compact":
            worker_count = min(worker_count, 4)

    loop_start = time.perf_counter()

    def _build_feature_row(
        index_feature: Tuple[int, str]
    ) -> Tuple[Dict[str, object], np.ndarray, pd.DataFrame, pd.DataFrame]:
        index, feature = index_feature
        meta = _lookup_feature_meta(feature_dict, feature)
        train_edges = np.asarray(
            build_reference_quantile_bins(train_frame[feature], bins=10),
            dtype=float,
        )
        train_stats = _build_feature_bin_stats(
            train_frame,
            feature=feature,
            label_col=label_col,
            edges=train_edges,
        )
        oot_stats = _build_feature_bin_stats(
            oot_frame,
            feature=feature,
            label_col=label_col,
            edges=train_edges,
        )
        ks_train = float(train_stats["ks"].max()) if not train_stats.empty else np.nan
        ks_oot = float(oot_stats["ks"].max()) if not oot_stats.empty else np.nan
        psi_value = feature_psi_lookup.get(feature, float("nan"))
        row = {
            "序号": index,
            "vars": feature,
            "变量解释含义": meta.get("中文名", ""),
            "来源": meta.get("来源", ""),
            "数据类型": str(train_frame[feature].dtype),
            "缺失率_train": float(train_missing_rate.get(feature, np.nan)),
            "缺失率_oot": float(
                oot_missing_rate.get(feature, np.nan) if not oot_frame.empty else np.nan
            ),
            "iv_train": float(train_iv_lookup.get(feature, np.nan)),
            "iv_oot": float(oot_iv_lookup.get(feature, np.nan)),
            "ks_train": ks_train,
            "ks_oot": ks_oot,
            "gain": _lookup_importance(importance_lookup, feature, "gain"),
            "gain_per": _lookup_importance(importance_lookup, feature, "gain_per"),
            "weight": _lookup_importance(importance_lookup, feature, "weight"),
            "weight_per": _lookup_importance(importance_lookup, feature, "weight_per"),
            "psi": psi_value,
            "指标表英文名": meta.get("指标表英文名", feature) or feature,
        }
        if index == 1 or index % 25 == 0 or index == total_features:
            LOGGER.debug(
                "build_feature_analysis_table progress | processed=%d/%d "
                "elapsed=%.3fs feature=%s",
                index,
                total_features,
                time.perf_counter() - loop_start,
                feature,
            )
        return row, train_edges, train_stats, oot_stats

    indexed_features = list(enumerate(feature_cols, start=1))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            feature_results = list(executor.map(_build_feature_row, indexed_features))
    else:
        feature_results = [
            _build_feature_row(index_feature) for index_feature in indexed_features
        ]

    rows = [item[0] for item in feature_results]
    artifacts = FeatureComputationArtifacts()
    for (_, feature), (_, edges, train_stats, oot_stats) in zip(
        indexed_features,
        feature_results,
    ):
        artifacts.edges_by_feature[feature] = edges
        artifacts.train_bin_stats_by_feature[feature] = train_stats
        artifacts.oot_bin_stats_by_feature[feature] = oot_stats

    result = pd.DataFrame(rows)
    if result.empty:
        return result, artifacts
    result = result.sort_values(
        ["gain", "weight"], ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    result["序号"] = np.arange(1, len(result) + 1)
    LOGGER.debug(
        "build_feature_analysis_table completed | elapsed=%.3fs rows=%d",
        time.perf_counter() - table_start,
        len(result),
    )
    return result, artifacts


def _calculate_batch_iv_with_fallback(
    X: pd.DataFrame,
    y: pd.Series,
    engine: str = "rust",
) -> pd.DataFrame:
    if engine == "python":
        return calculate_batch_iv(X, y, engine="python")
    try:
        return calculate_batch_iv(X, y, engine="rust")
    except Exception:
        return calculate_batch_iv(X, y, engine="python")


def _lookup_iv_value(iv_table: pd.DataFrame, feature: str) -> float:
    if iv_table.empty:
        return np.nan
    matched = iv_table.loc[iv_table["feature"] == feature, "iv"]
    if matched.empty:
        return np.nan
    return float(matched.iloc[0])


def _build_feature_selection_summary(
    feature_table: pd.DataFrame,
    feature_dict: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if feature_table.empty:
        return pd.DataFrame(), pd.DataFrame()
    total = len(feature_table)
    selected = min(total, 30)
    base = pd.DataFrame(
        [
            {
                "筛选条件": "重要性",
                "阈值": "IV>=0.02 / CORR<=0.7 / VIF<=5 / PSI<0.25",
                "筛选变量数量": total,
                "剩余变量数量": selected,
            }
        ]
    )
    if (
        feature_dict.empty
        or "来源" not in feature_dict.columns
        or "英文名" not in feature_dict.columns
    ):
        return base, pd.DataFrame()
    feature_source_map = (
        feature_dict.loc[:, ["英文名", "来源"]]
        .rename(columns={"来源": "来源_字典"})
        .drop_duplicates(
            subset=["英文名"],
            keep="first",
        )
    )
    type_table = (
        feature_table.merge(
            feature_source_map,
            left_on="vars",
            right_on="英文名",
            how="left",
        )
        .groupby("来源_字典", dropna=False)
        .size()
        .reset_index(name="变量数量")
    )
    type_table["重要性占比"] = type_table["变量数量"] / max(type_table["变量数量"].sum(), 1)
    type_table = type_table.rename(columns={"来源_字典": "变量类型"})
    return base, type_table


def _build_feature_bin_stats(
    frame: pd.DataFrame,
    feature: str,
    label_col: str,
    edges: Sequence[float],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    values, labels = _extract_feature_arrays(frame, feature, label_col)
    if values.size == 0:
        return pd.DataFrame()

    edges_array = np.asarray(edges, dtype=float)
    if len(edges_array) < 2:
        return pd.DataFrame()

    bin_indices = _assign_bin_indices(values, edges_array)
    non_missing_bins = len(edges_array) - 1
    all_bins = non_missing_bins + 1  # plus Missing bucket
    total_counts = np.bincount(bin_indices, minlength=all_bins).astype(float)
    bad_counts = np.bincount(
        bin_indices, weights=labels.astype(float), minlength=all_bins
    )
    good_counts = total_counts - bad_counts

    rows: List[Dict[str, object]] = []
    for bin_index in range(all_bins):
        total = total_counts[bin_index]
        if total <= 0:
            continue
        if bin_index == non_missing_bins:
            bin_label: object = "Missing"
            min_value = np.nan
            max_value = np.nan
        else:
            min_value = float(edges_array[bin_index])
            max_value = float(edges_array[bin_index + 1])
            bin_label = pd.Interval(left=min_value, right=max_value, closed="right")
        rows.append(
            {
                "bin": bin_label,
                "min": min_value,
                "max": max_value,
                "goods": float(good_counts[bin_index]),
                "bads": float(bad_counts[bin_index]),
                "total": float(total),
            }
        )

    grouped = pd.DataFrame(rows)
    if grouped.empty:
        return grouped

    grouped["total_prop"] = grouped["total"] / max(grouped["total"].sum(), 1)
    grouped["good_prop"] = grouped["goods"] / max(grouped["goods"].sum(), 1)
    grouped["bad_prop"] = grouped["bads"] / max(grouped["bads"].sum(), 1)
    grouped["bad_rate"] = grouped["bads"] / grouped["total"].clip(lower=1)
    grouped["woe"] = np.log(
        grouped["good_prop"].clip(lower=1e-8) / grouped["bad_prop"].clip(lower=1e-8)
    )
    grouped["iv"] = (grouped["good_prop"] - grouped["bad_prop"]) * grouped["woe"]
    grouped = grouped.assign(
        _missing_order=grouped["bin"].astype(str).eq("Missing").astype(int),
        _bin_order=grouped["min"].map(_interval_sort_key),
    ).sort_values(
        ["_missing_order", "_bin_order"],
        ascending=[True, True],
        kind="mergesort",
    )
    grouped["cum_bads_prop"] = grouped["bads"].cumsum() / max(grouped["bads"].sum(), 1)
    grouped["cum_goods_prop"] = grouped["goods"].cumsum() / max(
        grouped["goods"].sum(), 1
    )
    grouped["ks"] = abs(grouped["cum_bads_prop"] - grouped["cum_goods_prop"])
    overall_bad_rate = grouped["bads"].sum() / max(grouped["total"].sum(), 1)
    grouped["lift"] = grouped["bad_rate"] / max(overall_bad_rate, 1e-8)
    return grouped[
        [
            "bin",
            "min",
            "max",
            "goods",
            "bads",
            "total",
            "total_prop",
            "good_prop",
            "bad_prop",
            "bad_rate",
            "woe",
            "iv",
            "ks",
            "lift",
        ]
    ].reset_index(drop=True)


def _build_feature_monthly_metrics(
    all_data: pd.DataFrame,
    train_frame: pd.DataFrame,
    feature: str,
    label_col: str,
    month_col: str,
    edges: Sequence[float],
    engine: str = "python",
    metrics_mode: str = "exact",
) -> pd.DataFrame:
    if all_data.empty:
        return pd.DataFrame()
    edges_array = np.asarray(edges, dtype=float)
    train_values, train_labels = _extract_feature_arrays(
        train_frame, feature, label_col
    )
    if train_values.size == 0 or len(edges_array) < 2:
        return pd.DataFrame()
    train_indices = _assign_bin_indices(train_values, edges_array)
    non_missing_bins = len(edges_array) - 1
    all_bins = non_missing_bins + 1
    train_total_counts = np.bincount(train_indices, minlength=all_bins).astype(float)
    train_bad_counts = np.bincount(
        train_indices,
        weights=train_labels.astype(float),
        minlength=all_bins,
    )
    train_bin_scores = np.full(all_bins, np.nan, dtype=float)
    observed = train_total_counts > 0
    train_bin_scores[observed] = train_bad_counts[observed] / train_total_counts[
        observed
    ].clip(min=1.0)

    rows: List[Dict[str, object]] = []
    ordered_months = _ordered_month_values(all_data[month_col])
    month_data_list: List[Tuple[object, np.ndarray, np.ndarray]] = []
    for month_value in ordered_months:
        month_frame = all_data.loc[all_data[month_col] == month_value]
        month_vals, month_labs = _extract_feature_arrays(
            month_frame,
            feature=feature,
            label_col=label_col,
        )
        if month_vals.size == 0:
            continue
        month_data_list.append((month_value, month_vals, month_labs))

    # Batch PSI: one call for all months
    if month_data_list:
        psi_values = calculate_psi_batch(
            expected=pd.Series(train_values),
            actual_groups=[
                pd.Series(month_vals) for _, month_vals, _ in month_data_list
            ],
            buckets=10,
            engine=engine,
        )
    else:
        psi_values = []

    metric_groups: List[Tuple[np.ndarray, np.ndarray]] = []
    month_payload: List[Tuple[object, np.ndarray, np.ndarray]] = []
    for month_value, month_vals, month_labs in month_data_list:
        month_indices = _assign_bin_indices(month_vals, edges_array)
        month_bin_scores = train_bin_scores[month_indices]
        fallback = (month_indices == non_missing_bins).astype(float)
        month_bin_scores = np.where(
            np.isnan(month_bin_scores), fallback, month_bin_scores
        )
        metric_groups.append((month_labs, month_bin_scores))
        month_payload.append((month_value, month_labs, month_bin_scores))

    metric_rows = _unified_binary_metrics_batch(
        groups=metric_groups,
        metrics_mode=metrics_mode,
        engine=engine,
    )

    for idx, (month_value, _, _) in enumerate(month_payload):
        metrics = metric_rows[idx] if idx < len(metric_rows) else {}
        rows.append(
            {
                "month": month_value,
                **metrics,
                "PSI": (
                    float(psi_values[idx]) if idx < len(psi_values) else float("nan")
                ),
            }
        )
    return _sort_report_table(pd.DataFrame(rows), month_column="month")


def _extract_feature_arrays(
    frame: pd.DataFrame,
    feature: str,
    label_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if frame.empty:
        return np.array([], dtype=float), np.array([], dtype=np.int8)
    labels = frame[label_col]
    binary_mask = labels.isin([0, 1])
    if not bool(binary_mask.any()):
        return np.array([], dtype=float), np.array([], dtype=np.int8)
    numeric_values = pd.to_numeric(
        frame.loc[binary_mask, feature],
        errors="coerce",
    ).to_numpy(dtype=float)
    label_values = pd.to_numeric(
        labels.loc[binary_mask],
        errors="coerce",
    ).to_numpy(dtype=np.int8)
    return numeric_values, label_values


def _assign_bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    non_missing_bins = len(edges) - 1
    indices = np.empty(values.shape[0], dtype=np.int32)
    nan_mask = np.isnan(values)
    indices[nan_mask] = non_missing_bins
    if (~nan_mask).any():
        indices[~nan_mask] = np.searchsorted(
            edges[1:-1],
            values[~nan_mask],
            side="right",
        ).astype(np.int32)
    return indices


def _calculate_feature_metric_score(
    frame: pd.DataFrame,
    feature: str,
    label_col: str,
    edges: Sequence[float],
    metric_name: str,
) -> float:
    if frame.empty:
        return np.nan
    stats = _build_feature_bin_stats(frame, feature, label_col, edges)
    if stats.empty:
        return np.nan
    if metric_name == "ks":
        return float(stats["ks"].max())
    raise ValueError(f"Unsupported metric: {metric_name}")


def _load_feature_dictionary(feature_path: Optional[str]) -> pd.DataFrame:
    if not feature_path:
        return pd.DataFrame()
    path = Path(feature_path)
    if not path.exists():
        return pd.DataFrame()
    feature_dict = pd.read_csv(path)
    rename_map: Dict[str, str] = {}
    metric_alias_columns: List[str] = []
    for column in feature_dict.columns:
        normalized = str(column).strip().lower()
        if normalized in {"英文名", "english_name", "var_name", "variable_name"}:
            rename_map[column] = "英文名"
        elif normalized in {"中文名", "chinese_name", "variable_desc", "desc"}:
            rename_map[column] = "中文名"
        elif normalized in {
            "指标表英文名",
            "metric_table_english_name",
            "metric_table_name",
            "metric_table_en_name",
        }:
            rename_map[column] = "指标表英文名"
        elif normalized in {"表名", "table_name"}:
            rename_map[column] = "_legacy_metric_table_name"
            metric_alias_columns.append("_legacy_metric_table_name")
        elif normalized in {"数据源类型", "来源", "source", "source_type"}:
            rename_map[column] = "来源"
    feature_dict = feature_dict.rename(columns=rename_map)
    feature_dict = _coalesce_duplicate_columns(feature_dict)
    if "指标表英文名" not in feature_dict.columns:
        feature_dict["指标表英文名"] = np.nan
    for alias_column in metric_alias_columns:
        if alias_column in feature_dict.columns:
            feature_dict["指标表英文名"] = feature_dict["指标表英文名"].where(
                feature_dict["指标表英文名"].notna(),
                feature_dict[alias_column],
            )
            feature_dict = feature_dict.drop(columns=[alias_column], errors="ignore")
    return feature_dict


def _lookup_feature_meta(feature_dict: pd.DataFrame, feature: str) -> Dict[str, object]:
    if feature_dict.empty or "英文名" not in feature_dict.columns:
        return {"英文名": feature, "指标表英文名": feature}
    matched = feature_dict.loc[feature_dict["英文名"] == feature]
    if matched.empty:
        return {"英文名": feature, "指标表英文名": feature}
    meta = matched.iloc[0].to_dict()
    if not meta.get("指标表英文名"):
        meta["指标表英文名"] = feature
    return meta


def _coalesce_duplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if not frame.columns.duplicated().any():
        return frame
    combined: Dict[str, pd.Series] = {}
    ordered_names: List[str] = []
    for index, name in enumerate(frame.columns):
        column_values = frame.iloc[:, index]
        if name in combined:
            combined[name] = combined[name].where(
                combined[name].notna(),
                column_values,
            )
            continue
        combined[name] = column_values
        ordered_names.append(name)
    return pd.DataFrame({name: combined[name] for name in ordered_names})


def _lookup_importance(
    importance_lookup: pd.DataFrame, feature: str, column: str
) -> float:
    if importance_lookup.empty or feature not in importance_lookup.index:
        return 0.0
    return float(importance_lookup.loc[feature, column])


def _determine_feature_columns(
    data: pd.DataFrame,
    model_adapter: ModelAdapter,
    excluded: Sequence[str],
) -> List[str]:
    model_features = model_adapter.get_feature_names()
    if model_features:
        return [feature for feature in model_features if feature in data.columns]
    excluded_set = set(excluded)
    return [
        column
        for column in data.columns
        if column not in excluded_set and pd.api.types.is_numeric_dtype(data[column])
    ]


def _build_score_metric_options(
    score_direction_summary: pd.DataFrame,
) -> Dict[str, Dict[str, bool]]:
    if score_direction_summary.empty or "分数字段" not in score_direction_summary.columns:
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


def _ordered_month_values(values: pd.Series) -> List[object]:
    return _shared_ordered_month_values(values)


def _ordered_tag_values(values: pd.Series) -> List[object]:
    return _shared_ordered_tag_values(values)


def _month_sort_key(value: object) -> str:
    return _shared_month_sort_key(value)


def _tag_sort_key(value: object) -> tuple[int, str]:
    return _shared_tag_sort_key(value)


def _interval_left(value: object) -> float:
    if hasattr(value, "left"):
        return float(value.left)
    if value == "Missing":
        return np.nan
    text = str(value).replace("[", "").replace("(", "")
    return float(text.split(",")[0].strip())


def _interval_right(value: object) -> float:
    if hasattr(value, "right"):
        return float(value.right)
    if value == "Missing":
        return np.nan
    text = str(value).replace("]", "").replace(")", "")
    return float(text.split(",")[1].strip())


def _interval_sort_key(value: object) -> float:
    if pd.isna(value):
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")
