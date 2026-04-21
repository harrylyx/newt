"""Main sheet builders for report generation."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from newt.metrics.reporting import (
    build_reference_quantile_bins,
    calculate_bin_performance_table,
    summarize_label_distribution,
)
from newt.reporting.model_adapter import ModelAdapter
from newt.reporting.table_context import ReportBuildContext
from newt.results import ReportBlock, ReportChart, ReportSheet

from . import feature_metrics, group_metrics

LOGGER = logging.getLogger("newt.reporting.tables")


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
            title=f"{section_mapping[current_section]}、目标与设计方案",
            blank_rows_after=1,
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
                title=f"{section_mapping[current_section]}、分维度对比",
                blank_rows_after=1,
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
                title=f"{section_mapping[current_section]}、画像变量",
                blank_rows_after=1,
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
    """Build model design sheet."""
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
        for tag_value in group_metrics._ordered_tag_values(data[tag_col]):
            tag_frame = data.loc[data[tag_col] == tag_value]
            metrics = group_metrics._calculate_report_metrics(
                frame=tag_frame,
                label_col=label_col,
                score_col=score_col,
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
            effect_table = group_metrics._build_group_metrics(
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
                metric_basis="count",
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
    """Build variable analysis sheet."""
    sheet_start = time.perf_counter()

    step_start = time.perf_counter()
    importance = model_adapter.get_importance_table()
    LOGGER.debug(
        "build_variable_analysis_sheet step finished | step=get_importance_table "
        "elapsed=%.3fs rows=%d",
        time.perf_counter() - step_start,
        len(importance),
    )
    lr_feature_summary = model_adapter.get_lr_feature_summary_table()
    lr_model_summary = model_adapter.get_lr_model_summary_table()

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
    feature_bin_edges = {}
    get_feature_bin_edges = getattr(model_adapter, "get_feature_bin_edges", None)
    if callable(get_feature_bin_edges):
        feature_bin_edges = {
            feature: edges
            for feature in feature_cols
            for edges in [get_feature_bin_edges(feature)]
            if edges is not None
        }
    feature_table, feature_artifacts = feature_metrics._build_feature_analysis_table(
        train_frame=train_frame,
        oot_frame=oot_frame,
        month_frame=data,
        tag_col=tag_col,
        month_col=month_col,
        label_col=primary_label,
        feature_cols=feature_cols,
        feature_dict=feature_dict,
        importance=importance,
        feature_bin_edges=feature_bin_edges,
        lr_feature_summary=lr_feature_summary,
        build_context=build_context,
    )
    LOGGER.debug(
        "build_variable_analysis_sheet step finished | "
        "step=build_feature_analysis_table elapsed=%.3fs feature_rows=%d",
        time.perf_counter() - step_start,
        len(feature_table),
    )

    # Hide gain/weight columns for LR-style and scorecard models to avoid
    # misinterpreting proxy importance as tree gain.
    wrapped_model = getattr(model_adapter, "model", None)
    model_class_name = getattr(
        getattr(wrapped_model, "__class__", None), "__name__", ""
    )
    model_module_name = getattr(
        getattr(wrapped_model, "__class__", None), "__module__", ""
    )
    model_class_name = str(model_class_name).lower()
    model_module_name = str(model_module_name).lower()
    hide_gain_weight_columns = model_adapter.model_family == "scorecard" or (
        "logistic" in model_class_name or "logistic" in model_module_name
    )
    if hide_gain_weight_columns and not feature_table.empty:
        feature_table = feature_table.drop(
            columns=["gain", "gain_per", "weight", "weight_per"],
            errors="ignore",
        )

    if feature_table.empty:
        top_features: List[str] = []
    elif model_adapter.model_family == "scorecard":
        top_features = feature_table["vars"].tolist()
    else:
        top_features = feature_table["vars"].head(30).tolist()

    summary_table, type_table = feature_metrics._build_feature_selection_summary(
        feature_table,
        feature_dict,
        selected_count=len(top_features),
    )
    blocks = [
        ReportBlock(title="一、变量筛选", data=summary_table),
    ]
    if not type_table.empty:
        blocks.append(ReportBlock(title="变量类型分布", data=type_table))

    blocks.append(ReportBlock(title="二、变量分析", data=feature_table))
    if not lr_model_summary.empty:
        blocks.append(ReportBlock(title="三、模型统计摘要", data=lr_model_summary))
        blocks.append(ReportBlock(title="四、单变量分析", blank_rows_after=1))
    else:
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
    ) -> Tuple[int, str, str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
        rank, feature_name = job
        feature_start = time.perf_counter()
        edges = feature_artifacts.edges_by_feature.get(feature_name)
        if edges is None:
            edges = build_reference_quantile_bins(train_frame[feature_name], bins=10)
        train_bins = feature_artifacts.train_bin_stats_by_feature.get(feature_name)
        if train_bins is None:
            train_bins = feature_metrics._build_feature_bin_stats(
                frame=train_frame,
                feature=feature_name,
                label_col=primary_label,
                edges=edges,
            )
        oot_bins = feature_artifacts.oot_bin_stats_by_feature.get(feature_name)
        if oot_bins is None:
            oot_bins = feature_metrics._build_feature_bin_stats(
                frame=oot_frame,
                feature=feature_name,
                label_col=primary_label,
                edges=edges,
            )
        monthly_table = feature_metrics._build_feature_monthly_metrics(
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
            use_fixed_edges_for_psi=(
                feature_artifacts.use_feature_edges_for_psi_by_feature.get(
                    feature_name,
                    False,
                )
            ),
        )
        display_name = feature_metrics._lookup_feature_meta(
            feature_dict, feature_name
        ).get("中文名", "")
        title_prefix = f"{rank}.{feature_name}"
        full_title = f"{title_prefix} {display_name}" if display_name else title_prefix
        chart_title = str(display_name or feature_name)
        return (
            rank,
            feature_name,
            full_title,
            chart_title,
            train_bins,
            oot_bins,
            monthly_table,
            time.perf_counter() - feature_start,
        )

    feature_results: List[
        Tuple[int, str, str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]
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
        train_bins,
        oot_bins,
        monthly_table,
        elapsed,
    ) in feature_results:
        train_block_title = f"{full_title} 训练分箱表"
        oot_block_title = f"{full_title} OOT分箱表"
        blocks.append(
            ReportBlock(
                title=full_title,
                blank_rows_after=0,
            )
        )
        blocks.append(ReportBlock(title=train_block_title, data=train_bins))
        blocks.append(ReportBlock(title=oot_block_title, data=oot_bins))
        blocks.append(ReportBlock(title=f"{full_title} 按月效果", data=monthly_table))
        if not train_bins.empty:
            blocks.append(
                ReportBlock(
                    title="",
                    chart=ReportChart(
                        chart_type="combo",
                        category_column="bin",
                        value_columns=["total_prop"],
                        secondary_value_columns=["bad_rate"],
                        title=f"{chart_title} 训练",
                        source_block_title=train_block_title,
                    ),
                    blank_rows_after=1,
                )
            )
        if not oot_bins.empty:
            blocks.append(
                ReportBlock(
                    title="",
                    chart=ReportChart(
                        chart_type="combo",
                        category_column="bin",
                        value_columns=["total_prop"],
                        secondary_value_columns=["bad_rate"],
                        title=f"{chart_title} OOT",
                        source_block_title=oot_block_title,
                    ),
                    blank_rows_after=2,
                )
            )
        feature_timings.append((feature_name, elapsed))
        LOGGER.debug(
            "build_variable_analysis_sheet feature finished | feature=%s "
            "elapsed=%.3fs train_bins=%d oot_bins=%d monthly_rows=%d",
            feature_name,
            elapsed,
            len(train_bins),
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


def build_scorecard_details_sheet(
    model_adapter: ModelAdapter,
) -> ReportSheet:
    """Build scorecard decomposition sheet."""
    base_table = model_adapter.get_scorecard_base_table()
    points_table = model_adapter.get_scorecard_points_table()
    if not points_table.empty:
        points_table = points_table.rename(
            columns={
                "feature": "变量",
                "bin": "分箱",
                "woe": "WOE",
                "points": "分值",
            }
        )

    blocks = [ReportBlock(title="一、评分卡计算参数", data=base_table)]
    blocks.append(ReportBlock(title="二、评分卡分值拆解", data=points_table))
    return ReportSheet(name="评分卡计算明细", blocks=blocks)


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
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
    build_context: Optional[ReportBuildContext] = None,
) -> ReportSheet:
    """Build model performance sheet."""
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
        tag_metrics, month_metrics = group_metrics._build_split_metrics_tables(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            raw_date_col=raw_date_col,
            label_list=label_list,
            score_col=score_col,
            model_name=model_name,
            reverse_auc_label=reverse_auc_label,
            metrics_mode=metrics_mode,
            metric_basis="count",
            prin_bal_amount_col=prin_bal_amount_col,
            loan_amount_col=loan_amount_col,
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
    for tag_value in group_metrics._ordered_tag_values(binary_data[tag_col]):
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

    for month_value in group_metrics._ordered_month_values(binary_data[month_col]):
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
