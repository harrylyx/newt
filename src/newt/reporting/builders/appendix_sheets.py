"""Appendix sheet builders for report generation."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from newt.metrics.reporting import (
    calculate_portrait_means_by_score_bin,
    calculate_score_correlation_matrix,
)
from newt.reporting.table_context import ReportBuildContext
from newt.results import ReportBlock, ReportSheet

from . import feature_metrics, group_metrics

_AMOUNT_SHEET_METRIC_COLUMNS = [
    "放款金额",
    "逾期本金",
    "金额坏占比",
    "放款金额占比",
    "逾期本金占比",
    "金额AUC",
    "金额KS",
    "10%金额lift",
    "5%金额lift",
    "2%金额lift",
    "1%金额lift",
]
_AMOUNT_SHEET_PSI_COLUMNS = [
    "train和各集合的PSI",
    "近期月对比各集合PSI",
]


def _project_amount_sheet_table(
    table: pd.DataFrame,
    key_columns: Sequence[str],
) -> pd.DataFrame:
    output_columns = [
        *key_columns,
        *_AMOUNT_SHEET_METRIC_COLUMNS,
        *_AMOUNT_SHEET_PSI_COLUMNS,
    ]
    if table.empty:
        return pd.DataFrame(columns=output_columns)

    present_key_columns = [col for col in key_columns if col in table.columns]
    projected = table.loc[:, present_key_columns].copy()
    value_candidates = {
        "放款金额": ["放款金额", "总"],
        "逾期本金": ["逾期本金", "坏"],
        "金额坏占比": ["金额坏占比", "坏占比"],
        "放款金额占比": ["放款金额占比"],
        "逾期本金占比": ["逾期本金占比"],
        "金额AUC": ["金额AUC", "AUC"],
        "金额KS": ["金额KS", "KS"],
        "10%金额lift": ["10%金额lift", "10%lift"],
        "5%金额lift": ["5%金额lift", "5%lift"],
        "2%金额lift": ["2%金额lift", "2%lift"],
        "1%金额lift": ["1%金额lift", "1%lift"],
    }
    for target_col, candidates in value_candidates.items():
        source_col = next((col for col in candidates if col in table.columns), None)
        projected[target_col] = table[source_col] if source_col is not None else np.nan

    psi_columns = [col for col in _AMOUNT_SHEET_PSI_COLUMNS if col in table.columns]
    for psi_col in psi_columns:
        projected[psi_col] = table[psi_col]

    ordered = [
        *[col for col in key_columns if col in projected.columns],
        *_AMOUNT_SHEET_METRIC_COLUMNS,
        *psi_columns,
    ]
    return projected.reindex(columns=ordered)


def build_dimensional_comparison_sheet(
    data: pd.DataFrame,
    tag_col: str,
    dim_list: Sequence[str],
    label_list: Sequence[str],
    score_model_columns: Sequence[Tuple[str, str]],
    score_metric_options: dict,
    oot_frame: Optional[pd.DataFrame] = None,
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
    build_context: Optional[ReportBuildContext] = None,
) -> ReportSheet:
    """Build appendix sheet for dimensional model-effect comparison."""
    oot_frame = oot_frame if oot_frame is not None else data.loc[data[tag_col] == "oot"]
    dim_table = (
        group_metrics._build_dimensional_comparison(
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
            metric_basis="count",
            prin_bal_amount_col=prin_bal_amount_col,
            loan_amount_col=loan_amount_col,
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
    score_metric_options: dict,
    oot_frame: Optional[pd.DataFrame] = None,
    prin_bal_amount_col: Optional[str] = None,
    loan_amount_col: Optional[str] = None,
    build_context: Optional[ReportBuildContext] = None,
) -> ReportSheet:
    """Build appendix sheet for old/new model comparison."""
    blocks = [ReportBlock(title="一、新老模型对比", blank_rows_after=1)]
    if len(model_columns) >= 2:
        for old_model_name, old_score_col in model_columns[1:]:
            pair_models = [model_columns[0], (old_model_name, old_score_col)]
            tag_compare = group_metrics._build_model_pair_comparison(
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
                metric_basis="count",
                prin_bal_amount_col=prin_bal_amount_col,
                loan_amount_col=loan_amount_col,
                build_context=build_context,
            )
            month_compare = group_metrics._build_model_pair_comparison(
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
                metric_basis="count",
                prin_bal_amount_col=prin_bal_amount_col,
                loan_amount_col=loan_amount_col,
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
        meta = feature_metrics._lookup_feature_meta(feature_dict, str(variable_name))
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


def build_amount_metrics_sheet(
    data: pd.DataFrame,
    tag_col: str,
    month_col: str,
    raw_date_col: str,
    label_list: Sequence[str],
    primary_model_name: str,
    primary_score_col: str,
    model_columns: Sequence[Tuple[str, str]],
    dim_list: Sequence[str],
    score_metric_options: dict,
    prin_bal_amount_col: str,
    loan_amount_col: str,
    build_context: Optional[ReportBuildContext] = None,
) -> ReportSheet:
    """Build appendix sheet for amount-basis model metrics."""
    blocks = [ReportBlock(title="一、金额口径指标", blank_rows_after=1)]
    tag_metrics, month_metrics = group_metrics._build_split_metrics_tables(
        data=data,
        tag_col=tag_col,
        month_col=month_col,
        raw_date_col=raw_date_col,
        label_list=label_list,
        score_col=primary_score_col,
        model_name=primary_model_name,
        reverse_auc_label=group_metrics._lookup_reverse_auc(
            score_metric_options, primary_model_name
        ),
        metrics_mode=(
            build_context.options.metrics_mode if build_context is not None else "exact"
        ),
        metric_basis="amount",
        prin_bal_amount_col=prin_bal_amount_col,
        loan_amount_col=loan_amount_col,
        build_context=build_context,
    )
    tag_metrics = _project_amount_sheet_table(
        tag_metrics,
        key_columns=["样本标签", "模型", "样本集", "观察点月"],
    )
    month_metrics = _project_amount_sheet_table(
        month_metrics,
        key_columns=["样本标签", "模型", "样本集", "观察点月"],
    )
    blocks.append(ReportBlock(title="按tag模型效果", data=tag_metrics))
    blocks.append(ReportBlock(title="按月模型效果", data=month_metrics))

    oot_frame = data.loc[data[tag_col] == "oot"]
    if dim_list and not oot_frame.empty:
        dim_table = group_metrics._build_dimensional_comparison(
            data=oot_frame,
            dim_list=dim_list,
            label_list=label_list,
            score_model_columns=model_columns,
            score_metric_options=score_metric_options,
            metrics_mode=(
                build_context.options.metrics_mode
                if build_context is not None
                else "exact"
            ),
            metric_basis="amount",
            prin_bal_amount_col=prin_bal_amount_col,
            loan_amount_col=loan_amount_col,
        )
        dim_table = _project_amount_sheet_table(
            dim_table,
            key_columns=["维度列", "维度值", "样本标签", "模型"],
        )
        blocks.append(ReportBlock(title="分维度对比", data=dim_table))

    if len(model_columns) >= 2:
        for old_model_name, old_score_col in model_columns[1:]:
            pair_models = [model_columns[0], (old_model_name, old_score_col)]
            tag_compare = group_metrics._build_model_pair_comparison(
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
                metric_basis="amount",
                prin_bal_amount_col=prin_bal_amount_col,
                loan_amount_col=loan_amount_col,
                build_context=build_context,
            )
            month_compare = group_metrics._build_model_pair_comparison(
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
                metric_basis="amount",
                prin_bal_amount_col=prin_bal_amount_col,
                loan_amount_col=loan_amount_col,
                build_context=build_context,
            )
            tag_compare = _project_amount_sheet_table(
                tag_compare,
                key_columns=["样本标签", "模型", "样本集", "观察点月"],
            )
            month_compare = _project_amount_sheet_table(
                month_compare,
                key_columns=["样本标签", "模型", "样本集", "观察点月"],
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
    return ReportSheet(name="金额指标", blocks=blocks)


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
            key=feature_metrics._interval_left,
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
