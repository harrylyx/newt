"""Table builders for the Excel model report."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from newt.features.analysis.batch_iv import calculate_batch_iv
from newt.metrics.reporting import (
    assign_reference_bins,
    build_reference_quantile_bins,
    calculate_bin_performance_table,
    calculate_binary_metrics,
    calculate_feature_psi,
    calculate_grouped_binary_metrics,
    calculate_latest_month_psi,
    calculate_portrait_means_by_score_bin,
    calculate_score_correlation_matrix,
    summarize_label_distribution,
)
from newt.reporting.model_adapter import ModelAdapter
from newt.results import ModelReportResult, ReportBlock, ReportChart, ReportSheet


SHEET_NAME_MAP = {
    1: "总览",
    2: "模型设计",
    3: "变量分析",
    4: "模型表现",
}


def resolve_sheet_names(sheet_list: Optional[Sequence[object]]) -> List[str]:
    """Resolve sheet selection input into ordered names."""
    if not sheet_list:
        return [SHEET_NAME_MAP[index] for index in sorted(SHEET_NAME_MAP)]

    resolved: List[str] = []
    for item in sheet_list:
        if isinstance(item, int):
            if item not in SHEET_NAME_MAP:
                raise ValueError(f"Unknown sheet index: {item}")
            sheet_name = SHEET_NAME_MAP[item]
        else:
            sheet_name = str(item)
            if sheet_name not in SHEET_NAME_MAP.values():
                raise ValueError(f"Unknown sheet name: {sheet_name}")
        if sheet_name not in resolved:
            resolved.append(sheet_name)
    return resolved


def build_report_result(
    data: pd.DataFrame,
    model_adapter: ModelAdapter,
    tag_col: str,
    score_col: str,
    month_col: str,
    label_list: Sequence[str],
    score_list: Sequence[str],
    dim_list: Sequence[str],
    var_list: Sequence[str],
    feature_path: Optional[str],
    selected_sheets: Sequence[str],
) -> ModelReportResult:
    """Build the full report result object."""
    sheets: Dict[str, ReportSheet] = {}
    feature_dict = _load_feature_dictionary(feature_path)
    feature_cols = _determine_feature_columns(
        data,
        model_adapter,
        excluded=[tag_col, month_col, score_col, *label_list, *score_list, *dim_list, *var_list],
    )
    primary_label = label_list[0]
    score_edges = build_reference_quantile_bins(
        data.loc[data[tag_col] == "train", score_col],
        bins=10,
    )

    builders = {
        "总览": lambda: build_overview_sheet(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            score_col=score_col,
            label_list=label_list,
            score_list=score_list,
            dim_list=dim_list,
            var_list=var_list,
        ),
        "模型设计": lambda: build_model_design_sheet(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            primary_label=primary_label,
            label_list=label_list,
            score_col=score_col,
        ),
        "变量分析": lambda: build_variable_analysis_sheet(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            primary_label=primary_label,
            score_col=score_col,
            feature_cols=feature_cols,
            feature_dict=feature_dict,
            model_adapter=model_adapter,
        ),
        "模型表现": lambda: build_model_performance_sheet(
            data=data,
            tag_col=tag_col,
            month_col=month_col,
            label_list=label_list,
            primary_label=primary_label,
            score_col=score_col,
            model_adapter=model_adapter,
            score_edges=score_edges,
        ),
    }

    for sheet_name in selected_sheets:
        sheets[sheet_name] = builders[sheet_name]()

    return ModelReportResult(
        sheets=sheets,
        metadata={
            "score_col": score_col,
            "label_list": list(label_list),
            "feature_columns": feature_cols,
        },
    )


def build_overview_sheet(
    data: pd.DataFrame,
    tag_col: str,
    month_col: str,
    score_col: str,
    label_list: Sequence[str],
    score_list: Sequence[str],
    dim_list: Sequence[str],
    var_list: Sequence[str],
) -> ReportSheet:
    """Build sheet 1."""
    blocks = [ReportBlock(title="一、目标与设计方案", blank_rows_after=1)]
    blocks.append(
        ReportBlock(
            title="表格1",
            data=pd.DataFrame(columns=["迭代目标", "问题", "原因", "迭代方向"]),
        )
    )
    blocks.append(ReportBlock(title="二、迭代效果", blank_rows_after=1))

    monthly_metrics = _build_monthly_metrics_table(
        data=data,
        tag_col=tag_col,
        month_col=month_col,
        label_list=label_list,
        score_col=score_col,
        model_name=score_col,
    )
    blocks.append(ReportBlock(title="按月模型效果", data=monthly_metrics))

    oot_frame = data.loc[data[tag_col] == "oot"].copy()
    if dim_list and not oot_frame.empty:
        dim_table = _build_dimensional_comparison(
            data=oot_frame,
            dim_list=dim_list,
            label_list=label_list,
            score_cols=[score_col, *score_list],
        )
        blocks.append(ReportBlock(title="分维度对比模型效果", data=dim_table))

    if score_list:
        tag_compare = _build_model_comparison(
            data=data,
            group_cols=[tag_col],
            label_list=label_list,
            score_cols=[score_col, *score_list],
            tag_col=tag_col,
            month_col=month_col,
        )
        month_compare = _build_model_comparison(
            data=data,
            group_cols=[tag_col, month_col],
            label_list=label_list,
            score_cols=[score_col, *score_list],
            tag_col=tag_col,
            month_col=month_col,
        )
        blocks.append(ReportBlock(title="按tag新老模型对比", data=tag_compare))
        blocks.append(ReportBlock(title="按月新老模型对比", data=month_compare))
        corr = calculate_score_correlation_matrix(oot_frame, [score_col, *score_list])
        blocks.append(ReportBlock(title="OOT相关性矩阵", data=corr.reset_index()))

    if var_list and not oot_frame.empty:
        portrait = calculate_portrait_means_by_score_bin(
            data=oot_frame,
            score_cols=[score_col, *score_list],
            variable_cols=var_list,
        )
        blocks.append(ReportBlock(title="OOT画像变量均值对比", data=portrait))

    return ReportSheet(name="总览", blocks=blocks)


def build_model_design_sheet(
    data: pd.DataFrame,
    tag_col: str,
    month_col: str,
    primary_label: str,
    label_list: Sequence[str],
    score_col: str,
) -> ReportSheet:
    """Build sheet 2."""
    blocks = [
        ReportBlock(title="一、模型设计方案", blank_rows_after=2),
        ReportBlock(title="二、样本和Y定义", blank_rows_after=2),
        ReportBlock(
            title="原始样本分布表",
            data=summarize_label_distribution(
                data=data,
                label_col=primary_label,
                tag_col=tag_col,
                month_col=month_col,
                include_blank_channel=True,
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
            ),
        ),
    ]

    sample_rows = []
    for label_col in label_list:
        for tag_value, tag_frame in data.groupby(tag_col, dropna=False):
            metrics = calculate_binary_metrics(tag_frame[label_col], tag_frame[score_col])
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
    for label_col in label_list:
        effect_table = _build_group_metrics(
            data=data,
            group_cols=[tag_col],
            label_col=label_col,
            score_col=score_col,
            tag_col=tag_col,
            month_col=month_col,
            model_name=score_col,
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
    score_col: str,
    feature_cols: Sequence[str],
    feature_dict: pd.DataFrame,
    model_adapter: ModelAdapter,
) -> ReportSheet:
    """Build sheet 3."""
    importance = model_adapter.get_importance_table()
    train_frame = data.loc[(data[tag_col] == "train") & data[primary_label].isin([0, 1])]
    oot_frame = data.loc[(data[tag_col] == "oot") & data[primary_label].isin([0, 1])]
    feature_table = _build_feature_analysis_table(
        train_frame=train_frame,
        oot_frame=oot_frame,
        month_frame=data,
        tag_col=tag_col,
        month_col=month_col,
        label_col=primary_label,
        feature_cols=feature_cols,
        feature_dict=feature_dict,
        importance=importance,
    )
    top_features = (
        feature_table.sort_values("gain", ascending=False)["vars"].head(30).tolist()
        if not feature_table.empty
        else []
    )

    blocks = [
        ReportBlock(
            title="1. 变量筛选",
            data=_build_feature_selection_summary(feature_table, feature_dict),
        ),
        ReportBlock(title="2. 变量分析", data=feature_table),
    ]

    for rank, feature in enumerate(top_features, start=1):
        feature_name = str(feature)
        edges = build_reference_quantile_bins(train_frame[feature_name], bins=10)
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
        )
        display_name = _lookup_feature_meta(feature_dict, feature_name).get("中文名", "")
        title_prefix = f"{rank}.{feature_name}"
        if display_name:
            title_prefix = f"{title_prefix} {display_name}"
        blocks.append(
            ReportBlock(
                title=f"{title_prefix} 分箱表",
                data=oot_bins,
                chart=ReportChart(
                    chart_type="combo",
                    category_column="bin",
                    value_columns=["total_prop"],
                    secondary_value_columns=["bad_rate"],
                    title=feature_name,
                ),
            )
        )
        blocks.append(
            ReportBlock(
                title=f"{title_prefix} 按月效果",
                data=monthly_table,
            )
        )

    return ReportSheet(name="变量分析", blocks=blocks)


def build_model_performance_sheet(
    data: pd.DataFrame,
    tag_col: str,
    month_col: str,
    label_list: Sequence[str],
    primary_label: str,
    score_col: str,
    model_adapter: ModelAdapter,
    score_edges: Sequence[float],
) -> ReportSheet:
    """Build sheet 4."""
    blocks = [ReportBlock(title="3.1 建模方法选择", data=model_adapter.get_param_table())]
    model_effect = _build_monthly_metrics_table(
        data=data,
        tag_col=tag_col,
        month_col=month_col,
        label_list=label_list,
        score_col=score_col,
        model_name=score_col,
    )
    blocks.append(ReportBlock(title="3.2 模型效果", data=model_effect))

    bin_rows: List[pd.DataFrame] = []
    for tag_value, tag_frame in data.loc[data[primary_label].isin([0, 1])].groupby(tag_col):
        table = calculate_bin_performance_table(
            data=tag_frame,
            label_col=primary_label,
            score_col=score_col,
            edges=score_edges,
        )
        if not table.empty:
            table.insert(0, "样本集", tag_value)
            table.insert(1, "观察点月", "")
            bin_rows.append(table)
            for month_value, month_frame in tag_frame.groupby(month_col):
                month_table = calculate_bin_performance_table(
                    data=month_frame,
                    label_col=primary_label,
                    score_col=score_col,
                    edges=score_edges,
                )
                if not month_table.empty:
                    month_table.insert(0, "样本集", tag_value)
                    month_table.insert(1, "观察点月", month_value)
                    bin_rows.append(month_table)
    blocks.append(
        ReportBlock(
            title="3.3 模型分箱表现",
            data=pd.concat(bin_rows, ignore_index=True) if bin_rows else pd.DataFrame(),
        )
    )
    return ReportSheet(name="模型表现", blocks=blocks)


def _build_monthly_metrics_table(
    data: pd.DataFrame,
    tag_col: str,
    month_col: str,
    label_list: Sequence[str],
    score_col: str,
    model_name: str,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    latest_psi = calculate_latest_month_psi(data, tag_col=tag_col, month_col=month_col, score_col=score_col)
    for label_col in label_list:
        train_reference = data.loc[
            (data[tag_col] == "train") & data[label_col].isin([0, 1]),
            score_col,
        ]
        table = _build_group_metrics(
            data=data,
            group_cols=[tag_col, month_col],
            label_col=label_col,
            score_col=score_col,
            tag_col=tag_col,
            month_col=month_col,
            train_reference=train_reference,
            latest_month_psi=latest_psi,
            model_name=model_name,
        )
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _build_model_comparison(
    data: pd.DataFrame,
    group_cols: Sequence[str],
    label_list: Sequence[str],
    score_cols: Sequence[str],
    tag_col: str,
    month_col: str,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for score_col in score_cols:
        latest_psi = calculate_latest_month_psi(
            data,
            tag_col=tag_col,
            month_col=month_col,
            score_col=score_col,
        )
        for label_col in label_list:
            train_reference = data.loc[
                (data[tag_col] == "train") & data[label_col].isin([0, 1]),
                score_col,
            ]
            frames.append(
                _build_group_metrics(
                    data=data,
                    group_cols=group_cols,
                    label_col=label_col,
                    score_col=score_col,
                    tag_col=tag_col,
                    month_col=month_col,
                    train_reference=train_reference,
                    latest_month_psi=latest_psi,
                    model_name=score_col,
                )
            )
    return pd.concat(frames, ignore_index=True)


def _build_group_metrics(
    data: pd.DataFrame,
    group_cols: Sequence[str],
    label_col: str,
    score_col: str,
    tag_col: str,
    month_col: str,
    train_reference: Optional[pd.Series] = None,
    latest_month_psi: Optional[pd.DataFrame] = None,
    model_name: str = "",
) -> pd.DataFrame:
    return calculate_grouped_binary_metrics(
        data=data,
        group_cols=group_cols,
        label_col=label_col,
        score_col=score_col,
        tag_col=tag_col,
        month_col=month_col,
        train_reference=train_reference,
        latest_month_psi=latest_month_psi,
        model_name=model_name,
    )


def _build_dimensional_comparison(
    data: pd.DataFrame,
    dim_list: Sequence[str],
    label_list: Sequence[str],
    score_cols: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dim_col in dim_list:
        for dim_value, dim_frame in data.groupby(dim_col, dropna=False):
            for label_col in label_list:
                for score_col in score_cols:
                    metrics = calculate_binary_metrics(dim_frame[label_col], dim_frame[score_col])
                    rows.append(
                        {
                            "维度列": dim_col,
                            "维度值": dim_value,
                            "样本标签": label_col,
                            "模型": score_col,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


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
) -> pd.DataFrame:
    if not feature_cols:
        return pd.DataFrame()

    train_iv = calculate_batch_iv(train_frame.loc[:, feature_cols], train_frame[label_col], engine="rust")
    oot_iv = (
        calculate_batch_iv(oot_frame.loc[:, feature_cols], oot_frame[label_col], engine="rust")
        if not oot_frame.empty
        else pd.DataFrame({"feature": feature_cols, "iv": np.nan})
    )
    importance_lookup = importance.set_index("feature") if not importance.empty else pd.DataFrame()
    rows: List[Dict[str, object]] = []

    for index, feature in enumerate(feature_cols, start=1):
        meta = _lookup_feature_meta(feature_dict, feature)
        train_edges = build_reference_quantile_bins(train_frame[feature], bins=10)
        ks_train = _calculate_feature_metric_score(
            train_frame,
            feature=feature,
            label_col=label_col,
            edges=train_edges,
            metric_name="ks",
        )
        ks_oot = _calculate_feature_metric_score(
            oot_frame,
            feature=feature,
            label_col=label_col,
            edges=train_edges,
            metric_name="ks",
        )
        psi_value = calculate_feature_psi(
            train_frame[feature],
            oot_frame[feature] if not oot_frame.empty else pd.Series(dtype=float),
            edges=train_edges,
        )
        row = {
            "序号": index,
            "vars": feature,
            "变量解释含义": meta.get("中文名", ""),
            "来源": meta.get("来源", ""),
            "数据类型": str(train_frame[feature].dtype),
            "缺失率_train": train_frame[feature].isna().mean(),
            "缺失率_oot": oot_frame[feature].isna().mean() if not oot_frame.empty else np.nan,
            "iv_train": train_iv.loc[train_iv["feature"] == feature, "iv"].iloc[0],
            "iv_oot": oot_iv.loc[oot_iv["feature"] == feature, "iv"].iloc[0]
            if not oot_iv.empty and feature in oot_iv["feature"].values
            else np.nan,
            "ks_train": ks_train,
            "ks_oot": ks_oot,
            "gain": _lookup_importance(importance_lookup, feature, "gain"),
            "gain_per": _lookup_importance(importance_lookup, feature, "gain_per"),
            "weight": _lookup_importance(importance_lookup, feature, "weight"),
            "weight_per": _lookup_importance(importance_lookup, feature, "weight_per"),
            "psi": psi_value,
            "指标表英文名": feature,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _build_feature_selection_summary(
    feature_table: pd.DataFrame,
    feature_dict: pd.DataFrame,
) -> pd.DataFrame:
    if feature_table.empty:
        return pd.DataFrame()
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
    if feature_dict.empty or "来源" not in feature_dict.columns:
        return base
    type_table = (
        feature_table.merge(feature_dict, left_on="vars", right_on="英文名", how="left")
        .groupby("来源", dropna=False)
        .size()
        .reset_index(name="变量数量")
    )
    type_table["重要性占比"] = type_table["变量数量"] / max(type_table["变量数量"].sum(), 1)
    type_table = type_table.rename(columns={"来源": "变量类型"})
    return pd.concat([base, type_table], ignore_index=True, sort=False)


def _build_feature_bin_stats(
    frame: pd.DataFrame,
    feature: str,
    label_col: str,
    edges: Sequence[float],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame.loc[frame[label_col].isin([0, 1]), [feature, label_col]].copy()
    working["bin"] = assign_reference_bins(working[feature], edges)
    grouped = (
        working.groupby("bin", dropna=False)[label_col]
        .agg(total="count", bads="sum")
        .reset_index()
    )
    grouped["goods"] = grouped["total"] - grouped["bads"]
    grouped["total_prop"] = grouped["total"] / max(grouped["total"].sum(), 1)
    grouped["good_prop"] = grouped["goods"] / max(grouped["goods"].sum(), 1)
    grouped["bad_prop"] = grouped["bads"] / max(grouped["bads"].sum(), 1)
    grouped["bad_rate"] = grouped["bads"] / grouped["total"].clip(lower=1)
    grouped["woe"] = np.log(
        grouped["good_prop"].clip(lower=1e-8) / grouped["bad_prop"].clip(lower=1e-8)
    )
    grouped["iv"] = (grouped["good_prop"] - grouped["bad_prop"]) * grouped["woe"]
    grouped["cum_bads_prop"] = grouped["bads"].cumsum() / max(grouped["bads"].sum(), 1)
    grouped["cum_goods_prop"] = grouped["goods"].cumsum() / max(grouped["goods"].sum(), 1)
    grouped["ks"] = abs(grouped["cum_bads_prop"] - grouped["cum_goods_prop"])
    overall_bad_rate = grouped["bads"].sum() / max(grouped["total"].sum(), 1)
    grouped["lift"] = grouped["bad_rate"] / max(overall_bad_rate, 1e-8)
    grouped["min"] = grouped["bin"].apply(_interval_left)
    grouped["max"] = grouped["bin"].apply(_interval_right)
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
    ]


def _build_feature_monthly_metrics(
    all_data: pd.DataFrame,
    train_frame: pd.DataFrame,
    feature: str,
    label_col: str,
    month_col: str,
    edges: Sequence[float],
) -> pd.DataFrame:
    if all_data.empty:
        return pd.DataFrame()
    train_stats = _build_feature_bin_stats(train_frame, feature, label_col, edges)
    train_bin_scores = (
        train_stats.set_index("bin")["bad_rate"].to_dict() if not train_stats.empty else {}
    )
    rows: List[Dict[str, object]] = []
    for month_value, month_frame in all_data.groupby(month_col):
        clean = month_frame.loc[month_frame[label_col].isin([0, 1]), [feature, label_col]].copy()
        if clean.empty:
            continue
        clean["bin"] = assign_reference_bins(clean[feature], edges)
        clean["bin_score"] = clean["bin"].map(train_bin_scores).fillna(clean["bin"].astype(str).eq("Missing").astype(float))
        metrics = calculate_binary_metrics(clean[label_col], clean["bin_score"])
        rows.append(
            {
                "month": month_value,
                **metrics,
                "PSI": calculate_feature_psi(train_frame[feature], clean[feature], edges),
            }
        )
    return pd.DataFrame(rows)


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
    rename_map = {}
    for column in feature_dict.columns:
        normalized = str(column).strip().lower()
        if normalized in {"英文名", "english_name", "var_name", "variable_name"}:
            rename_map[column] = "英文名"
        elif normalized in {"中文名", "chinese_name", "variable_desc", "desc"}:
            rename_map[column] = "中文名"
        elif normalized in {"表名", "table_name"}:
            rename_map[column] = "表名"
        elif normalized in {"数据源类型", "来源", "source", "source_type"}:
            rename_map[column] = "来源"
    return feature_dict.rename(columns=rename_map)


def _lookup_feature_meta(feature_dict: pd.DataFrame, feature: str) -> Dict[str, object]:
    if feature_dict.empty or "英文名" not in feature_dict.columns:
        return {"英文名": feature}
    matched = feature_dict.loc[feature_dict["英文名"] == feature]
    if matched.empty:
        return {"英文名": feature}
    return matched.iloc[0].to_dict()


def _lookup_importance(importance_lookup: pd.DataFrame, feature: str, column: str) -> float:
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
