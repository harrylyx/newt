"""Feature-level metric helpers for variable analysis sheets."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from newt.features.analysis.batch_iv import calculate_batch_iv
from newt.metrics.binary_metrics import (
    calculate_binary_metrics_batch as _unified_binary_metrics_batch,
)
from newt.metrics.psi import calculate_feature_psi_pairs_batch, calculate_psi_batch
from newt.metrics.reporting import build_reference_quantile_bins
from newt.reporting.table_context import FeatureComputationArtifacts, ReportBuildContext

from . import group_metrics

LOGGER = logging.getLogger("newt.reporting.tables")


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
    lr_feature_summary: Optional[pd.DataFrame] = None,
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
        from concurrent.futures import ThreadPoolExecutor

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
    if lr_feature_summary is not None and not lr_feature_summary.empty:
        merged_lr_summary = lr_feature_summary.rename(
            columns={"feature": "vars"}
        ).copy()
        result = result.merge(merged_lr_summary, on="vars", how="left")
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
    selected_count: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if feature_table.empty:
        return pd.DataFrame(), pd.DataFrame()
    total = len(feature_table)
    selected = min(total, 30) if selected_count is None else min(total, selected_count)
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
    all_bins = non_missing_bins + 1
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
    ordered_months = group_metrics._ordered_month_values(all_data[month_col])
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
    return group_metrics._sort_report_table(pd.DataFrame(rows), month_column="month")


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
    model_features: Sequence[str],
    excluded: Sequence[str],
) -> List[str]:
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


def _interval_sort_key(value: object) -> float:
    if pd.isna(value):
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")
