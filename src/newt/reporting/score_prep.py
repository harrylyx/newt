"""Prepare report-only score columns and direction metadata."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from newt.metrics.auc import calculate_auc


def prepare_report_scores(
    data: pd.DataFrame,
    tag_col: str,
    label_col: str,
    score_names: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame]:
    """Create report-only score columns and direction summary."""
    prepared = data.copy()
    score_columns: Dict[str, str] = {}
    rows: List[Dict[str, object]] = []

    for index, score_name in enumerate(_dedupe(score_names), start=1):
        numeric = pd.to_numeric(prepared[score_name], errors="coerce")
        report_column = f"__report_score_{index}"
        prepared[report_column] = numeric

        raw_auc, source_name = _resolve_direction_auc(
            frame=prepared,
            tag_col=tag_col,
            label_col=label_col,
            score=numeric,
        )
        reverse_score = (
            "score" in str(score_name).lower() and pd.notna(raw_auc) and raw_auc < 0.5
        )
        if reverse_score:
            prepared[report_column] = -numeric

        rows.append(
            {
                "分数字段": score_name,
                "原始AUC": raw_auc,
                "原始方向": _original_direction(score_name, raw_auc),
                "报表计算方向": "已转为坏向分" if reverse_score else "保持坏向分",
                "判断依据": _build_reason(
                    score_name, raw_auc, source_name, reverse_score
                ),
            }
        )
        score_columns[score_name] = report_column

    return prepared, score_columns, pd.DataFrame(rows)


def _resolve_direction_auc(
    frame: pd.DataFrame,
    tag_col: str,
    label_col: str,
    score: pd.Series,
) -> Tuple[float, str]:
    binary_mask = frame[label_col].isin([0, 1]) & pd.notna(score)
    train_mask = binary_mask & frame[tag_col].eq("train")

    if _is_auc_ready(frame.loc[train_mask, label_col], score.loc[train_mask]):
        return (
            calculate_auc(
                frame.loc[train_mask, label_col].astype(int),
                score.loc[train_mask],
            ),
            "train",
        )

    if _is_auc_ready(frame.loc[binary_mask, label_col], score.loc[binary_mask]):
        return (
            calculate_auc(
                frame.loc[binary_mask, label_col].astype(int),
                score.loc[binary_mask],
            ),
            "all",
        )

    return np.nan, "unavailable"


def _is_auc_ready(y_true: pd.Series, score: pd.Series) -> bool:
    return len(y_true) > 0 and y_true.nunique() >= 2 and score.notna().any()


def _original_direction(score_name: str, raw_auc: float) -> str:
    if "score" not in str(score_name).lower():
        return "未判断"
    if pd.isna(raw_auc):
        return "无法判断"
    if raw_auc < 0.5:
        return "高分代表低风险"
    return "高分代表高风险"


def _build_reason(
    score_name: str,
    raw_auc: float,
    source_name: str,
    reverse_score: bool,
) -> str:
    if "score" not in str(score_name).lower():
        return "列名不包含score，未做方向判断"
    if pd.isna(raw_auc):
        return "无可用二分类样本，未做方向判断"

    source_text = "train样本" if source_name == "train" else "全量二分类样本"
    operator = "<" if reverse_score else ">="
    return f"{source_text}原始AUC={raw_auc:.4f} {operator} 0.5000"


def _dedupe(values: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return ordered
