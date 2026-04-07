"""Shared sort helpers for report tables and grouped metric frames."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_TAG_ORDER: Dict[str, int] = {"train": 0, "test": 1, "oot": 2, "oos": 3}


def month_sort_key(value: object) -> str:
    """Sort key for month-like values (`YYYYMM` first, fallback to datetime parse)."""
    if pd.isna(value) or value == "":
        return "999999"
    text = str(value).strip()
    if text.isdigit() and len(text) == 6:
        return text
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y%m")
    return f"999998{text}"


def tag_sort_key(
    value: object, tag_order: Optional[Dict[str, int]] = None
) -> Tuple[int, str]:
    """Sort key for sample tags (`train/test/oot/oos` first, then lexical)."""
    text = str(value)
    order_map = tag_order or DEFAULT_TAG_ORDER
    return order_map.get(text.lower(), len(order_map)), text


def ordered_month_values(values: pd.Series) -> List[object]:
    """Return unique month-like values sorted with report semantics."""
    unique_values = pd.Series(values).drop_duplicates().tolist()
    return sorted(unique_values, key=month_sort_key)


def ordered_tag_values(
    values: pd.Series, tag_order: Optional[Dict[str, int]] = None
) -> List[object]:
    """Return unique tag-like values sorted with report semantics."""
    unique_values = pd.Series(values).drop_duplicates().tolist()
    return sorted(unique_values, key=lambda item: tag_sort_key(item, tag_order))


def sort_report_frame(
    frame: pd.DataFrame,
    tag_column: Optional[str] = None,
    month_column: Optional[str] = None,
    leading_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Sort report-like frame with optional tag/month helper ordering."""
    if frame.empty:
        return frame

    ordered = frame.copy()
    sort_columns: List[str] = []
    helper_columns: List[str] = []

    for column in leading_columns or ():
        if column in ordered.columns:
            sort_columns.append(column)

    if tag_column and tag_column in ordered.columns:
        ordered["_tag_order"] = ordered[tag_column].map(tag_sort_key)
        sort_columns.append("_tag_order")
        helper_columns.append("_tag_order")

    if month_column and month_column in ordered.columns:
        ordered["_month_order"] = ordered[month_column].map(month_sort_key)
        sort_columns.append("_month_order")
        helper_columns.append("_month_order")

    if not sort_columns:
        return ordered.reset_index(drop=True)

    ordered = ordered.sort_values(sort_columns, kind="mergesort")
    return ordered.drop(columns=helper_columns, errors="ignore").reset_index(drop=True)
