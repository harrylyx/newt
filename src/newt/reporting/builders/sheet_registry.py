"""Sheet registry and selector resolution for report builders."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd

from newt.reporting.model_adapter import ModelAdapter

MAIN_SHEET_KEY_ORDER = [
    "overview",
    "model_design",
    "variable_analysis",
    "scorecard_details",
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
    5: "scorecard_details",
}

SHEET_NAME_SELECTOR_MAP = {
    "总览": "overview",
    "模型设计": "model_design",
    "变量分析": "variable_analysis",
    "评分卡计算明细": "scorecard_details",
    "模型表现": "model_performance",
    "分维度对比": "dimensional_comparison",
    "新老模型对比": "model_comparison",
    "画像变量": "portrait",
}

MAIN_SHEET_OUTPUT_NAME_MAP = {
    "overview": "总览",
    "model_design": "1.模型设计",
    "variable_analysis": "2.变量分析",
    "scorecard_details": "评分卡计算明细",
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


def resolve_optional_sheet_availability(
    data: pd.DataFrame,
    tag_col: str,
    dim_list: Sequence[str],
    score_list: Sequence[str],
    var_list: Sequence[str],
    model_adapter: ModelAdapter,
) -> Dict[str, bool]:
    """Resolve optional sheet availability under the current report context."""
    has_oot = bool((data[tag_col] == "oot").any())
    return {
        "dimensional_comparison": bool(dim_list) and has_oot,
        "model_comparison": bool(score_list),
        "portrait": bool(var_list) and has_oot,
        "scorecard_details": model_adapter.model_family == "scorecard",
    }


def filter_output_sheet_keys(
    requested_keys: Sequence[str],
    availability: Dict[str, bool],
) -> List[str]:
    """Filter output keys by optional sheet availability."""
    output: List[str] = []
    for key in requested_keys:
        if key in availability and not availability[key]:
            continue
        if key not in output:
            output.append(key)
    return output


def resolve_build_keys(
    output_keys: Sequence[str],
    availability: Dict[str, bool],
) -> List[str]:
    """Resolve all builder keys required to produce requested output sheets."""
    build_keys = set(output_keys)
    if "overview" in output_keys:
        build_keys.add("model_performance")
        for appendix_key in APPENDIX_SHEET_KEY_ORDER:
            if availability.get(appendix_key, False):
                build_keys.add(appendix_key)
    return [key for key in SHEET_KEY_ORDER if key in build_keys]


def resolve_output_sheet_names(output_keys: Sequence[str]) -> Dict[str, str]:
    """Map logical sheet keys to final workbook sheet names."""
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
