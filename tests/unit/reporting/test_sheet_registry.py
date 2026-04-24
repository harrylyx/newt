"""Tests for report sheet registry and tables facade forwarding."""

from newt.reporting import tables
from newt.reporting.builders import sheet_registry


def test_sheet_registry_resolves_scorecard_sheet_by_name_and_index():
    assert sheet_registry.resolve_sheet_keys(["评分卡计算明细"]) == [
        "scorecard_details"
    ]
    assert sheet_registry.resolve_sheet_keys([5]) == ["scorecard_details"]


def test_tables_facade_uses_registry_consistently():
    selected = tables.resolve_sheet_keys([1, "评分卡计算明细", 5])
    assert selected == ["overview", "scorecard_details"]
    assert selected == sheet_registry.resolve_sheet_keys([1, "评分卡计算明细", 5])
