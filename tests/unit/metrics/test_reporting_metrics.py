import importlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from newt.metrics import (
    calculate_feature_psi_against_base,
    calculate_grouped_psi,
    calculate_psi,
    calculate_psi_batch,
)
from newt.metrics.reporting import (
    build_reference_quantile_bins,
    calculate_bin_performance_table,
    calculate_latest_month_psi,
    summarize_label_distribution,
)


def test_calculate_latest_month_psi_uses_latest_month_as_base():
    data = pd.DataFrame(
        {
            "tag": ["train"] * 9,
            "month": ["202401"] * 3 + ["202402"] * 3 + ["202403"] * 3,
            "score": [0.1, 0.2, 0.3, 0.25, 0.35, 0.45, 0.4, 0.5, 0.6],
        }
    )

    psi_table = calculate_latest_month_psi(
        data=data,
        tag_col="tag",
        month_col="month",
        score_col="score",
    )

    jan_expected = calculate_psi(
        expected=data.loc[data["month"] == "202403", "score"],
        actual=data.loc[data["month"] == "202401", "score"],
    )

    jan_psi = psi_table.loc[psi_table["month"] == "202401", "latest_month_psi"].iloc[0]
    latest_psi = psi_table.loc[psi_table["month"] == "202403", "latest_month_psi"].iloc[
        0
    ]

    assert np.isclose(jan_psi, jan_expected)
    assert latest_psi == 0.0


def test_build_reference_quantile_bins_reuses_train_edges():
    train = pd.Series(np.linspace(0.01, 0.99, 20), name="score")
    oot = pd.Series([0.02, 0.25, 0.49, 0.74, 0.98], name="score")

    edges = build_reference_quantile_bins(train, bins=5)
    oot_bins = pd.cut(
        oot,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )

    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    assert oot_bins.notna().all()
    assert len(edges) >= 3


def test_summarize_label_distribution_counts_grey_good_bad(report_frame):
    distribution = summarize_label_distribution(
        data=report_frame,
        label_col="label_main",
        tag_col="tag",
        month_col="obs_date",
        include_blank_channel=True,
    )

    first_row = distribution.iloc[0]

    assert {"好", "坏", "灰", "总数（去掉灰样本）", "坏占比（去掉灰样本）"}.issubset(
        distribution.columns
    )
    assert first_row["渠道"] == ""
    assert first_row["灰"] >= 0


def test_calculate_bin_performance_table_sorts_by_bin_order_and_recomputes_cumulative():
    data = pd.DataFrame(
        {
            "label": [0, 1, 0, 0, 1, 1],
            "score": [0.10, 0.12, 0.30, 0.32, 0.70, 0.72],
        }
    )
    edges = np.array([-np.inf, 0.2, 0.5, np.inf], dtype=float)

    table = calculate_bin_performance_table(
        data=data,
        label_col="label",
        score_col="score",
        edges=edges,
    )

    assert table["min"].tolist() == [-np.inf, 0.2, 0.5]
    assert np.isclose(table.iloc[0]["cum_bad_rate"], 0.5)
    assert np.isclose(table.iloc[0]["cum_bads_prop"], 1 / 3)
    assert np.isclose(table.iloc[1]["cum_bads_prop"], 1 / 3)
    assert np.isclose(table.iloc[2]["cum_bads_prop"], 1.0)


def test_psi_batch_rust_engine_matches_python_engine():
    reference = pd.Series([0.1, 0.2, 0.2, 0.5, 0.8, np.nan])
    groups = [
        pd.Series([0.1, 0.3, 0.4, np.nan]),
        pd.Series([0.6, 0.8, 0.9, np.nan, np.nan]),
    ]

    python_values = calculate_psi_batch(
        expected=reference,
        actual_groups=groups,
        engine="python",
    )
    rust_values = calculate_psi_batch(
        expected=reference,
        actual_groups=groups,
        engine="rust",
    )

    assert np.allclose(rust_values, python_values, equal_nan=True)


def test_psi_batch_rust_raises_and_auto_falls_back_when_extension_missing():
    reference = pd.Series([0.1, 0.2, 0.2, 0.5, 0.8, np.nan])
    groups = [
        pd.Series([0.1, 0.3, 0.4, np.nan]),
        pd.Series([0.6, 0.8, 0.9, np.nan, np.nan]),
    ]

    original_import = importlib.import_module

    def selective_import(name, *args, **kwargs):
        if name in ("newt._newt_native", "_newt_native"):
            raise ImportError(f"mocked missing: {name}")
        return original_import(name, *args, **kwargs)

    python_values = calculate_psi_batch(
        expected=reference,
        actual_groups=groups,
        engine="python",
    )
    with patch.object(importlib, "import_module", side_effect=selective_import):
        with pytest.raises(ImportError, match="native extension is unavailable"):
            calculate_psi_batch(
                expected=reference,
                actual_groups=groups,
                engine="rust",
            )
        auto_values = calculate_psi_batch(
            expected=reference,
            actual_groups=groups,
            engine="auto",
        )

    assert np.allclose(auto_values, python_values, equal_nan=True)


def test_psi_batch_rust_matches_scalar_calculate_psi_with_exclude_strategy():
    reference = pd.Series([0.15, np.nan, 0.35, 0.55, 0.75])
    groups = [
        pd.Series([0.10, 0.20, np.nan, 0.30]),
        pd.Series([0.50, np.nan, 0.70, 0.80]),
    ]

    rust_values = calculate_psi_batch(
        expected=reference,
        actual_groups=groups,
        nan_strategy="exclude",
        engine="rust",
    )
    scalar_values = [
        calculate_psi(
            expected=reference,
            actual=group,
            nan_strategy="exclude",
        )
        for group in groups
    ]

    assert np.allclose(rust_values, scalar_values, equal_nan=True)


def test_psi_batch_python_matches_scalar_calculate_psi_for_both_nan_strategies():
    reference = pd.Series([0.1, 0.2, np.nan, 0.4, 0.6, 0.9])
    groups = [
        pd.Series([0.12, np.nan, 0.33, 0.44]),
        pd.Series([0.2, 0.3, 0.5, np.nan, 0.8]),
    ]

    for strategy in ["separate", "exclude"]:
        batch_values = calculate_psi_batch(
            expected=reference,
            actual_groups=groups,
            nan_strategy=strategy,
            engine="python",
        )
        scalar_values = [
            calculate_psi(
                expected=reference,
                actual=group,
                nan_strategy=strategy,
            )
            for group in groups
        ]
        assert np.allclose(batch_values, scalar_values, equal_nan=True)


def test_calculate_grouped_psi_supports_latest_reference_with_stats():
    frame = pd.DataFrame(
        {
            "tag": ["train"] * 9 + ["oot"] * 9,
            "month": (["202401"] * 3 + ["202402"] * 3 + ["202403"] * 3) * 2,
            "score": [
                0.10,
                0.20,
                0.30,
                0.25,
                0.35,
                0.45,
                0.40,
                0.50,
                0.60,
                0.15,
                0.25,
                0.35,
                0.20,
                0.30,
                0.40,
                0.45,
                0.55,
                np.nan,
            ],
        }
    )

    result = calculate_grouped_psi(
        data=frame,
        group_cols=["month"],
        score_col="score",
        partition_cols=["tag"],
        reference_mode="latest",
        reference_col="month",
        include_stats=True,
        engine="rust",
    )

    assert {"tag", "month", "psi", "is_reference"}.issubset(result.columns)
    assert {
        "sample_count",
        "missing_count",
        "reference_sample_count",
        "reference_missing_count",
    }.issubset(result.columns)

    latest_rows = result.loc[result["month"] == "202403"]
    assert not latest_rows.empty
    assert latest_rows["is_reference"].all()
    assert np.allclose(latest_rows["psi"], 0.0, equal_nan=True)


def test_calculate_grouped_psi_supports_value_reference():
    frame = pd.DataFrame(
        {
            "tag": ["train", "train", "oot", "oot", "test", "test"],
            "score": [0.1, 0.2, 0.15, 0.25, 0.3, 0.4],
        }
    )

    result = calculate_grouped_psi(
        data=frame,
        group_cols=["tag"],
        score_col="score",
        reference_mode="value",
        reference_col="tag",
        reference_value="train",
        engine="rust",
    )

    train_row = result.loc[result["tag"] == "train"].iloc[0]
    oot_row = result.loc[result["tag"] == "oot"].iloc[0]
    manual_oot = calculate_psi(
        expected=frame.loc[frame["tag"] == "train", "score"],
        actual=frame.loc[frame["tag"] == "oot", "score"],
    )

    assert bool(train_row["is_reference"]) is True
    assert np.isclose(float(train_row["psi"]), 0.0)
    assert np.isclose(float(oot_row["psi"]), manual_oot)


def test_calculate_feature_psi_against_base_supports_month_and_tag_scenarios():
    frame = pd.DataFrame(
        {
            "month": [
                "202401",
                "202401",
                "202402",
                "202402",
                "202403",
                "202403",
            ],
            "tag": ["train", "oot", "train", "oot", "train", "oot"],
            "f1": [0.10, 0.11, 0.30, 0.31, 0.50, 0.51],
            "f2": [1.00, 0.90, 0.70, 0.60, np.nan, 0.40],
        }
    )

    month_result = calculate_feature_psi_against_base(
        data=frame,
        feature_cols=["f1", "f2"],
        base_col="month",
        base_value="202403",
        engine="rust",
    )
    tag_result = calculate_feature_psi_against_base(
        data=frame,
        feature_cols=["f1", "f2"],
        base_col="tag",
        base_value="train",
        compare_values=["train", "oot"],
        engine="rust",
    )

    expected_columns = {
        "feature",
        "base_col",
        "base_value",
        "compare_col",
        "compare_value",
        "psi",
        "is_reference",
    }
    assert expected_columns.issubset(month_result.columns)
    assert expected_columns.issubset(tag_result.columns)

    month_reference = month_result.loc[
        (month_result["compare_col"] == "month")
        & (month_result["compare_value"] == "202403")
    ]
    assert np.allclose(month_reference["psi"], 0.0, equal_nan=True)
    assert month_reference["is_reference"].all()

    tag_reference = tag_result.loc[tag_result["compare_value"] == "train"]
    assert np.allclose(tag_reference["psi"], 0.0, equal_nan=True)
    assert tag_reference["is_reference"].all()


def test_calculate_feature_psi_against_base_matches_manual_loop():
    frame = pd.DataFrame(
        {
            "month": ["202401", "202401", "202402", "202402", "202403", "202403"],
            "f1": [0.10, 0.20, 0.25, 0.35, 0.45, 0.55],
            "f2": [1.0, np.nan, 0.8, 0.7, 0.6, 0.5],
        }
    )

    result = calculate_feature_psi_against_base(
        data=frame,
        feature_cols=["f1", "f2"],
        base_col="month",
        base_value="202403",
        compare_values=["202401", "202402", "202403"],
        engine="rust",
    )

    for feature in ["f1", "f2"]:
        base = frame.loc[frame["month"] == "202403", feature]
        for compared in ["202401", "202402", "202403"]:
            actual = frame.loc[frame["month"] == compared, feature]
            expected_psi = calculate_psi(base, actual)
            observed = result.loc[
                (result["feature"] == feature) & (result["compare_value"] == compared),
                "psi",
            ].iloc[0]
            assert np.isclose(observed, expected_psi, equal_nan=True)
