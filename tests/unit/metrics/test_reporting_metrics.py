import numpy as np
import pandas as pd

from newt.metrics import calculate_psi
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
