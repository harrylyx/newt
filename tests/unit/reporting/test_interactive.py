"""Unit tests for the interactive reporting API."""

import numpy as np
import pandas as pd
import pytest

from newt import (
    calculate_bin_metrics,
    calculate_dimensional_comparison,
    calculate_model_comparison,
    calculate_split_metrics,
)


def _build_dummy_data() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "tag": ["train", "train", "oot", "oot", "oot", "train"],
            "obs_date": [
                "2023-01-01",
                "2023-01-15",
                "2023-02-01",
                "2023-02-15",
                "2023-03-01",
                "2023-01-10",
            ],
            "target": [0, 1, 0, 1, 0, 1],
            "score": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            "old_score": [0.15, 0.85, 0.25, 0.75, 0.35, 0.65],
            "channel": ["A", "B", "A", "B", "A", "A"],
            "prin_bal_amount": [10.0, 30.0, 12.0, 24.0, 6.0, 14.0],
            "loan_amount": [100.0, 150.0, 90.0, 120.0, 60.0, 80.0],
        }
    )
    return data


def test_calculate_split_metrics():
    df = _build_dummy_data()
    tag_df, month_df = calculate_split_metrics(
        data=df,
        tag_col="tag",
        date_col="obs_date",
        label_list=["target"],
        score_col="score",
        model_name="new_model",
    )

    # Basic structure checks
    assert not tag_df.empty
    assert not month_df.empty
    assert "样本标签" in tag_df.columns
    assert "样本集" in tag_df.columns
    assert "观察点月" in month_df.columns

    # Value checks
    assert tag_df["AUC"].notna().all()


def test_calculate_split_metrics_supports_score_type_and_amount_metrics():
    df = _build_dummy_data()
    tag_prob, _ = calculate_split_metrics(
        data=df,
        tag_col="tag",
        date_col="obs_date",
        label_list=["target"],
        score_col="score",
        model_name="new_model",
        score_type="probability",
        prin_bal_amount_col="prin_bal_amount",
        loan_amount_col="loan_amount",
    )
    tag_score, _ = calculate_split_metrics(
        data=df,
        tag_col="tag",
        date_col="obs_date",
        label_list=["target"],
        score_col="score",
        model_name="new_model",
        score_type="score",
        prin_bal_amount_col="prin_bal_amount",
        loan_amount_col="loan_amount",
    )

    merged = tag_prob.merge(
        tag_score,
        on=["样本标签", "模型", "样本集", "观察点月"],
        suffixes=("_prob", "_score"),
    )
    assert np.allclose(
        merged["AUC_prob"] + merged["AUC_score"],
        np.ones(len(merged)),
    )
    assert {
        "逾期本金",
        "放款金额",
        "金额坏占比",
        "放款金额占比",
        "逾期本金占比",
        "金额lift",
    }.issubset(tag_prob.columns)


def test_calculate_dimensional_comparison():
    df = _build_dummy_data()
    dim_df = calculate_dimensional_comparison(
        data=df[df["tag"] == "oot"],
        dim_list=["channel"],
        label_list=["target"],
        score_model_columns=[("new_model", "score")],
    )

    assert not dim_df.empty
    assert "维度列" in dim_df.columns
    assert "维度值" in dim_df.columns
    assert "AUC" in dim_df.columns
    assert set(dim_df["维度列"]) == {"channel"}


def test_calculate_dimensional_comparison_supports_amount_metrics():
    df = _build_dummy_data()
    dim_df = calculate_dimensional_comparison(
        data=df[df["tag"] == "oot"],
        dim_list=["channel"],
        label_list=["target"],
        score_model_columns=[("new_model", "score")],
        prin_bal_amount_col="prin_bal_amount",
        loan_amount_col="loan_amount",
    )

    assert {
        "逾期本金",
        "放款金额",
        "金额坏占比",
        "放款金额占比",
        "逾期本金占比",
        "金额lift",
    }.issubset(dim_df.columns)


def test_calculate_model_comparison():
    df = _build_dummy_data()
    compare_df = calculate_model_comparison(
        data=df,
        tag_col="tag",
        date_col="obs_date",
        label_list=["target"],
        model_columns=[("new_model", "score"), ("old_model", "old_score")],
        group_mode="month",
    )

    assert not compare_df.empty
    assert "模型" in compare_df.columns
    assert set(compare_df["模型"]) == {"new_model", "old_model"}
    assert "观察点月" in compare_df.columns


def test_calculate_model_comparison_supports_score_type_and_amount_metrics():
    df = _build_dummy_data()
    compare_prob = calculate_model_comparison(
        data=df,
        tag_col="tag",
        date_col="obs_date",
        label_list=["target"],
        model_columns=[("new_model", "score"), ("old_model", "old_score")],
        group_mode="tag",
        score_type="probability",
        prin_bal_amount_col="prin_bal_amount",
        loan_amount_col="loan_amount",
    )
    compare_score = calculate_model_comparison(
        data=df,
        tag_col="tag",
        date_col="obs_date",
        label_list=["target"],
        model_columns=[("new_model", "score"), ("old_model", "old_score")],
        group_mode="tag",
        score_type="score",
        prin_bal_amount_col="prin_bal_amount",
        loan_amount_col="loan_amount",
    )

    joined = compare_prob.merge(
        compare_score,
        on=["样本标签", "模型", "样本集", "观察点月"],
        suffixes=("_prob", "_score"),
    )
    assert np.allclose(
        joined["AUC_prob"] + joined["AUC_score"],
        np.ones(len(joined)),
    )
    assert "金额lift" in compare_prob.columns


def test_calculate_split_metrics_validates_inputs():
    df = _build_dummy_data()
    with pytest.raises(ValueError, match="score_type must be 'probability' or 'score'"):
        calculate_split_metrics(
            data=df,
            tag_col="tag",
            date_col="obs_date",
            label_list=["target"],
            score_col="score",
            model_name="new_model",
            score_type="unknown",
        )

    with pytest.raises(
        ValueError,
        match="prin_bal_amount_col and loan_amount_col must be provided together",
    ):
        calculate_split_metrics(
            data=df,
            tag_col="tag",
            date_col="obs_date",
            label_list=["target"],
            score_col="score",
            model_name="new_model",
            prin_bal_amount_col="prin_bal_amount",
        )


def test_calculate_bin_metrics_supports_quantile_and_custom_bins_with_amount_metrics():
    df = _build_dummy_data()

    q_table = calculate_bin_metrics(
        data=df,
        label_col="target",
        score_col="score",
        q=3,
        prin_bal_amount_col="prin_bal_amount",
        loan_amount_col="loan_amount",
    )
    custom_table = calculate_bin_metrics(
        data=df,
        label_col="target",
        score_col="score",
        bins=[-np.inf, 0.3, 0.6, np.inf],
        prin_bal_amount_col="prin_bal_amount",
        loan_amount_col="loan_amount",
    )

    assert not q_table.empty
    assert not custom_table.empty
    expected_amount_columns = {
        "逾期本金",
        "放款金额",
        "金额坏占比",
        "放款金额占比",
        "逾期本金占比",
        "金额lift",
    }
    assert expected_amount_columns.issubset(q_table.columns)
    assert expected_amount_columns.issubset(custom_table.columns)


def test_calculate_bin_metrics_validates_inputs():
    df = _build_dummy_data()

    with pytest.raises(
        ValueError,
        match="prin_bal_amount_col and loan_amount_col must be provided together",
    ):
        calculate_bin_metrics(
            data=df,
            label_col="target",
            score_col="score",
            prin_bal_amount_col="prin_bal_amount",
        )

    with pytest.raises(ValueError, match="bins must be strictly increasing"):
        calculate_bin_metrics(
            data=df,
            label_col="target",
            score_col="score",
            bins=[0.0, 0.5, 0.5, 1.0],
        )
