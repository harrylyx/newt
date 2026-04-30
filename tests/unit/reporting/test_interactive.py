"""Unit tests for the interactive reporting API."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, roc_curve

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
    assert np.allclose(
        merged["金额AUC_prob"] + merged["金额AUC_score"],
        np.ones(len(merged)),
    )
    assert np.allclose(
        merged["金额KS_prob"],
        merged["金额KS_score"],
        equal_nan=True,
    )
    expected_prefix = [
        "样本标签",
        "模型",
        "样本集",
        "观察点月",
        "总",
        "好",
        "坏",
        "坏占比",
        "KS",
        "AUC",
        "10%lift",
        "5%lift",
        "2%lift",
        "1%lift",
        "放款金额",
        "逾期本金",
        "金额坏占比",
        "金额AUC",
        "金额KS",
        "10%金额lift",
        "5%金额lift",
        "2%金额lift",
        "1%金额lift",
    ]
    assert tag_prob.columns[: len(expected_prefix)].tolist() == expected_prefix


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

    expected_columns = [
        "维度列",
        "维度值",
        "样本标签",
        "模型",
        "总",
        "好",
        "坏",
        "坏占比",
        "KS",
        "AUC",
        "10%lift",
        "5%lift",
        "2%lift",
        "1%lift",
        "放款金额",
        "逾期本金",
        "金额坏占比",
        "金额AUC",
        "金额KS",
        "10%金额lift",
        "5%金额lift",
        "2%金额lift",
        "1%金额lift",
    ]
    assert dim_df.columns.tolist() == expected_columns


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
    assert np.allclose(
        joined["金额AUC_prob"] + joined["金额AUC_score"],
        np.ones(len(joined)),
    )
    assert np.allclose(
        joined["金额KS_prob"],
        joined["金额KS_score"],
        equal_nan=True,
    )
    expected_prefix = [
        "样本标签",
        "模型",
        "样本集",
        "观察点月",
        "总",
        "好",
        "坏",
        "坏占比",
        "KS",
        "AUC",
        "10%lift",
        "5%lift",
        "2%lift",
        "1%lift",
        "放款金额",
        "逾期本金",
        "金额坏占比",
        "金额AUC",
        "金额KS",
        "10%金额lift",
        "5%金额lift",
        "2%金额lift",
        "1%金额lift",
    ]
    assert compare_prob.columns[: len(expected_prefix)].tolist() == expected_prefix


def test_calculate_model_comparison_uses_positive_score_intersection():
    df = _build_dummy_data()
    df.loc[1, "old_score"] = np.nan
    df.loc[2, "old_score"] = 0.0
    df.loc[4, "old_score"] = -0.1

    compare_df = calculate_model_comparison(
        data=df,
        tag_col="tag",
        date_col="obs_date",
        label_list=["target"],
        model_columns=[("new_model", "score"), ("old_model", "old_score")],
        group_mode="tag",
    )

    new_rows = compare_df.loc[compare_df["模型"] == "new_model"].set_index("样本集")
    old_rows = compare_df.loc[compare_df["模型"] == "old_model"].set_index("样本集")
    assert (new_rows["总"] == old_rows["总"]).all()

    score = pd.to_numeric(df["score"], errors="coerce")
    old_score = pd.to_numeric(df["old_score"], errors="coerce")
    intersection_mask = (
        score.notna()
        & old_score.notna()
        & np.isfinite(score.to_numpy(dtype=float))
        & np.isfinite(old_score.to_numpy(dtype=float))
        & score.gt(0)
        & old_score.gt(0)
    )
    expected_counts = df.loc[intersection_mask].groupby("tag").size()
    for tag_value, expected_total in expected_counts.items():
        assert int(new_rows.loc[tag_value, "总"]) == int(expected_total)
        assert int(old_rows.loc[tag_value, "总"]) == int(expected_total)


def test_calculate_split_metrics_amount_auc_and_ks_are_weighted():
    df = pd.DataFrame(
        {
            "tag": ["train", "train", "train", "train"],
            "obs_date": ["2023-01-01"] * 4,
            "target": [1, 0, 1, 0],
            "score": [0.9, 0.8, 0.7, 0.1],
            "prin_bal_amount": [50.0, 1.0, 20.0, 1.0],
            "loan_amount": [100.0, 1.0, 1.0, 100.0],
        }
    )
    tag_df, _ = calculate_split_metrics(
        data=df,
        tag_col="tag",
        date_col="obs_date",
        label_list=["target"],
        score_col="score",
        model_name="new_model",
        prin_bal_amount_col="prin_bal_amount",
        loan_amount_col="loan_amount",
    )

    row = tag_df.loc[tag_df["样本集"] == "train"].iloc[0]
    expected_weighted_auc = float(
        roc_auc_score(df["target"], df["score"], sample_weight=df["loan_amount"])
    )
    fpr, tpr, _ = roc_curve(
        df["target"].to_numpy(),
        df["score"].to_numpy(),
        sample_weight=df["loan_amount"].to_numpy(),
    )
    expected_weighted_ks = float(np.max(np.abs(tpr - fpr)))

    assert float(row["金额AUC"]) == pytest.approx(expected_weighted_auc, rel=1e-9)
    assert float(row["金额KS"]) == pytest.approx(expected_weighted_ks, rel=1e-9)


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
    expected_sample_columns = {"total_prop", "goods_prop", "bads_prop"}
    assert expected_sample_columns.issubset(q_table.columns)
    assert expected_sample_columns.issubset(custom_table.columns)
    assert q_table["total_prop"].sum() == pytest.approx(1.0)
    assert q_table["goods_prop"].sum() == pytest.approx(1.0)
    assert q_table["bads_prop"].sum() == pytest.approx(1.0)
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
