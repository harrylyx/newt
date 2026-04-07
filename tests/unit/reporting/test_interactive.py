"""Unit tests for the interactive reporting API."""

import pandas as pd

from newt import (
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
