"""Unit tests for tables.py _build_group_metrics behavior."""

import numpy as np
import pandas as pd

from newt.reporting.table_context import FeatureComputationArtifacts
from newt.reporting.tables import _build_group_metrics, build_variable_analysis_sheet


def _make_frame(
    tag_values: list,
    month_values: list,
    label_values: list,
    score_values: list,
    raw_date_values: list,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tag": pd.Categorical(tag_values),
            "month": pd.Categorical(month_values),
            "label": label_values,
            "score": score_values,
            "raw_date": raw_date_values,
        }
    )


class TestBuildGroupMetricsDateColumnExclusion:
    """raw_date_col should only be in the sort when group_cols == [tag_col].

    This was the root cause of a hang: when raw_date_col contains mixed-format
    datetime strings with high cardinality, including it in the sort columns
    of groupby(sort=True) made each groupby O(n log n) string sort — very slow
    on 10M+ rows.
    """

    def test_tag_only_groupby_includes_raw_date(self):
        """Tag-only groupby should include raw_date_col for date-range display."""
        df = _make_frame(
            tag_values=["train", "train", "oot"],
            month_values=["2024-01", "2024-01", "2024-01"],
            label_values=[0, 1, 0],
            score_values=[0.1, 0.9, 0.2],
            raw_date_values=["2024-01-01", "2024-01-15", "2024-02-01"],
        )
        result = _build_group_metrics(
            data=df,
            group_cols=["tag"],
            label_col="label",
            score_col="score",
            tag_col="tag",
            month_col="month",
            raw_date_col="raw_date",
            model_name="test_model",
            metrics_mode="exact",
            build_context=None,
        )
        # Date range should be populated for tag-only groups
        rows_with_date = result[result["观察点月"].notna()]
        assert len(rows_with_date) >= 1

    def test_month_groupby_excludes_raw_date(self):
        """Month-only groupby should NOT include raw_date_col in sort columns."""
        df = _make_frame(
            tag_values=["train", "train", "oot", "oot"],
            month_values=["2024-01", "2024-01", "2024-02", "2024-02"],
            label_values=[0, 1, 0, 1],
            score_values=[0.1, 0.9, 0.2, 0.8],
            raw_date_values=["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"],
        )
        result = _build_group_metrics(
            data=df,
            group_cols=["month"],
            label_col="label",
            score_col="score",
            tag_col="tag",
            month_col="month",
            raw_date_col="raw_date",
            model_name="test_model",
            metrics_mode="exact",
            build_context=None,
        )
        # Month-only groups: result should have one row per unique month
        assert len(result) == 2
        # Should not crash and AUC should be valid for these 2-class groups
        assert result["AUC"].notna().all()

    def test_tag_month_cross_groupby_excludes_raw_date(self):
        """Tag×month groupby should NOT include raw_date_col in sort columns."""
        df = _make_frame(
            tag_values=["train", "train", "oot", "oot"],
            month_values=["2024-01", "2024-01", "2024-01", "2024-01"],
            label_values=[0, 1, 0, 1],
            score_values=[0.1, 0.9, 0.2, 0.8],
            raw_date_values=["2024-01-01", "2024-01-15", "2024-01-20", "2024-01-25"],
        )
        result = _build_group_metrics(
            data=df,
            group_cols=["tag", "month"],
            label_col="label",
            score_col="score",
            tag_col="tag",
            month_col="month",
            raw_date_col="raw_date",
            model_name="test_model",
            metrics_mode="exact",
            build_context=None,
        )
        # Should have exactly 2 groups (train/2024-01, oot/2024-01)
        assert len(result) == 2
        # raw_date_col excluded → no crash
        assert "AUC" in result.columns


class TestBuildGroupMetricsNoPhantomGroup:
    """fillna('') on a Categorical column should not create a phantom '' group."""

    def test_no_phantom_group_when_column_has_no_na(self):
        """Columns without NaN should not produce a phantom '' group."""
        df = pd.DataFrame(
            {
                "tag": ["train", "train", "oot", "oot"],
                "month": ["2024-01", "2024-01", "2024-02", "2024-02"],
                "label": [0, 1, 0, 1],
                "score": [0.1, 0.9, 0.2, 0.8],
                "raw_date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"],
            }
        )
        result = _build_group_metrics(
            data=df,
            group_cols=["tag", "month"],
            label_col="label",
            score_col="score",
            tag_col="tag",
            month_col="month",
            raw_date_col="raw_date",
            model_name="test_model",
            metrics_mode="exact",
            build_context=None,
        )
        # With 4 rows and 2 unique (tag,month) pairs: exactly 2 groups, no phantom
        assert len(result) == 2
        # No group should have 好=0 (no empty groups from phantom categories)
        assert (result["好"] > 0).all()
        assert (result["坏"] > 0).all()

    def test_no_phantom_group_with_categorical_input(self):
        """Categorical columns without NaN should not produce phantom groups."""
        df = _make_frame(
            tag_values=["train", "train", "oot", "oot"],
            month_values=["2024-01", "2024-01", "2024-02", "2024-02"],
            label_values=[0, 1, 0, 1],
            score_values=[0.1, 0.9, 0.2, 0.8],
            raw_date_values=["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"],
        )
        result = _build_group_metrics(
            data=df,
            group_cols=["tag", "month"],
            label_col="label",
            score_col="score",
            tag_col="tag",
            month_col="month",
            raw_date_col="raw_date",
            model_name="test_model",
            metrics_mode="exact",
            build_context=None,
        )
        # Key assertion: no row should have (好=0 AND 坏=0).
        # This confirms observed=True (or phantom fix) worked.
        assert not (
            (result["好"] == 0) & (result["坏"] == 0)
        ).any(), "Found phantom group with 0 samples"

    def test_correct_metrics_with_phantom_group_fix(self):
        """The phantom-group fix should not affect metric correctness."""
        df = pd.DataFrame(
            {
                "tag": ["train", "train", "oot", "oot"],
                "month": ["2024-01", "2024-01", "2024-02", "2024-02"],
                "label": [0, 1, 0, 1],
                "score": [0.1, 0.9, 0.2, 0.8],
                "raw_date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"],
            }
        )
        result = _build_group_metrics(
            data=df,
            group_cols=["tag", "month"],
            label_col="label",
            score_col="score",
            tag_col="tag",
            month_col="month",
            raw_date_col="raw_date",
            model_name="test_model",
            metrics_mode="exact",
            build_context=None,
        )
        assert len(result) == 2
        assert result["AUC"].notna().all()
        assert result["KS"].notna().all()
        # AUC should be 1.0 for these perfectly separated scores
        assert (result["AUC"] == 1.0).all()


class TestBuildGroupMetricsWithNaN:
    """Columns with actual NaN values should still work correctly."""

    def test_groupby_with_actual_nan_in_group_column(self):
        """NaN values in group columns should be filled and grouped correctly."""
        # Use pairs so each group has both classes to assert AUC.notna()
        df = pd.DataFrame(
            {
                "tag": pd.Categorical(
                    ["train", "train", "oot", "oot", None, None, "oot", "oot"]
                ),
                "month": pd.Categorical(
                    [
                        "2024-01",
                        "2024-01",
                        "2024-01",
                        "2024-01",
                        "2024-02",
                        "2024-02",
                        None,
                        None,
                    ]
                ),
                "label": [0, 1, 0, 1, 0, 1, 0, 1],
                "score": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
                "raw_date": [
                    "2024-01-01",
                    "2024-01-15",
                    "2024-01-20",
                    "2024-01-25",
                    "2024-02-01",
                    "2024-02-15",
                    "2024-03-01",
                    "2024-03-15",
                ],
            }
        )
        result = _build_group_metrics(
            data=df,
            group_cols=["tag", "month"],
            label_col="label",
            score_col="score",
            tag_col="tag",
            month_col="month",
            raw_date_col="raw_date",
            model_name="test_model",
            metrics_mode="exact",
            build_context=None,
        )
        # Should have 4 groups (train/2024-01, oot/2024-01, ""/2024-02, oot/"")
        assert len(result) == 4
        # "" group should exist
        assert any(result["样本集"] == "")
        assert any(result["观察点月"] == "")
        # Non-empty groups with both classes should have valid metrics
        assert result["AUC"].notna().all()


def test_scorecard_variable_analysis_does_not_truncate_to_top_30(monkeypatch):
    features = [f"f{i}" for i in range(31)]
    frame = pd.DataFrame(
        {
            "tag": ["train"] * 20 + ["oot"] * 20,
            "month": ["202401"] * 20 + ["202402"] * 20,
            "label": [0, 1] * 20,
        }
    )
    for index, feature in enumerate(features):
        frame[feature] = np.linspace(0, 1, len(frame)) + index

    class _DummyScorecardAdapter:
        model_family = "scorecard"

        def get_importance_table(self):
            return pd.DataFrame(
                {
                    "feature": features,
                    "gain": np.linspace(31, 1, 31),
                    "gain_per": np.full(31, 1 / 31),
                    "weight": np.linspace(31, 1, 31),
                    "weight_per": np.full(31, 1 / 31),
                }
            )

        def get_lr_feature_summary_table(self):
            return pd.DataFrame({"feature": features, "p_value": np.nan})

        def get_lr_model_summary_table(self):
            return pd.DataFrame()

    def _fake_build_feature_analysis_table(**kwargs):
        rows = []
        for idx, feature in enumerate(features, start=1):
            rows.append(
                {
                    "序号": idx,
                    "vars": feature,
                    "gain": float(32 - idx),
                    "gain_per": float(1 / 31),
                    "weight": float(32 - idx),
                    "weight_per": float(1 / 31),
                    "缺失率_train": 0.0,
                    "缺失率_oot": 0.0,
                    "iv_train": 0.1,
                    "iv_oot": 0.1,
                    "ks_train": 0.1,
                    "ks_oot": 0.1,
                    "psi": 0.0,
                    "指标表英文名": feature,
                }
            )
        return pd.DataFrame(rows), FeatureComputationArtifacts()

    monkeypatch.setattr(
        "newt.reporting.tables._build_feature_analysis_table",
        _fake_build_feature_analysis_table,
    )
    monkeypatch.setattr(
        "newt.reporting.tables._build_feature_bin_stats",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "bin": ["(-inf, inf]"],
                "min": [0.0],
                "max": [1.0],
                "total_prop": [1.0],
                "bad_rate": [0.5],
                "iv": [0.0],
                "woe": [0.0],
            }
        ),
    )
    monkeypatch.setattr(
        "newt.reporting.tables._build_feature_monthly_metrics",
        lambda *args, **kwargs: pd.DataFrame(
            {"month": ["202401"], "AUC": [0.5], "KS": [0.0], "PSI": [0.0]}
        ),
    )

    sheet = build_variable_analysis_sheet(
        data=frame,
        tag_col="tag",
        month_col="month",
        primary_label="label",
        feature_cols=features,
        feature_dict=pd.DataFrame(),
        model_adapter=_DummyScorecardAdapter(),
        build_context=None,
    )

    feature_bin_blocks = [b for b in sheet.blocks if b.title.endswith(" 分箱表")]
    assert len(feature_bin_blocks) == 31
