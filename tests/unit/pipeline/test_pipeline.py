from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from newt.pipeline.pipeline import ScorecardPipeline


@pytest.fixture
def sample_data():
    X = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
        }
    )
    y = (X["feature1"] + X["feature2"] > 0).astype(int)
    return X, y


def test_pipeline_init(sample_data):
    X, y = sample_data
    pipeline = ScorecardPipeline(X, y)
    assert pipeline.X_train.shape == X.shape
    assert pipeline.y_train.shape == y.shape
    assert pipeline.steps_ == []


@patch("newt.features.binning.Binner")
def test_pipeline_bin(mock_binner_cls, sample_data):
    X, y = sample_data
    pipeline = ScorecardPipeline(X, y)

    # Mock binner instance and transform
    mock_instance = mock_binner_cls.return_value
    mock_instance.transform.return_value = X.copy()

    pipeline = pipeline.bin(method="opt")

    assert pipeline.binner_ is not None
    assert "bin" in pipeline.steps_
    assert pipeline.X_binned_ is not None


@patch("newt.features.binning.Binner")
def test_pipeline_bin_defaults_to_chi(mock_binner_cls, sample_data):
    X, y = sample_data
    pipeline = ScorecardPipeline(X, y)

    mock_instance = mock_binner_cls.return_value
    mock_instance.transform.return_value = X.copy()

    pipeline.bin()

    assert mock_instance.fit.call_args.kwargs["method"] == "chi"


@patch("newt.features.analysis.woe_calculator.WOEEncoder")
def test_pipeline_woe(mock_woe_cls, sample_data):
    X, y = sample_data
    pipeline = ScorecardPipeline(X, y)

    # Manually set binned data to allow woe_transform
    pipeline.X_binned_ = X.copy()

    # Mock WOE encoder instance
    mock_instance = mock_woe_cls.return_value
    mock_instance.transform.return_value = X["feature1"]  # Just return a series

    pipeline = pipeline.woe_transform()

    assert len(pipeline.woe_encoders_) > 0
    assert "woe_transform" in pipeline.steps_
    assert pipeline.X_woe_ is not None


@patch("newt.modeling.logistic.LogisticModel")
def test_pipeline_build_model(mock_model_cls, sample_data):
    X, y = sample_data
    pipeline = ScorecardPipeline(X, y)

    pipeline = pipeline.build_model()

    assert pipeline.model_ is not None
    assert "model" in pipeline.steps_
    mock_model_cls.return_value.fit.assert_called()


def test_pipeline_summary(sample_data):
    X, y = sample_data
    pipeline = ScorecardPipeline(X, y)
    summary = pipeline.summary()
    assert isinstance(summary, dict)
    assert "n_features_initial" in summary


def test_pipeline_real_flow_scores_missing_values():
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, 300),
            "feature2": rng.normal(0, 1, 300),
        }
    )
    X_test = pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, 120),
            "feature2": rng.normal(0, 1, 120),
        }
    )

    train_signal = X_train["feature1"].fillna(0) + 0.6 * X_train["feature2"].fillna(0)
    test_signal = X_test["feature1"].fillna(0) + 0.6 * X_test["feature2"].fillna(0)
    y_train = (train_signal + rng.normal(0, 0.8, len(X_train)) > 0).astype(int)
    y_test = (test_signal + rng.normal(0, 0.8, len(X_test)) > 0).astype(int)

    X_train.loc[[5, 17, 29], "feature1"] = np.nan
    X_test.loc[[3, 11], "feature1"] = np.nan

    pipeline = (
        ScorecardPipeline(X_train, y_train, X_test, y_test)
        .prefilter(iv_threshold=0.0, missing_threshold=1.0, corr_threshold=0.99)
        .bin(method="quantile", n_bins=4)
        .woe_transform()
        .postfilter(psi_threshold=1.0, vif_threshold=50.0)
        .build_model()
        .generate_scorecard()
    )

    scores = pipeline.score(X_test)

    assert scores.notna().all()
    assert "scorecard" in pipeline.steps_
    missing_row = X_test.loc[X_test["feature1"].isna()].iloc[[0]].copy()
    duplicate_missing = pd.concat([missing_row, missing_row], ignore_index=True)
    duplicate_scores = pipeline.score(duplicate_missing)
    assert duplicate_scores.nunique() == 1


def test_generate_scorecard_requires_woe_transform(sample_data):
    X, y = sample_data
    pipeline = ScorecardPipeline(X, y).bin(method="quantile", n_bins=4).build_model()

    with pytest.raises(ValueError, match="woe_transform"):
        pipeline.generate_scorecard()
