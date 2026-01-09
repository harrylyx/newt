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
    assert "build_model" in pipeline.steps_
    mock_model_cls.return_value.fit.assert_called()


def test_pipeline_summary(sample_data):
    X, y = sample_data
    pipeline = ScorecardPipeline(X, y)
    summary = pipeline.summary()
    assert isinstance(summary, dict)
    assert "n_features_initial" in summary
