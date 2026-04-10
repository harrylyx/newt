from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from newt.features.binning import Binner
from newt.modeling.scorecard import Scorecard


@pytest.fixture
def mock_components():
    # Mock Binner
    binner = MagicMock()
    binner.binners_ = {"feature1": MagicMock()}
    binner.binners_["feature1"].splits_ = [0.5]
    binner.rules_ = {"feature1": [0.5]}
    binner.get_splits.return_value = [0.5]

    # Mock WOE Encoder
    woe_enc_instance = MagicMock()
    woe_enc_instance.woe_map_ = {"bin1": 0.5, "bin2": -0.5}
    woe_encoder = {"feature1": woe_enc_instance}

    # Mock Model
    model = MagicMock()
    model.to_dict.return_value = {"intercept": 0.1, "coefficients": {"feature1": 1.5}}

    return model, binner, woe_encoder


def test_scorecard_init():
    sc = Scorecard(base_score=600, pdo=50)
    assert sc.base_score == 600
    assert sc.pdo == 50
    assert sc.factor == 50 / np.log(2)


def test_from_model(mock_components):
    model, binner, woe_encoder = mock_components
    sc = Scorecard()
    sc.from_model(model, binner, woe_encoder)

    assert sc.is_built_
    assert "feature1" in sc.scorecard_
    assert not sc.scorecard_["feature1"].empty
    assert "points" in sc.scorecard_["feature1"].columns


def test_scorecard_from_model_preserves_logistic_summary_stats(mock_components):
    _, binner, woe_encoder = mock_components

    class _FakeResult:
        aic = 10.5
        bic = 12.1
        llf = -4.8
        prsquared = 0.21
        nobs = 100

    class _FakeLogisticModel:
        def __init__(self):
            self.coefficients_ = pd.DataFrame(
                [
                    {
                        "feature": "const",
                        "coefficient": 0.1,
                        "std_error": 0.2,
                        "z_value": 0.5,
                        "p_value": 0.61,
                        "ci_lower": -0.3,
                        "ci_upper": 0.5,
                        "odds_ratio": float(np.exp(0.1)),
                    },
                    {
                        "feature": "feature1",
                        "coefficient": 1.5,
                        "std_error": 0.4,
                        "z_value": 3.75,
                        "p_value": 0.0002,
                        "ci_lower": 0.7,
                        "ci_upper": 2.3,
                        "odds_ratio": float(np.exp(1.5)),
                    },
                ]
            )
            self.result_ = _FakeResult()

        def to_dict(self):
            return {"intercept": 0.1, "coefficients": {"feature1": 1.5}}

    sc = Scorecard().from_model(_FakeLogisticModel(), binner, woe_encoder)

    assert not sc.feature_statistics_.empty
    assert "p_value" in sc.feature_statistics_.columns
    assert sc.feature_statistics_.set_index("feature").loc[
        "feature1", "p_value"
    ] == pytest.approx(0.0002)
    assert sc.model_statistics_["aic"] == pytest.approx(10.5)
    assert sc.model_statistics_["pseudo_r2"] == pytest.approx(0.21)

    restored = Scorecard().from_dict(sc.to_dict())
    assert not restored.feature_statistics_.empty
    assert restored.model_statistics_["bic"] == pytest.approx(12.1)


def test_scorecard_score(mock_components):
    model, binner, woe_encoder = mock_components
    sc = Scorecard()
    sc.from_model(model, binner, woe_encoder)

    X = pd.DataFrame({"feature1": [0.1, 0.9]})
    scores = sc.score(X)

    assert len(scores) == 2
    assert scores.dtype == float


def test_scorecard_export(mock_components):
    model, binner, woe_encoder = mock_components
    sc = Scorecard()
    sc.from_model(model, binner, woe_encoder)

    df = sc.export()
    assert isinstance(df, pd.DataFrame)
    assert "feature" in df.columns
    assert "points" in df.columns
    assert "Intercept" in df["feature"].values
    assert "feature1" in df["feature"].values


def test_not_built_error():
    sc = Scorecard()
    with pytest.raises(ValueError, match="Scorecard is not built"):
        sc.score(pd.DataFrame({"a": [1]}))


def test_scorecard_summary(mock_components):
    model, binner, woe_encoder = mock_components
    sc = Scorecard()
    sc.from_model(model, binner, woe_encoder)

    s = sc.summary()
    assert isinstance(s, str)
    assert "Scorecard Summary" in s
    assert "Base Score" in s


def test_scorecard_score_uses_missing_bucket_points():
    X = pd.DataFrame({"x": [1, 2, 3, np.nan, 5, 6, np.nan, 8, 9, 10]})
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], name="target")

    binner = Binner()
    binner.fit(X, y, method="quantile", n_bins=2)

    scorecard = Scorecard()
    scorecard.from_model(
        {"intercept": 0.0, "coefficients": {"x": 1.0}},
        binner,
        binner.woe_encoders_,
    )

    missing_points = scorecard.scorecard_["x"].set_index("bin").loc["Missing", "points"]
    assert missing_points != 0.0

    scores = scorecard.score(X)
    expected_missing_score = scorecard.intercept_points_ + missing_points
    missing_scores = scores[X["x"].isna()]

    assert all(
        score == pytest.approx(expected_missing_score) for score in missing_scores
    )


def test_scorecard_to_dict_includes_binning_rules(mock_components):
    model, binner, woe_encoder = mock_components
    sc = Scorecard()
    sc.from_model(model, binner, woe_encoder)

    exported = sc.to_dict()

    assert "binning_rules" in exported
    assert exported["binning_rules"]["feature1"]["splits"] == [0.5]


def test_scorecard_round_trip_from_dict_preserves_scores():
    X = pd.DataFrame({"x": [1, 2, 3, np.nan, 5, 6, np.nan, 8, 9, 10]})
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], name="target")

    binner = Binner()
    binner.fit(X, y, method="quantile", n_bins=2)

    scorecard = Scorecard().from_model(
        {"intercept": 0.0, "coefficients": {"x": 1.0}},
        binner,
        binner.woe_encoders_,
    )
    restored = Scorecard().from_dict(scorecard.to_dict())

    pd.testing.assert_series_equal(scorecard.score(X), restored.score(X))
