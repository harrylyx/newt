from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


from newt.modeling.scorecard import Scorecard


@pytest.fixture
def mock_components():
    # Mock Binner
    binner = MagicMock()
    binner.binners_ = {"feature1": MagicMock()}
    binner.binners_["feature1"].splits_ = [0.5]

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


def test_scorecard_score(mock_components):
    model, binner, woe_encoder = mock_components
    sc = Scorecard()
    sc.from_model(model, binner, woe_encoder)

    # Mock transform return
    binner.binners_["feature1"].transform.return_value = pd.Series(["bin1", "bin2"])

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
