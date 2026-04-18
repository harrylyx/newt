import json
import pickle
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from newt.features.binning import Binner
from newt.modeling.scorecard import Scorecard


class _SerializableLogisticModel:
    def __init__(self):
        self.fit_intercept = True
        self.method = "bfgs"
        self.maxiter = 123
        self.regularization = None
        self.alpha = 0.15
        self.extra_kwargs = {"tol": 1e-6}
        self.coefficients_ = pd.DataFrame()
        self.result_ = None

    def to_dict(self):
        return {"intercept": -0.1, "coefficients": {"x": 0.8}}


@pytest.fixture
def mock_components():
    # Mock Binner
    binner = MagicMock()
    binner._missing_label = "Missing"
    binner.binners_ = {"feature1": MagicMock()}
    binner.binners_["feature1"].splits_ = [0.5]
    binner.rules_ = {"feature1": [0.5]}
    binner.get_splits.return_value = [0.5]
    binner.get_woe_map.return_value = {"bin1": 0.5, "bin2": -0.5}

    # Mock Model
    model = MagicMock()
    model.to_dict.return_value = {"intercept": 0.1, "coefficients": {"feature1": 1.5}}

    return model, binner


def _build_numeric_scorecard() -> Scorecard:
    X = pd.DataFrame({"x": [1, 2, 3, np.nan, 5, 6, np.nan, 8, 9, 10]})
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], name="target")
    binner = Binner()
    binner.fit(X, y, method="quantile", n_bins=2)
    return Scorecard().from_model(
        {"intercept": 0.0, "coefficients": {"x": 1.0}},
        binner,
    )


def test_scorecard_init():
    sc = Scorecard(base_score=600, pdo=50)
    assert sc.base_score == 600
    assert sc.pdo == 50
    assert sc.factor == 50 / np.log(2)


def test_from_model(mock_components):
    model, binner = mock_components
    sc = Scorecard()
    sc.from_model(model, binner)

    assert sc.is_built_
    assert "feature1" in sc.scorecard_
    assert not sc.scorecard_["feature1"].empty
    assert "points" in sc.scorecard_["feature1"].columns
    assert sc.lr_model_ is None
    assert sc._binner is None


def test_scorecard_from_model_preserves_logistic_summary_stats(mock_components):
    _, binner = mock_components

    class _FakeResult:
        aic = 10.5
        bic = 12.1
        llf = -4.8
        prsquared = 0.21
        nobs = 100

    class _FakeLogisticModel:
        def __init__(self):
            self.fit_intercept = True
            self.method = "bfgs"
            self.maxiter = 99
            self.regularization = None
            self.alpha = 0.0
            self.extra_kwargs = {"tol": 1e-6}
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

    sc = Scorecard().from_model(_FakeLogisticModel(), binner)

    assert not sc.feature_statistics_.empty
    assert "p_value" in sc.feature_statistics_.columns
    assert sc.feature_statistics_.set_index("feature").loc[
        "feature1", "p_value"
    ] == pytest.approx(0.0002)
    assert sc.model_statistics_["aic"] == pytest.approx(10.5)
    assert sc.model_statistics_["pseudo_r2"] == pytest.approx(0.21)
    assert sc.lr_parameters_["fit_intercept"] is True
    assert sc.lr_parameters_["method"] == "bfgs"

    restored = Scorecard().from_dict(sc.to_dict())
    assert not restored.feature_statistics_.empty
    assert restored.model_statistics_["bic"] == pytest.approx(12.1)
    assert restored.lr_parameters_["maxiter"] == 99


def test_scorecard_score(mock_components):
    model, binner = mock_components
    sc = Scorecard()
    sc.from_model(model, binner)

    X = pd.DataFrame({"feature1": [0.1, 0.9]})
    scores = sc.score(X)

    assert len(scores) == 2
    assert scores.dtype == float


def test_scorecard_export(mock_components):
    model, binner = mock_components
    sc = Scorecard()
    sc.from_model(model, binner)

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
    model, binner = mock_components
    sc = Scorecard()
    sc.from_model(model, binner)

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
    scorecard.from_model({"intercept": 0.0, "coefficients": {"x": 1.0}}, binner)

    missing_points = scorecard.scorecard_["x"].set_index("bin").loc["Missing", "points"]
    assert missing_points != 0.0

    scores = scorecard.score(X)
    expected_missing_score = scorecard.intercept_points_ + missing_points
    missing_scores = scores[X["x"].isna()]

    assert all(
        score == pytest.approx(expected_missing_score) for score in missing_scores
    )


def test_scorecard_to_dict_includes_binning_rules(mock_components):
    model, binner = mock_components
    sc = Scorecard()
    sc.from_model(model, binner)

    exported = sc.to_dict()

    assert "binning_rules" in exported
    assert exported["binning_rules"]["feature1"]["splits"] == [0.5]


def test_scorecard_round_trip_from_dict_preserves_scores():
    X = pd.DataFrame({"x": [1, 2, 3, np.nan, 5, 6, np.nan, 8, 9, 10]})
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], name="target")

    binner = Binner()
    binner.fit(X, y, method="quantile", n_bins=2)

    scorecard = Scorecard().from_model(
        {"intercept": 0.0, "coefficients": {"x": 1.0}}, binner
    )
    restored = Scorecard().from_dict(scorecard.to_dict())

    pd.testing.assert_series_equal(scorecard.score(X), restored.score(X))


def test_scorecard_pickle_default_does_not_preserve_original_lr_object():
    X = pd.DataFrame({"x": [1, 2, 3, np.nan, 5, 6, np.nan, 8, 9, 10]})
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], name="target")
    binner = Binner()
    binner.fit(X, y, method="quantile", n_bins=2)

    model = _SerializableLogisticModel()
    scorecard = Scorecard().from_model(model, binner)
    restored = pickle.loads(pickle.dumps(scorecard))

    assert restored.lr_model_ is None
    assert restored._binner is None
    assert "coef__x" in restored.lr_parameters_
    assert restored.lr_parameters_["method"] == "bfgs"
    assert restored.lr_parameters_["maxiter"] == 123


def test_scorecard_from_model_can_keep_training_artifacts():
    X = pd.DataFrame({"x": [1, 2, 3, np.nan, 5, 6, np.nan, 8, 9, 10]})
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], name="target")
    binner = Binner()
    binner.fit(X, y, method="quantile", n_bins=2)

    model = _SerializableLogisticModel()
    scorecard = Scorecard().from_model(model, binner, keep_training_artifacts=True)

    assert isinstance(scorecard.lr_model_, _SerializableLogisticModel)
    assert scorecard._binner is binner


def test_scorecard_dump_load_round_trip(tmp_path):
    X = pd.DataFrame({"x": [1, 2, 3, np.nan, 5, 6, np.nan, 8, 9, 10]})
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], name="target")
    binner = Binner()
    binner.fit(X, y, method="quantile", n_bins=2)

    scorecard = Scorecard().from_model(
        {"intercept": 0.0, "coefficients": {"x": 1.0}},
        binner,
    )
    path = tmp_path / "scorecard.json"

    scorecard.dump(path)
    restored = Scorecard.load(path)

    pd.testing.assert_series_equal(scorecard.score(X), restored.score(X))
    assert restored.to_sql() == scorecard.to_sql()


def test_scorecard_payload_contains_lr_snapshot_without_training_data(mock_components):
    model, binner = mock_components
    scorecard = Scorecard().from_model(model, binner)

    payload = scorecard.to_dict()
    assert "lr_snapshot" in payload
    assert "coefficients" in payload["lr_snapshot"]
    assert payload["lr_snapshot"]["coefficients"]["feature1"] == pytest.approx(1.5)

    restored = Scorecard().from_dict(payload)
    assert restored.lr_snapshot_["coefficients"]["feature1"] == pytest.approx(1.5)

    serialized = json.dumps(payload)
    for forbidden in ["X_train", "y_train", "training_data", "training_label"]:
        assert forbidden not in serialized


def test_scorecard_from_model_old_signature_raises_type_error(mock_components):
    model, binner = mock_components
    scorecard = Scorecard()

    with pytest.raises(TypeError):
        scorecard.from_model(
            model,
            binner,
            {"feature1": MagicMock()},
        )


def test_scorecard_to_sql_returns_select_query():
    scorecard = _build_numeric_scorecard()

    sql = scorecard.to_sql()

    assert sql.startswith("SELECT")
    assert "CASE WHEN x IS NULL THEN" in sql
    assert "ELSE 0.0 END" in sql
    assert " AS score" in sql
    assert "FROM input_table" in sql


def test_scorecard_to_sql_includes_feature_breakdown():
    scorecard = _build_numeric_scorecard()

    sql = scorecard.to_sql(
        table_name="credit_input",
        score_alias="final_score",
        include_breakdown=True,
    )

    assert "AS x_points" in sql
    assert " AS final_score" in sql
    assert "FROM credit_input" in sql


def test_scorecard_to_sql_uses_zero_for_missing_fallback_without_missing_bin():
    scorecard = _build_numeric_scorecard()
    payload = scorecard.to_dict()
    payload["features"]["x"] = [
        row for row in payload["features"]["x"] if row["bin"] != "Missing"
    ]

    restored = Scorecard().from_dict(payload)
    sql = restored.to_sql()

    assert "WHEN x IS NULL THEN 0.0" in sql


def test_scorecard_to_sql_after_from_dict():
    scorecard = _build_numeric_scorecard()
    restored = Scorecard().from_dict(scorecard.to_dict())

    sql = restored.to_sql(table_name="scoring_table")

    assert "FROM scoring_table" in sql
    assert " AS score" in sql


def test_scorecard_to_sql_not_built_error():
    scorecard = Scorecard()
    with pytest.raises(ValueError, match="Scorecard is not built"):
        scorecard.to_sql()
