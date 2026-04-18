import json

import numpy as np
import pandas as pd
import pytest

from newt.modeling.logistic import LogisticModel


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 200),
            "feature2": np.random.normal(0, 1, 200),
        }
    )
    # Add noise to prevent perfect separation
    noise = np.random.logistic(0, 1, 200)
    y = (X["feature1"] + X["feature2"] + noise > 0).astype(int)
    return X, y


def test_logistic_init():
    model = LogisticModel(fit_intercept=False, method="newton")
    assert model.fit_intercept is False
    assert model.method == "newton"


def test_logistic_fit(sample_data):
    X, y = sample_data
    model = LogisticModel()
    model.fit(X, y)

    assert model.is_fitted_
    assert model.model_ is not None
    assert model.result_ is not None
    assert len(model.coefficients_) == 3  # 2 features + intercept
    assert (
        "const" in model.feature_names_
        or "const" in model.coefficients_["feature"].values
    )


def test_logistic_predict(sample_data):
    X, y = sample_data
    model = LogisticModel()
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert len(probs) == len(X)
    assert np.all((probs >= 0) & (probs <= 1))

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_logistic_coefficients(sample_data):
    X, y = sample_data
    model = LogisticModel()
    model.fit(X, y)

    coefs = model.get_coefficients()
    assert isinstance(coefs, pd.DataFrame)
    required_cols = {"feature", "coefficient", "p_value", "odds_ratio"}
    assert required_cols.issubset(coefs.columns)


def test_logistic_to_dict(sample_data):
    X, y = sample_data
    model = LogisticModel()
    model.fit(X, y)

    model_dict = model.to_dict()
    assert "intercept" in model_dict
    assert "coefficients" in model_dict
    assert isinstance(model_dict["coefficients"], dict)
    assert "feature_statistics" in model_dict
    assert "model_statistics" in model_dict
    assert "summary_text" in model_dict
    assert model_dict["schema_version"] == LogisticModel.SERIALIZATION_VERSION


def test_logistic_from_dict_round_trip_prediction(sample_data):
    X, y = sample_data
    model = LogisticModel()
    model.fit(X, y)

    payload = model.to_dict()
    restored = LogisticModel.from_dict(payload)

    original = model.predict_proba(X)
    rebuilt = restored.predict_proba(X)
    np.testing.assert_allclose(original, rebuilt, rtol=1e-9, atol=1e-9)


def test_logistic_dump_load_round_trip(sample_data, tmp_path):
    X, y = sample_data
    model = LogisticModel()
    model.fit(X, y)

    path = tmp_path / "logistic_model.json"
    model.dump(path)
    loaded = LogisticModel.load(path)

    np.testing.assert_allclose(
        model.predict_proba(X),
        loaded.predict_proba(X),
        rtol=1e-9,
        atol=1e-9,
    )


def test_logistic_from_dict_uses_cached_summary_text(sample_data):
    X, y = sample_data
    model = LogisticModel()
    model.fit(X, y)

    payload = model.to_dict()
    restored = LogisticModel.from_dict(payload)

    assert restored.result_ is None
    assert isinstance(restored.summary(), str)
    assert restored.summary() == payload["summary_text"]


def test_logistic_to_dict_does_not_include_training_samples(sample_data):
    X, y = sample_data
    model = LogisticModel()
    model.fit(X, y)

    payload = model.to_dict()
    serialized = json.dumps(payload)

    forbidden_keys = [
        "X_train",
        "y_train",
        "training_data",
        "training_label",
        "sample_weight",
    ]
    for key in forbidden_keys:
        assert key not in payload
        assert key not in serialized


def test_logistic_not_fitted_error():
    model = LogisticModel()
    with pytest.raises(ValueError, match="LogisticModel is not fitted"):
        model.predict(pd.DataFrame({"a": [1]}))


def test_logistic_from_dict_with_minimal_payload_works(sample_data):
    X, _ = sample_data
    payload = {
        "intercept": 0.25,
        "coefficients": {"feature1": 0.7, "feature2": -0.4},
        "feature_names": ["feature1", "feature2"],
    }

    model = LogisticModel.from_dict(payload)
    probs = model.predict_proba(X)

    assert np.all((probs >= 0) & (probs <= 1))
    assert len(probs) == len(X)


def test_logistic_regularization_l1(sample_data):
    X, y = sample_data
    model = LogisticModel(regularization="l1", alpha=0.1)
    model.fit(X, y)
    assert model.is_fitted_


def test_logistic_regularization_l2(sample_data):
    X, y = sample_data
    model = LogisticModel(regularization="l2", alpha=0.1)
    model.fit(X, y)
    assert model.is_fitted_
