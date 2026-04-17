import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def german_credit_data():
    """
    Load German Credit Data and train a simple model to get realistic scores.
    Returns dictionary with y_true, y_prob (train and test).
    """
    # Locating the file relative to this test file or project root
    # Assuming tests are run from project root, but let's be robust
    base_path = os.path.dirname(os.path.dirname(__file__))  # risk_toolkit/
    data_path = os.path.join(
        base_path,
        "examples",
        "data",
        "statlog+german+credit+data",
        "german.data-numeric",
    )

    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found at {data_path}")

    # Load data
    # german.data-numeric has no headers, whitespace separated.
    # Last column (25th usually, or check doc) is target.
    # Doc says 24 numerical attributes. So 25th column is class.
    df = pd.read_csv(data_path, delim_whitespace=True, header=None)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Target: 1 = Good, 2 = Bad. Remap to 0 = Good, 1 = Bad.
    y = (y == 2).astype(int)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    # Get probabilities
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    return {
        "train": {"y_true": y_train.values, "y_prob": y_prob_train},
        "test": {"y_true": y_test.values, "y_prob": y_prob_test},
        "all": {
            "y_true": y,
            "y_prob": np.concatenate(
                [y_prob_train, y_prob_test]
                # Note: order mixed if not careful
            ),
        },
    }


@pytest.fixture
def report_frame():
    rows = []
    tags = ["train", "test", "oot", "oos"]
    months = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"])
    dims = ["ios", "android"]

    for tag_index, tag in enumerate(tags):
        for month_index, month in enumerate(months):
            for sample_index in range(4):
                score_base = 0.15 + 0.15 * sample_index + 0.03 * month_index
                is_bad = int(sample_index >= 2)
                main_label = -1 if sample_index == 0 and month_index == 0 else is_bad
                alt_label = int(sample_index in (1, 3))
                rows.append(
                    {
                        "tag": tag,
                        "obs_date": month,
                        "score_new": min(score_base + 0.05 * is_bad, 0.99),
                        "score_old_a": min(score_base + 0.02 * is_bad, 0.99),
                        "score_old_b": min(score_base + 0.01 * sample_index, 0.99),
                        "score": 1000 - 1000 * min(score_base + 0.02 * is_bad, 0.99),
                        "label_main": main_label,
                        "label_alt": alt_label,
                        "channel_dim": dims[(tag_index + sample_index) % len(dims)],
                        "profile_income": 5000 + 500 * month_index + 300 * sample_index,
                        "profile_age": 24 + month_index + sample_index,
                        "feature_a": 10 * month_index + sample_index,
                        "feature_b": 100 - 3 * sample_index - month_index,
                    }
                )

    return pd.DataFrame(rows)


class FakeLightGBMBooster:
    def __init__(self):
        self.params = {
            "n_estimators": 80,
            "learning_rate": 0.1,
            "objective": "binary",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 5,
            "reg_lambda": 5,
            "num_leaves": 10,
            "max_depth": -1,
        }
        self._feature_names = ["feature_a", "feature_b"]

    def feature_name(self):
        return list(self._feature_names)

    def feature_importance(self, importance_type="split"):
        if importance_type == "gain":
            return np.array([120.0, 80.0])
        if importance_type == "split":
            return np.array([18.0, 12.0])
        raise ValueError(f"Unsupported importance type: {importance_type}")


class FakeLightGBMModel:
    def __init__(self):
        self.booster_ = FakeLightGBMBooster()

    def get_params(self, deep=True):
        return dict(self.booster_.params)


@pytest.fixture
def fake_lightgbm_model():
    return FakeLightGBMModel()


class FakeXGBoostBooster:
    def __init__(self):
        self.feature_names = ["feature_a", "feature_b"]

    def get_score(self, importance_type="gain"):
        if importance_type == "gain":
            return {"feature_a": 12.0, "feature_b": 8.0}
        if importance_type == "weight":
            return {"feature_a": 18.0, "feature_b": 12.0}
        raise ValueError(f"Unsupported importance type: {importance_type}")


class FakeXGBoostModel:
    def __init__(self):
        self._booster = FakeXGBoostBooster()

    def get_booster(self):
        return self._booster

    def get_params(self, deep=True):
        return {
            "n_estimators": 60,
            "learning_rate": 0.05,
            "objective": "binary:logistic",
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 3,
            "reg_lambda": 4,
            "max_depth": 4,
        }


@pytest.fixture
def fake_xgboost_model():
    return FakeXGBoostModel()


@pytest.fixture
def fake_scorecard_model():
    from newt.modeling.scorecard import Scorecard

    binner = MagicMock()
    binner.rules_ = {
        "feature_a": [10.0],
        "feature_b": [95.0],
    }
    binner._missing_label = "Missing"
    binner.get_splits.side_effect = lambda feature: binner.rules_[feature]
    binner.get_woe_map.side_effect = lambda feature: {
        "feature_a": {"(-inf, 10.0]": -0.4, "(10.0, inf]": 0.6, "Missing": 0.0},
        "feature_b": {"(-inf, 95.0]": -0.2, "(95.0, inf]": 0.3, "Missing": 0.0},
    }.get(feature, {})

    class _FakeResult:
        aic = 123.45
        bic = 130.89
        llf = -58.22
        prsquared = 0.318
        nobs = 512

    class _FakeLogisticModel:
        def __init__(self):
            self.fit_intercept = True
            self.method = "bfgs"
            self.maxiter = 120
            self.regularization = None
            self.alpha = 0.0
            self.extra_kwargs = {"tol": 1e-6}
            self.coefficients_ = pd.DataFrame(
                [
                    {
                        "feature": "const",
                        "coefficient": -1.1,
                        "std_error": 0.2,
                        "z_value": -5.5,
                        "p_value": 3.0e-8,
                        "ci_lower": -1.5,
                        "ci_upper": -0.7,
                        "odds_ratio": float(np.exp(-1.1)),
                    },
                    {
                        "feature": "feature_a",
                        "coefficient": 0.8,
                        "std_error": 0.1,
                        "z_value": 8.0,
                        "p_value": 1.2e-6,
                        "ci_lower": 0.61,
                        "ci_upper": 0.99,
                        "odds_ratio": float(np.exp(0.8)),
                    },
                    {
                        "feature": "feature_b",
                        "coefficient": -0.5,
                        "std_error": 0.15,
                        "z_value": -3.33,
                        "p_value": 0.00086,
                        "ci_lower": -0.79,
                        "ci_upper": -0.21,
                        "odds_ratio": float(np.exp(-0.5)),
                    },
                ]
            )
            self.result_ = _FakeResult()

        def to_dict(self):
            return {
                "intercept": -1.1,
                "coefficients": {
                    "feature_a": 0.8,
                    "feature_b": -0.5,
                },
            }

    model = _FakeLogisticModel()
    return Scorecard().from_model(model, binner)
