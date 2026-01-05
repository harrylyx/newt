import pytest
import numpy as np
import pandas as pd

from src.newt.metrics.ks import calculate_ks
from src.newt.metrics.auc import calculate_auc
from src.newt.metrics.lift import calculate_lift
from src.newt.metrics.psi import calculate_psi
from src.newt.metrics.gini import calculate_gini


# Use the fixture defined in conftest.py
@pytest.fixture
def data(german_credit_data):
    return german_credit_data["test"]


def test_ks_score(data):
    y_true = data["y_true"]
    y_prob = data["y_prob"]

    ks = calculate_ks(y_true, y_prob)

    assert 0 <= ks <= 1

    # Check that KS is reasonably high for a trained model.
    # The dataset is small, but 0.1 is a low bar
    assert ks > 0.1


def test_auc_score(data):
    y_true = data["y_true"]
    y_prob = data["y_prob"]

    auc = calculate_auc(y_true, y_prob)
    assert 0.5 <= auc <= 1.0

    # Perfect AUC check
    assert calculate_auc(y_true, y_true) == 1.0


def test_gini_score(data):
    y_true = data["y_true"]
    y_prob = data["y_prob"]

    gini = calculate_gini(y_true, y_prob)
    auc = calculate_auc(y_true, y_prob)

    assert abs(gini - (2 * auc - 1)) < 1e-9
    assert -1 <= gini <= 1


def test_lift_table(data):
    y_true = data["y_true"]
    y_prob = data["y_prob"]

    lift_df = calculate_lift(y_true, y_prob, bins=10)

    assert isinstance(lift_df, pd.DataFrame)
    assert len(lift_df) == 10
    assert "lift" in lift_df.columns
    assert "event_rate" in lift_df.columns

    # Top decile (index 0) should generally have lift > 1 for a good model
    assert lift_df.iloc[0]["lift"] > 1.0

    # Check monotonicity of prob (min_prob of bin i >= min_prob of bin i+1) roughly
    # Actually, min_prob is minimum in that bin.
    # Since we sorted by index (highest prob first), min_prob should generally decrease.
    assert lift_df.iloc[0]["min_prob"] >= lift_df.iloc[-1]["min_prob"]


def test_psi_score(german_credit_data):
    # Use train as expected and test as actual
    expected = german_credit_data["train"]["y_prob"]
    actual = german_credit_data["test"]["y_prob"]

    psi = calculate_psi(expected, actual, buckets=10)

    assert psi >= 0
    # Since train and test are from same distribution (random split), PSI should be low
    # Typically < 0.1 is considered stable
    assert psi < 0.25  # relaxed threshold for small sample size volatility


def test_ks_edge_cases():
    # Perfect separation
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert calculate_ks(y_true, y_prob) == 1.0

    # No separation
    y_prob_equal = np.array([0.5, 0.5, 0.5, 0.5])
    # KS might be 0 or undefined behavior depending on implementation details of ties
    # But ks_2samp handles it.
    ks = calculate_ks(y_true, y_prob_equal)
    assert ks >= 0


def test_lift_bins():
    y_true = np.zeros(100)
    y_true[-20:] = (
        1  # 20% event rate (indices 80-99), correlated with high prob
    )
    y_prob = np.linspace(0, 1, 100)  # perfectly correlated

    # With 10 bins, 10 samples per bin.
    # Top 2 bins (80-100) will capture all events if perfectly correlated?
    # Actually y_prob 0.8-1.0 matches indices 80-99.

    lift_df = calculate_lift(y_true, y_prob, bins=5)
    assert len(lift_df) == 5
    # Overall event rate is 0.2
    # Top bin should have event rate 1.0 (if perfect) -> lift = 1.0 / 0.2 = 5
    # Let's see if our lift calc handles simple case
    assert lift_df.iloc[0]["lift"] > 1.0
