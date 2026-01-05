import pytest
import numpy as np
from src.newt.metrics.psi import calculate_psi
from src.newt.metrics.ks import calculate_ks
from src.newt.metrics.lift import calculate_lift_at_k


def test_psi_nan_bucket():
    # Case 1: NaNs in expected and actual. specific pattern.
    # If we treat NaN as bucket, it should contribute to PSI like any other bucket.
    expected = np.array([0.1, 0.1, np.nan, np.nan])
    actual = np.array([0.1, 0.1, np.nan, np.nan])
    # Dist is identical, PSI should be 0.
    psi = calculate_psi(expected, actual, buckets=2)
    assert psi < 1e-6

    # Case 2: NaNs in separate proportions
    # Expected: 50% NaN. Actual: 0% NaN.
    expected = np.array([0.1, np.nan])
    actual = np.array([0.1, 0.1])
    # This should generate significant PSI.
    psi = calculate_psi(expected, actual, buckets=1)
    # buckets=1 for non-nan means 0.1 goes to one bucket. NaN goes to another.
    # Expected: Bucket1(50%), BucketNaN(50%)
    # Actual: Bucket1(100%), BucketNaN(0%)
    # PSI = (1.0 - 0.5)*ln(1.0/0.5) + (0.0 - 0.5)*ln(eps/0.5) ...
    assert psi > 0.1


def test_psi_include_nan_param():
    # Test include_nan=False (drop NaNs)
    expected = np.array([0.1, np.nan])
    actual = np.array([0.1, 0.1])

    # With include_nan=False, we drop NaNs.
    # Expected becomes [0.1]. Actual becomes [0.1, 0.1].
    # Both define range [0.1, 0.1].
    # Expected: 1 item -> 100% in bucket 1.
    # Actual: 2 items -> 100% in bucket 1.
    # PSI should be 0.
    psi_drop = calculate_psi(expected, actual, buckets=1, include_nan=False)
    assert psi_drop < 1e-6

    # Comparison with include_nan=True (default)
    # As tested in test_psi_nan_bucket case 2, this should be > 0.1
    psi_keep = calculate_psi(expected, actual, buckets=1, include_nan=True)
    assert psi_keep > 0.1


def test_ks_single_impl():
    # Verify calculate_ks works as the fast version
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    ks = calculate_ks(y_true, y_prob)
    assert ks == 1.0


def test_lift_at_k():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 60% event rate
    y_prob = np.linspace(0.1, 1.0, 10)  # 0.1 ... 1.0
    # Top 10% (k=0.1) -> top 1 item. y_prob=1.0, y_true=1.
    # Event rate top 10% = 1.0. Global = 0.6. Lift = 1.666...

    lift_10 = calculate_lift_at_k(y_true, y_prob, k=0.1)
    assert abs(lift_10 - (1.0 / 0.6)) < 1e-4

    # Top 50% (k=0.5) -> top 5 items. all are 1.
    lift_50 = calculate_lift_at_k(y_true, y_prob, k=0.5)
    assert abs(lift_50 - (1.0 / 0.6)) < 1e-4

    # Top 100% -> should be 1.0
    lift_100 = calculate_lift_at_k(y_true, y_prob, k=1.0)
    assert abs(lift_100 - 1.0) < 1e-4


def test_lift_at_k_edge_cases():
    y_true = np.array([0, 1])
    y_prob = np.array([0.5, 0.6])

    # k out of range
    with pytest.warns(UserWarning):
        res = calculate_lift_at_k(y_true, y_prob, k=1.5)
    assert np.isnan(res)
