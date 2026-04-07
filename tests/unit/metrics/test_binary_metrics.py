import importlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from newt.metrics.binary_metrics import (
    calculate_binary_metrics,
    calculate_binary_metrics_batch,
)
from newt.metrics.psi import calculate_feature_psi_pairs_batch


def _assert_metric_dict_close(left, right, atol=1e-8):
    assert set(left.keys()) == set(right.keys())
    for key in left:
        left_value = float(left[key])
        right_value = float(right[key])
        if np.isnan(left_value) and np.isnan(right_value):
            continue
        assert np.isclose(left_value, right_value, atol=atol), key


def test_calculate_binary_metrics_batch_exact_matches_scalar_python():
    rng = np.random.default_rng(7)
    groups = []
    for size in [320, 480, 640]:
        y = rng.choice([0, 1, -1], size=size, p=[0.44, 0.46, 0.10]).astype(float)
        score = rng.normal(loc=0.0, scale=1.0, size=size)
        score[rng.random(size) < 0.05] = np.nan
        groups.append((y, score))

    batch_metrics = calculate_binary_metrics_batch(
        groups=groups,
        metrics_mode="exact",
        engine="python",
    )
    scalar_metrics = [
        calculate_binary_metrics(
            y_true=y_true,
            y_score=y_score,
            metrics_mode="exact",
        )
        for y_true, y_score in groups
    ]

    for batch_row, scalar_row in zip(batch_metrics, scalar_metrics):
        _assert_metric_dict_close(batch_row, scalar_row, atol=1e-8)


def test_calculate_binary_metrics_binned_close_to_exact():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=8000)
    score = y * 0.7 + rng.normal(loc=0.0, scale=0.7, size=8000)

    exact = calculate_binary_metrics(y_true=y, y_score=score, metrics_mode="exact")
    binned = calculate_binary_metrics(
        y_true=y,
        y_score=score,
        metrics_mode="binned",
        bins=80,
    )

    assert abs(float(exact["AUC"]) - float(binned["AUC"])) <= 0.005
    assert abs(float(exact["KS"]) - float(binned["KS"])) <= 0.005
    for key in ["10%lift", "5%lift", "2%lift", "1%lift"]:
        assert abs(float(exact[key]) - float(binned[key])) <= 0.03


def test_calculate_binary_metrics_batch_rust_matches_python_exact():
    rng = np.random.default_rng(23)
    groups = []
    for size in [500, 900]:
        y = rng.choice([0, 1], size=size, p=[0.55, 0.45]).astype(float)
        score = y * 0.9 + rng.normal(loc=0.0, scale=0.9, size=size)
        groups.append((y, score))

    python_metrics = calculate_binary_metrics_batch(
        groups=groups,
        metrics_mode="exact",
        engine="python",
    )
    rust_metrics = calculate_binary_metrics_batch(
        groups=groups,
        metrics_mode="exact",
        engine="rust",
    )

    for rust_row, python_row in zip(rust_metrics, python_metrics):
        _assert_metric_dict_close(rust_row, python_row, atol=1e-6)


def test_binary_metrics_batch_rust_falls_back_when_extension_missing():
    groups = [
        (
            np.array([0.0, 1.0, 1.0, 0.0, -1.0]),
            np.array([0.1, 0.9, 0.8, 0.3, np.nan]),
        )
    ]

    original_import = importlib.import_module

    def selective_import(name, *args, **kwargs):
        if name in ("newt._newt_iv_rust", "_newt_iv_rust"):
            raise ImportError(f"mocked missing: {name}")
        return original_import(name, *args, **kwargs)

    python_metrics = calculate_binary_metrics_batch(groups=groups, engine="python")
    with patch.object(importlib, "import_module", side_effect=selective_import):
        rust_metrics = calculate_binary_metrics_batch(groups=groups, engine="rust")

    _assert_metric_dict_close(rust_metrics[0], python_metrics[0], atol=1e-8)


def test_feature_psi_pairs_batch_rust_matches_python():
    expected_groups = [
        pd.Series([0.1, 0.2, 0.3, np.nan, 0.8]),
        pd.Series([1.0, 2.0, 2.5, 3.0, np.nan, 4.0]),
        pd.Series([10.0, 10.0, 10.0, np.nan]),
    ]
    actual_groups = [
        pd.Series([0.12, 0.22, np.nan, 0.31, 0.9]),
        pd.Series([1.1, 1.9, 2.6, np.nan, 3.8]),
        pd.Series([10.0, np.nan, 10.0]),
    ]

    python_values = calculate_feature_psi_pairs_batch(
        expected_groups=expected_groups,
        actual_groups=actual_groups,
        engine="python",
    )
    rust_values = calculate_feature_psi_pairs_batch(
        expected_groups=expected_groups,
        actual_groups=actual_groups,
        engine="rust",
    )

    assert np.allclose(rust_values, python_values, equal_nan=True)


def test_feature_psi_pairs_batch_length_mismatch_raises():
    with pytest.raises(ValueError, match="must have the same length"):
        calculate_feature_psi_pairs_batch(
            expected_groups=[pd.Series([0.1, 0.2])],
            actual_groups=[pd.Series([0.2]), pd.Series([0.3])],
        )
