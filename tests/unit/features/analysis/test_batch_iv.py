import numpy as np
import pandas as pd
import pytest

import newt.features.analysis.batch_iv as batch_iv_module
from newt._native import load_native_module
from newt.features.analysis.batch_iv import calculate_batch_iv


def test_calculate_batch_iv_rust_matches_python():
    data = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
            "x2": [1, np.nan, 1, np.nan, 2, 2, 3, 3],
            "x3": [5, 5, 5, 5, 5, 5, 5, 5],
        }
    )
    y = pd.Series([0, 0, 0, 1, 0, 1, 1, 1], name="target")

    rust_result = calculate_batch_iv(data, y, engine="rust", bins=4)
    python_result = calculate_batch_iv(data, y, engine="python", bins=4)

    merged = rust_result.merge(
        python_result, on="feature", suffixes=("_rust", "_python")
    )

    assert np.allclose(merged["iv_rust"], merged["iv_python"], atol=1e-6)


def test_calculate_batch_iv_handles_missing_and_constant_columns():
    data = pd.DataFrame(
        {
            "all_missing": [np.nan, np.nan, np.nan, np.nan],
            "constant": [1, 1, 1, 1],
            "signal": [0.1, 0.2, 0.8, 0.9],
        }
    )
    y = pd.Series([0, 0, 1, 1], name="target")

    result = calculate_batch_iv(data, y, engine="python", bins=2).set_index("feature")

    assert result.loc["all_missing", "iv"] == 0.0
    assert result.loc["constant", "iv"] == 0.0
    assert result.loc["signal", "iv"] > 0.0


def test_calculate_batch_iv_supports_mixed_dtypes():
    data = pd.DataFrame(
        {
            "num_signal": [0.1, 0.2, 0.3, 0.9, 1.1, 1.2],
            "cat_signal": ["A", "A", "B", "B", "C", "C"],
            "cat_missing": ["x", None, "x", "y", None, "y"],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1], name="target")

    python_result = calculate_batch_iv(data, y, engine="python", bins=3).set_index(
        "feature"
    )
    auto_result = calculate_batch_iv(data, y, engine="auto", bins=3).set_index(
        "feature"
    )

    assert python_result.loc["num_signal", "iv"] > 0
    assert python_result.loc["cat_signal", "iv"] > 0
    assert python_result.loc["cat_missing", "iv"] >= 0
    assert np.allclose(
        auto_result["iv"].to_numpy(),
        python_result["iv"].to_numpy(),
        atol=1e-6,
        equal_nan=True,
    )


def test_calculate_batch_iv_rust_mode_requires_native(monkeypatch):
    monkeypatch.setattr(
        "newt.features.analysis.batch_iv.require_native_module", lambda: None
    )
    data = pd.DataFrame({"x": [1, 2, 3], "y": [3, 4, 5]})
    target = pd.Series([0, 1, 0])

    with pytest.raises(ImportError):
        calculate_batch_iv(data, target, engine="rust")


@pytest.mark.skipif(
    load_native_module() is None, reason="native extension not available"
)
def test_calculate_batch_iv_rust_matches_python_for_mixed_dtypes():
    data = pd.DataFrame(
        {
            "num1": [0.0, 0.1, 0.2, 0.7, 0.8, 0.9, 1.2, 1.3],
            "num2": [1, 1, 1, 2, 2, 2, 3, 3],
            "cat1": ["a", "a", "b", "b", "c", "c", None, "d"],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 0], name="target")

    rust_result = calculate_batch_iv(data, y, engine="rust", bins=4)
    python_result = calculate_batch_iv(data, y, engine="python", bins=4)
    merged = rust_result.merge(
        python_result, on="feature", suffixes=("_rust", "_python")
    )
    assert np.allclose(merged["iv_rust"], merged["iv_python"], atol=1e-6)


def test_calculate_batch_iv_default_auto_falls_back_when_native_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(batch_iv_module, "load_native_module", lambda: None)
    data = pd.DataFrame(
        {
            "num": [0.1, 0.2, 0.3, 0.9, 1.0, 1.2],
            "cat": ["A", "A", "B", "B", "C", None],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1], name="target")

    default_result = calculate_batch_iv(data, y, bins=3).set_index("feature")
    python_result = calculate_batch_iv(data, y, bins=3, engine="python").set_index(
        "feature"
    )
    assert np.allclose(
        default_result["iv"].to_numpy(),
        python_result["iv"].to_numpy(),
        atol=1e-9,
        equal_nan=True,
    )
