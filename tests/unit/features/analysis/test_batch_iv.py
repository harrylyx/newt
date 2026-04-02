import numpy as np
import pandas as pd

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

    merged = rust_result.merge(python_result, on="feature", suffixes=("_rust", "_python"))

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
