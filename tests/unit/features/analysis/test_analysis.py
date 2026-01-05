import pytest
import pandas as pd
import numpy as np
from src.newt.features.analysis.correlation import (
    calculate_correlation_matrix,
    get_high_correlation_pairs,
)
from src.newt.features.analysis.iv_calculator import calculate_iv
from src.newt.features.analysis.woe_calculator import (
    calculate_woe_mapping,
    apply_woe_transform,
    WOEEncoder,
)


@pytest.fixture
def analysis_data():
    np.random.seed(42)
    n = 200
    # Create correlated data
    x1 = np.random.rand(n)
    x2 = x1 * 0.9 + np.random.rand(n) * 0.1  # High correlation
    x3 = np.random.rand(n)  # Random

    # Target related to x1
    # Simple logistic relationship
    prob = 1 / (1 + np.exp(-(x1 - 0.5) * 5))
    target = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target})
    return df


def test_correlation(analysis_data):
    df = analysis_data
    # 1. Matrix
    corr_matrix = calculate_correlation_matrix(df=df.drop(columns="target"))
    # My signature update was calculate_correlation_matrix(df, ...)
    # Wait, previous implementation takes df.

    corr_matrix = calculate_correlation_matrix(df)
    assert not corr_matrix.empty
    assert corr_matrix.shape == (4, 4)
    # x1 and x2 should be typically > 0.8
    assert corr_matrix.loc["x1", "x2"] > 0.8

    # 2. High correlation pairs
    pairs = get_high_correlation_pairs(corr_matrix, threshold=0.8)
    assert len(pairs) >= 1
    # Check structure
    assert "var1" in pairs[0]
    assert "var2" in pairs[0]
    assert "correlation" in pairs[0]

    # Verify values
    found_x1_x2 = False
    for p in pairs:
        vars = {p["var1"], p["var2"]}
        if "x1" in vars and "x2" in vars:
            found_x1_x2 = True
            break
    assert found_x1_x2


def test_iv_calculator(analysis_data):
    df = analysis_data
    # x1 is predictive
    res = calculate_iv(df, target="target", feature="x1", buckets=5)

    iv = res["iv"]
    table = res["woe_table"]

    # IV should be significant (> 0.1 usually for this signal)
    assert iv > 0.02  # Lower bound check
    assert isinstance(table, pd.DataFrame)
    assert not table.empty
    assert "woe" in table.columns
    assert "iv_contribution" in table.columns

    # Check x3 (random) -> Low IV
    res_rand = calculate_iv(df, target="target", feature="x3", buckets=5)
    assert res_rand["iv"] < res["iv"]


def test_woe_calculator(analysis_data):
    df = analysis_data
    # 1. Mapping
    woe_map = calculate_woe_mapping(df, target="target", feature="x1", bins=5)

    assert isinstance(woe_map, dict)
    assert len(woe_map) > 0

    # 2. Transform
    # We need to bin first if we want to use the map directly on raw data?
    # My implementation of calculate_woe_mapping returns map of *bin_label* -> woe.
    # calculate_woe_mapping bins internally. Returns map keyed by Intervals.
    # apply_woe_transform does .map() directly.
    # If the column is numeric x1, it won't have Interval values.
    # So apply_woe_transform as implemented assumes the input column matches the keys.
    # To test meaningful transform, we should bin the column first in test, or check if
    # calculate_woe_mapping returns data we can use.

    # Actually my implementation of apply_woe_transform:
    # "Simplified: assume input is already binned or categorical matches keys exactly."
    # So I must bin the data first to test applies.

    df["x1_bin"] = pd.qcut(df["x1"], q=5, duplicates="drop")
    woe_map_binned = calculate_woe_mapping(
        df, target="target", feature="x1_bin", bins=None
    )

    # Now transform
    # calculate_woe_mapping returns string keys for non-numeric (Intervals).
    # apply_woe_transform requires input column to match keys (strings).
    df["x1_bin_str"] = df["x1_bin"].astype(str)
    df_transformed = apply_woe_transform(
        df, feature="x1_bin_str", woe_map=woe_map_binned
    )

    col_name = "x1_bin_str_woe"
    assert col_name in df_transformed.columns
    assert not df_transformed[col_name].isnull().all()
    # Check values match map
    sample_val = df["x1_bin"].iloc[0]
    expected_woe = woe_map_binned[str(sample_val)]
    # Handle possible float mismatch or if key missing?
    # Should match exactly for categorical/interval objects
    assert abs(df_transformed[col_name].iloc[0] - expected_woe) < 1e-6


def test_woe_encoder_class(analysis_data):
    df = analysis_data
    encoder = WOEEncoder(buckets=5)

    # Test fit
    encoder.fit(df["x1"], df["target"])
    assert encoder.iv_ > 0.02
    assert not encoder.woe_map_ == {}
    assert not encoder.summary_.empty
    assert encoder.bins_ is not None  # x1 is numeric

    # Test transform
    # Use raw numeric column, encoder should handle binning using stored bins
    transformed = encoder.transform(df["x1"])
    assert len(transformed) == len(df)
    assert transformed.dtype == float or np.issubdtype(
        transformed.dtype, np.number
    )

    # Check consistency
    # Transform on same data should yield values present in woe_map_
    # Note: transformation maps bins to values.
    # We can check simple presence
    unique_vals = transformed.unique()
    assert len(unique_vals) <= 6  # 5 bins + potential nan/0 fallback

    # Fit transform
    enc2 = WOEEncoder(buckets=5)
    trans2 = enc2.fit_transform(df["x1"], df["target"])
    assert (transformed == trans2).all()
