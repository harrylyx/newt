import numpy as np
import pandas as pd
import pytest

from newt._native import load_native_module
from newt.features.selection import FeatureSelector


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5] * 20,  # Numeric, 100 values
            "x2": ["A", "B"] * 50,  # Low cardinality
            "x3": [1, 2, 3, 4, 5] * 20,  # Perfectly correlated with x1
            "x4": [np.nan] * 100,  # All missing
        }
    )
    # Target: 1 if x1 > 3 else 0.
    # x1 values: 1(0), 2(0), 3(0), 4(1), 5(1).
    # Clearly separable. High IV.
    y = pd.Series([0 if x <= 3 else 1 for x in df["x1"]])
    return df, y


def test_fit_calculates_stats(sample_data):
    X, y = sample_data
    fs = FeatureSelector(metrics=["mean", "missing_rate", "iv"])
    fs.fit(X, y)

    assert fs.is_fitted_
    assert not fs.eda_summary_.empty

    # Check columns
    cols = fs.eda_summary_["feature"].tolist()
    assert set(cols) == {"x1", "x2", "x3", "x4"}

    # Check stats present
    res = fs.eda_summary_.set_index("feature")
    assert "mean" in res.columns
    assert "iv" in res.columns

    # x4 is all missing
    assert res.loc["x4", "missing_rate"] == 1.0


def test_select_filtering(sample_data):
    X, y = sample_data
    fs = FeatureSelector(metrics=["iv", "missing_rate", "correlation"])
    fs.fit(X, y)

    # 1. Test missing rate filtering
    # x4 has 100% missing, should be removed
    # Set corr_threshold=1.01 to disable correlation filtering for this step
    fs.select(missing_threshold=0.5, corr_threshold=1.01)
    print(f"\nMissing filter results: Removed={fs.removed_features_}")
    assert (
        "x4" in fs.removed_features_
    ), f"x4 should be removed (missing=1.0). Removed: {fs.removed_features_}"
    assert (
        "x1" not in fs.removed_features_
    ), f"x1 should not be removed. Removed: {fs.removed_features_}"

    # 2. Test IV filtering
    # x1: High IV (>1.0).
    # x2: A/B vs 0/1.
    # x1 values: 1->0, 2->0, 3->0, 4->1, 5->1.
    # x2 values: A, B. A corresponds to 1, 3, 5... Mixed.
    # With 100 samples, x2 might have non-zero IV but much lower than x1.
    # Set corr_threshold=1.01 to disable correlation filtering for this step
    fs.select(iv_threshold=0.1, corr_threshold=1.01)
    print(f"IV filter results: Removed={fs.removed_features_}")

    # x1 and x3 should be kept (IV high)
    assert "x1" not in fs.removed_features_
    assert "x3" not in fs.removed_features_

    # 3. Test correlation filtering
    # x3 is idential to x1 (corr=1.0).
    fs.select(missing_threshold=0.5, iv_threshold=0.1, corr_threshold=0.99)
    print(f"Final Selection: {fs.selected_features_}")
    print(f"Removed: {fs.removed_features_}")

    selected = fs.selected_features_
    assert "x4" not in selected, "x4 should be filtered (missing)"

    # Check correlation logic
    has_x1 = "x1" in selected
    has_x3 = "x3" in selected

    assert has_x1 or has_x3
    assert not (
        has_x1 and has_x3
    ), "Perfectly correlated features should be deduplicated"


def test_report(sample_data):
    X, y = sample_data
    fs = FeatureSelector()
    fs.fit(X, y)
    fs.select()

    report = fs.report()
    assert "status" in report.columns
    assert "reason" in report.columns
    assert len(report) == 4


def test_select_without_fit_raises_error():
    fs = FeatureSelector()
    with pytest.raises(ValueError, match="not fitted"):
        fs.select()


def test_select_missing_metrics_raises_error(sample_data):
    X, y = sample_data
    # Initialize without IV but with missing_rate (which is checked first)
    fs = FeatureSelector(metrics=["mean", "missing_rate"])
    fs.fit(X, y)

    with pytest.raises(ValueError, match="Metric 'iv' was not calculated"):
        fs.select()


def test_corr_matrix_property(sample_data):
    """Test that corr_matrix property returns feature-to-feature correlations."""
    X, y = sample_data
    fs = FeatureSelector()
    fs.fit(X, y)

    # Access the corr_matrix property
    corr_matrix = fs.corr_matrix

    # Should be a DataFrame
    assert isinstance(corr_matrix, pd.DataFrame)

    # Should only contain numeric columns (x1, x3 - x2 is categorical, x4 is all NaN)
    assert "x1" in corr_matrix.columns
    assert "x3" in corr_matrix.columns

    # x1 and x3 are perfectly correlated
    assert corr_matrix.loc["x1", "x3"] == pytest.approx(1.0)

    # corr_matrix is feature-to-feature, NOT feature-to-target
    # So it should NOT contain target (y) as a column
    assert "y" not in corr_matrix.columns


def test_selector_exposes_analysis_and_selection_results(sample_data):
    X, y = sample_data
    fs = FeatureSelector(metrics=["iv", "missing_rate", "correlation"])
    fs.fit(X, y).select(iv_threshold=0.1, missing_threshold=0.5, corr_threshold=0.99)

    assert hasattr(fs, "analysis_result_")
    assert hasattr(fs, "selection_result_")
    assert fs.analysis_result_.summary.equals(fs.eda_summary_)
    assert fs.selection_result_.selected_features == fs.selected_features_


def test_selector_engine_validation():
    with pytest.raises(ValueError, match="engine must be one of"):
        FeatureSelector(engine="invalid")


def test_selector_auto_falls_back_without_native(monkeypatch, sample_data):
    X, y = sample_data
    monkeypatch.setattr(
        "newt.features.analysis.correlation.load_native_module", lambda: None
    )
    monkeypatch.setattr(
        "newt.features.analysis.batch_iv.load_native_module", lambda: None
    )
    monkeypatch.setattr("newt.metrics.binary_metrics._load_rust_module", lambda: None)

    fs = FeatureSelector(
        engine="auto",
        metrics=["iv", "ks", "lift_10", "missing_rate", "correlation"],
    )
    fs.fit(X, y)

    summary = fs.eda_summary_.set_index("feature")
    assert not summary.empty
    assert pd.notna(summary.loc["x1", "iv"])
    assert pd.notna(summary.loc["x1", "ks"])
    assert pd.notna(summary.loc["x1", "lift_10"])


def test_selector_rust_requires_native(monkeypatch, sample_data):
    X, y = sample_data
    monkeypatch.setattr(
        "newt.features.selection.analyzer.require_native_module",
        lambda: (_ for _ in ()).throw(ImportError("missing native")),
    )
    fs = FeatureSelector(engine="rust", metrics=["iv", "missing_rate", "correlation"])
    with pytest.raises(ImportError, match="missing native"):
        fs.fit(X, y)


@pytest.mark.skipif(
    load_native_module() is None, reason="native extension not available"
)
def test_selector_rust_matches_python_metrics():
    rng = np.random.default_rng(20260422)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = np.where(x1 > 0.3, "A", "B")
    target_prob = 1.0 / (1.0 + np.exp(-(1.5 * x1 - 0.8 * x2)))
    y = pd.Series((rng.random(n) < target_prob).astype(int))

    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x1 * 0.95 + rng.normal(scale=0.02, size=n),
            "cat": cat,
        }
    )

    metrics = ["iv", "ks", "lift_10", "missing_rate", "correlation"]
    fs_python = FeatureSelector(engine="python", metrics=metrics).fit(X, y)
    fs_rust = FeatureSelector(engine="rust", metrics=metrics).fit(X, y)

    py = fs_python.eda_summary_.set_index("feature")
    rust = fs_rust.eda_summary_.set_index("feature")

    for metric, atol in [
        ("iv", 1e-6),
        ("ks", 1e-6),
        ("lift_10", 1e-6),
        ("correlation", 1e-8),
    ]:
        py_values = py[metric].to_numpy(dtype=float)
        rust_values = rust[metric].to_numpy(dtype=float)
        assert np.allclose(py_values, rust_values, atol=atol, equal_nan=True)

    fs_python.select(iv_threshold=0.02, missing_threshold=0.8, corr_threshold=0.8)
    fs_rust.select(iv_threshold=0.02, missing_threshold=0.8, corr_threshold=0.8)
    assert set(fs_python.selected_features_) == set(fs_rust.selected_features_)
