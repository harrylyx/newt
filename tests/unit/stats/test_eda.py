"""
Unit tests for EDAAnalyzer.
"""

import numpy as np
import pandas as pd
import pytest

from src.newt.statistics import ALL_METRICS, BASIC_METRICS, LABEL_METRICS, EDAAnalyzer


class TestEDAAnalyzerBasics:
    """Test basic EDAAnalyzer functionality."""

    def test_init_default_metrics(self):
        """Test default initialization includes all metrics."""
        analyzer = EDAAnalyzer()
        assert analyzer.metrics == set(ALL_METRICS)

    def test_init_custom_metrics(self):
        """Test initialization with custom metrics."""
        analyzer = EDAAnalyzer(metrics=["mean", "std", "missing_rate"])
        assert analyzer.metrics == {"mean", "std", "missing_rate"}

    def test_init_invalid_metric(self):
        """Test that invalid metrics raise ValueError."""
        with pytest.raises(ValueError, match="Invalid metrics"):
            EDAAnalyzer(metrics=["mean", "invalid_metric"])

    def test_available_metrics(self):
        """Test static method returns all available metrics."""
        metrics = EDAAnalyzer.available_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) == len(ALL_METRICS)

    def test_basic_metrics(self):
        """Test basic metrics static method."""
        metrics = EDAAnalyzer.basic_metrics()
        assert set(metrics) == set(BASIC_METRICS)

    def test_label_metrics(self):
        """Test label metrics static method."""
        metrics = EDAAnalyzer.label_metrics()
        assert set(metrics) == set(LABEL_METRICS)


class TestEDAAnalyzerNumeric:
    """Test EDAAnalyzer with numeric data."""

    @pytest.fixture
    def numeric_series(self):
        """Create a simple numeric series."""
        np.random.seed(42)
        return pd.Series(np.random.randn(100))

    @pytest.fixture
    def binary_target(self):
        """Create a binary target."""
        np.random.seed(42)
        return pd.Series(np.random.randint(0, 2, 100))

    def test_basic_stats(self, numeric_series):
        """Test basic statistics computation."""
        analyzer = EDAAnalyzer(metrics=["mean", "median", "std", "variance"])
        result = analyzer.analyze(numeric_series)

        assert "mean" in result
        assert "median" in result
        assert "std" in result
        assert "variance" in result
        assert result["dtype"] == "numeric"

        # Verify values against pandas
        assert abs(result["mean"] - numeric_series.mean()) < 1e-10
        assert abs(result["median"] - numeric_series.median()) < 1e-10
        assert abs(result["std"] - numeric_series.std()) < 1e-10

    def test_quartiles(self, numeric_series):
        """Test quartile computation."""
        analyzer = EDAAnalyzer(metrics=["q1", "q3"])
        result = analyzer.analyze(numeric_series)

        assert abs(result["q1"] - numeric_series.quantile(0.25)) < 1e-10
        assert abs(result["q3"] - numeric_series.quantile(0.75)) < 1e-10

    def test_skewness_kurtosis(self, numeric_series):
        """Test skewness and kurtosis."""
        analyzer = EDAAnalyzer(metrics=["skewness", "kurtosis"])
        result = analyzer.analyze(numeric_series)

        assert "skewness" in result
        assert "kurtosis" in result
        # Just verify they are finite numbers
        assert np.isfinite(result["skewness"])
        assert np.isfinite(result["kurtosis"])

    def test_missing_rate(self):
        """Test missing rate calculation."""
        series = pd.Series([1, 2, np.nan, 4, np.nan])
        analyzer = EDAAnalyzer(metrics=["missing_rate"])
        result = analyzer.analyze(series)

        assert result["missing_rate"] == 0.4  # 2 out of 5

    def test_unique_values(self, numeric_series):
        """Test unique value statistics."""
        analyzer = EDAAnalyzer(metrics=["unique_count", "unique_ratio"])
        result = analyzer.analyze(numeric_series)

        assert result["unique_count"] == numeric_series.nunique()
        assert result["unique_ratio"] == numeric_series.nunique() / len(numeric_series)

    def test_mode_ratio(self):
        """Test mode ratio calculation."""
        series = pd.Series([1, 1, 1, 2, 2, 3])
        analyzer = EDAAnalyzer(metrics=["mode_ratio"])
        result = analyzer.analyze(series)

        assert result["mode_ratio"] == 0.5  # 3 out of 6
        assert result["mode_value"] == 1

    def test_nonzero_ratio(self):
        """Test non-zero ratio calculation."""
        series = pd.Series([0, 0, 1, 2, 3])
        analyzer = EDAAnalyzer(metrics=["nonzero_ratio"])
        result = analyzer.analyze(series)

        assert result["nonzero_ratio"] == 0.6  # 3 out of 5

    def test_outlier_detection(self, numeric_series):
        """Test outlier detection."""
        # Add some outliers
        series = pd.concat([numeric_series, pd.Series([100, -100])])
        analyzer = EDAAnalyzer()
        result = analyzer.analyze(series)

        assert "outlier_count" in result
        assert "outlier_ratio" in result
        assert result["outlier_count"] >= 2  # At least the two we added

    def test_label_metrics(self, numeric_series, binary_target):
        """Test metrics that require a label."""
        analyzer = EDAAnalyzer(metrics=["correlation", "ks", "iv", "lift_10"])
        result = analyzer.analyze(numeric_series, y=binary_target)

        assert "correlation" in result
        assert "ks" in result
        assert "iv" in result
        assert "lift_10" in result

        # KS and IV should be between 0 and some reasonable max
        assert 0 <= result["ks"] <= 1
        assert result["iv"] >= 0

    def test_no_label_returns_nan(self, numeric_series):
        """Test that label metrics return NaN when no label provided."""
        analyzer = EDAAnalyzer(metrics=["correlation", "ks", "iv"])
        result = analyzer.analyze(numeric_series, y=None)

        assert np.isnan(result["correlation"])
        assert np.isnan(result["ks"])
        assert np.isnan(result["iv"])


class TestEDAAnalyzerCategorical:
    """Test EDAAnalyzer with categorical data."""

    @pytest.fixture
    def categorical_series(self):
        """Create a categorical series."""
        return pd.Series(["A", "B", "A", "C", "B", "A", "B", "C", "A", "A"])

    def test_categorical_dtype(self, categorical_series):
        """Test that categorical data is detected correctly."""
        analyzer = EDAAnalyzer()
        result = analyzer.analyze(categorical_series)

        assert result["dtype"] == "categorical"

    def test_categorical_unique_values(self, categorical_series):
        """Test unique values for categorical."""
        analyzer = EDAAnalyzer(metrics=["unique_count", "unique_ratio"])
        result = analyzer.analyze(categorical_series)

        assert result["unique_count"] == 3  # A, B, C
        assert result["unique_ratio"] == 0.3  # 3 out of 10

    def test_categorical_mode_ratio(self, categorical_series):
        """Test mode ratio for categorical."""
        analyzer = EDAAnalyzer(metrics=["mode_ratio"])
        result = analyzer.analyze(categorical_series)

        assert result["mode_ratio"] == 0.5  # A appears 5 times out of 10
        assert result["mode_value"] == "A"

    def test_categorical_numeric_metrics_nan(self, categorical_series):
        """Test that numeric metrics return NaN for categorical data."""
        analyzer = EDAAnalyzer(metrics=["mean", "std", "skewness"])
        result = analyzer.analyze(categorical_series)

        assert np.isnan(result["mean"])
        assert np.isnan(result["std"])
        assert np.isnan(result["skewness"])


class TestEDAAnalyzerDataFrame:
    """Test EDAAnalyzer DataFrame analysis."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame with multiple features."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "num_feature1": np.random.randn(100),
                "num_feature2": np.random.randint(0, 10, 100),
                "cat_feature": np.random.choice(["A", "B", "C"], 100),
                "target": np.random.randint(0, 2, 100),
            }
        )

    def test_analyze_dataframe_all_features(self, sample_dataframe):
        """Test analyzing all non-target features."""
        analyzer = EDAAnalyzer(metrics=["mean", "unique_count", "iv"])
        result = analyzer.analyze_dataframe(sample_dataframe, target="target")

        assert len(result) == 3  # 3 non-target features
        assert "feature" in result.columns
        assert list(result["feature"]) == [
            "num_feature1",
            "num_feature2",
            "cat_feature",
        ]

    def test_analyze_dataframe_selected_features(self, sample_dataframe):
        """Test analyzing selected features only."""
        analyzer = EDAAnalyzer(metrics=["mean", "std"])
        result = analyzer.analyze_dataframe(
            sample_dataframe, target="target", features=["num_feature1"]
        )

        assert len(result) == 1
        assert result.iloc[0]["feature"] == "num_feature1"

    def test_analyze_dataframe_no_target(self, sample_dataframe):
        """Test analyzing without target column."""
        analyzer = EDAAnalyzer(metrics=["mean", "unique_count"])
        result = analyzer.analyze_dataframe(sample_dataframe)

        assert len(result) == 4  # All columns

    def test_summary_method(self, sample_dataframe):
        """Test summary method returns last results."""
        analyzer = EDAAnalyzer(metrics=["mean", "std"])
        analyzer.analyze_dataframe(sample_dataframe, target="target")

        summary = analyzer.summary()
        assert len(summary) == 3


class TestEDAAnalyzerEdgeCases:
    """Test edge cases."""

    def test_empty_series(self):
        """Test handling of empty series."""
        analyzer = EDAAnalyzer(metrics=["mean", "missing_rate"])
        result = analyzer.analyze(pd.Series([], dtype=float))

        assert result["count"] == 0

    def test_constant_series(self):
        """Test handling of constant values."""
        series = pd.Series([5, 5, 5, 5, 5])
        analyzer = EDAAnalyzer(metrics=["mean", "std", "variance", "unique_count"])
        result = analyzer.analyze(series)

        assert result["mean"] == 5.0
        assert result["std"] == 0.0
        assert result["variance"] == 0.0
        assert result["unique_count"] == 1

    def test_all_missing(self):
        """Test handling of all missing values."""
        series = pd.Series([np.nan, np.nan, np.nan])
        analyzer = EDAAnalyzer(metrics=["mean", "missing_rate"])
        result = analyzer.analyze(series)

        assert result["missing_rate"] == 1.0

    def test_series_with_name(self):
        """Test that series name is preserved."""
        series = pd.Series([1, 2, 3], name="my_feature")
        analyzer = EDAAnalyzer(metrics=["mean"])
        result = analyzer.analyze(series)

        assert result["feature"] == "my_feature"

    def test_custom_feature_name(self):
        """Test custom feature name override."""
        series = pd.Series([1, 2, 3], name="original_name")
        analyzer = EDAAnalyzer(metrics=["mean"])
        result = analyzer.analyze(series, feature_name="custom_name")

        assert result["feature"] == "custom_name"


class TestEDAAnalyzerWithRealData:
    """Integration tests with German Credit Data."""

    def test_with_german_credit(self, german_credit_data):
        """Test with real credit data."""
        y_true = german_credit_data["test"]["y_true"]
        y_prob = german_credit_data["test"]["y_prob"]

        # Create a simple DataFrame
        df = pd.DataFrame({"score": y_prob, "target": y_true})

        analyzer = EDAAnalyzer()
        result = analyzer.analyze_dataframe(df, target="target")

        assert len(result) == 1
        assert result.iloc[0]["feature"] == "score"

        # Check that metrics were computed
        score_result = result.iloc[0]
        assert 0 <= score_result["ks"] <= 1
        assert score_result["iv"] >= 0
        assert 0 <= score_result["mean"] <= 1
