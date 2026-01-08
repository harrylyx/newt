"""Unit tests for StepwiseSelector."""

import numpy as np
import pandas as pd
import pytest

from newt.features.selection.stepwise import StepwiseSelector


@pytest.fixture
def stepwise_data():
    """Create test data with predictive features."""
    np.random.seed(42)
    n = 500

    # Create features with different predictive power
    x1 = np.random.randn(n)  # Strong predictor
    x2 = np.random.randn(n)  # Moderate predictor
    x3 = np.random.randn(n)  # Weak predictor
    x4 = np.random.randn(n)  # Noise
    x5 = x1 * 0.8 + np.random.randn(n) * 0.2  # Correlated with x1

    # Target based on x1 and x2
    logit = 0.8 * x1 + 0.4 * x2 + 0.1 * x3 + np.random.randn(n) * 0.3
    prob = 1 / (1 + np.exp(-logit))
    y = (np.random.rand(n) < prob).astype(int)

    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})
    y = pd.Series(y, name="target")

    return X, y


class TestStepwiseSelectorInit:
    """Test StepwiseSelector initialization."""

    def test_default_init(self):
        """Test default initialization."""
        selector = StepwiseSelector()
        assert selector.direction == "both"
        assert selector.criterion == "aic"
        assert selector.p_enter == 0.05
        assert selector.p_remove == 0.10
        assert not selector.is_fitted_

    def test_custom_init(self):
        """Test custom initialization."""
        selector = StepwiseSelector(
            direction="forward",
            criterion="bic",
            p_enter=0.01,
            p_remove=0.05,
            exclude=["x1"],
        )
        assert selector.direction == "forward"
        assert selector.criterion == "bic"
        assert selector.p_enter == 0.01
        assert selector.p_remove == 0.05
        assert selector.exclude == ["x1"]

    def test_invalid_direction(self):
        """Test invalid direction raises error."""
        with pytest.raises(ValueError):
            StepwiseSelector(direction="invalid")

    def test_invalid_criterion(self):
        """Test invalid criterion raises error."""
        with pytest.raises(ValueError):
            StepwiseSelector(criterion="invalid")


class TestStepwiseSelectorFit:
    """Test StepwiseSelector fit methods."""

    def test_forward_selection(self, stepwise_data):
        """Test forward selection."""
        X, y = stepwise_data
        selector = StepwiseSelector(direction="forward", criterion="aic")
        selector.fit(X, y)

        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0
        assert len(selector.selected_features_) <= len(X.columns)
        # x1 should likely be selected as strongest predictor
        assert "x1" in selector.selected_features_

    def test_backward_elimination(self, stepwise_data):
        """Test backward elimination."""
        X, y = stepwise_data
        selector = StepwiseSelector(direction="backward", criterion="aic")
        selector.fit(X, y)

        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0

    def test_bidirectional_selection(self, stepwise_data):
        """Test bidirectional stepwise selection."""
        X, y = stepwise_data
        selector = StepwiseSelector(direction="both", criterion="aic")
        selector.fit(X, y)

        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0
        # x1 should be in selection (strongest predictor)
        assert "x1" in selector.selected_features_

    def test_pvalue_criterion(self, stepwise_data):
        """Test p-value based selection."""
        X, y = stepwise_data
        selector = StepwiseSelector(direction="both", criterion="pvalue")
        selector.fit(X, y)

        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0

    def test_bic_criterion(self, stepwise_data):
        """Test BIC criterion selection."""
        X, y = stepwise_data
        selector = StepwiseSelector(direction="forward", criterion="bic")
        selector.fit(X, y)

        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0

    def test_exclude_features(self, stepwise_data):
        """Test excluding features (force include)."""
        X, y = stepwise_data
        selector = StepwiseSelector(direction="backward", criterion="aic", exclude=["x4"])
        selector.fit(X, y)

        assert selector.is_fitted_
        # x4 should be kept even if not significant
        assert "x4" in selector.selected_features_


class TestStepwiseSelectorTransform:
    """Test StepwiseSelector transform methods."""

    def test_transform(self, stepwise_data):
        """Test transform returns correct columns."""
        X, y = stepwise_data
        selector = StepwiseSelector(direction="both", criterion="aic")
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert list(X_transformed.columns) == selector.selected_features_
        assert len(X_transformed) == len(X)

    def test_fit_transform(self, stepwise_data):
        """Test fit_transform."""
        X, y = stepwise_data
        selector = StepwiseSelector(direction="both", criterion="aic")

        X_transformed = selector.fit_transform(X, y)

        assert selector.is_fitted_
        assert isinstance(X_transformed, pd.DataFrame)

    def test_transform_before_fit_raises(self, stepwise_data):
        """Test transform before fit raises error."""
        X, _ = stepwise_data
        selector = StepwiseSelector()

        with pytest.raises(ValueError):
            selector.transform(X)


class TestStepwiseSelectorReport:
    """Test StepwiseSelector reporting methods."""

    def test_report(self, stepwise_data):
        """Test report generation."""
        X, y = stepwise_data
        selector = StepwiseSelector(direction="forward", criterion="aic")
        selector.fit(X, y)

        report = selector.report()

        assert isinstance(report, pd.DataFrame)
        assert "iteration" in report.columns
        assert "action" in report.columns
        assert "feature" in report.columns

    def test_summary(self, stepwise_data):
        """Test summary generation."""
        X, y = stepwise_data
        selector = StepwiseSelector(direction="both", criterion="aic")
        selector.fit(X, y)

        summary = selector.summary()

        assert isinstance(summary, str)
        assert "Stepwise Selection Summary" in summary
        assert "Selected" in summary

    def test_report_before_fit_raises(self):
        """Test report before fit raises error."""
        selector = StepwiseSelector()

        with pytest.raises(ValueError):
            selector.report()

    def test_summary_before_fit_raises(self):
        """Test summary before fit raises error."""
        selector = StepwiseSelector()

        with pytest.raises(ValueError):
            selector.summary()
