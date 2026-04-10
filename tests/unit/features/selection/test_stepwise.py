"""Unit tests for StepwiseSelector."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from newt.features.selection.stepwise import StepwiseSelector

try:
    from newt._newt_native import (
        batch_fit_logistic_regression_numpy,
        fit_logistic_regression_numpy,
    )
except ImportError:
    batch_fit_logistic_regression_numpy = None
    fit_logistic_regression_numpy = None


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


def _require_stepwise_rust():
    if (
        fit_logistic_regression_numpy is None
        or batch_fit_logistic_regression_numpy is None
    ):
        pytest.skip("Rust stepwise engine is not available in this environment")


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

    def test_invalid_engine(self):
        """Test invalid engine raises error."""
        with pytest.raises(ValueError):
            StepwiseSelector(engine="invalid")


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
        selector = StepwiseSelector(
            direction="backward", criterion="aic", exclude=["x4"]
        )
        selector.fit(X, y)

        assert selector.is_fitted_
        # x4 should be kept even if not significant
        assert "x4" in selector.selected_features_

    @pytest.mark.parametrize("direction", ["forward", "backward", "both"])
    @pytest.mark.parametrize("criterion", ["aic", "bic", "pvalue"])
    def test_rust_matches_python_feature_selection(
        self, stepwise_data, direction, criterion
    ):
        """Rust and Python engines should select the same features."""
        _require_stepwise_rust()
        X, y = stepwise_data

        python_selector = StepwiseSelector(
            direction=direction,
            criterion=criterion,
            engine="python",
            verbose=False,
        )
        rust_selector = StepwiseSelector(
            direction=direction,
            criterion=criterion,
            engine="rust",
            verbose=False,
        )

        python_selector.fit(X, y)
        rust_selector.fit(X, y)

        assert rust_selector.selected_features_ == python_selector.selected_features_

    def test_rust_matches_python_with_excluded_features(self, stepwise_data):
        """Force-included features should match across engines."""
        _require_stepwise_rust()
        X, y = stepwise_data

        python_selector = StepwiseSelector(
            direction="backward",
            criterion="aic",
            exclude=["x4"],
            engine="python",
            verbose=False,
        )
        rust_selector = StepwiseSelector(
            direction="backward",
            criterion="aic",
            exclude=["x4"],
            engine="rust",
            verbose=False,
        )

        python_selector.fit(X, y)
        rust_selector.fit(X, y)

        assert rust_selector.selected_features_ == python_selector.selected_features_

    def test_rust_matches_python_without_intercept(self, stepwise_data):
        """Selection should remain aligned without an intercept term."""
        _require_stepwise_rust()
        X, y = stepwise_data

        python_selector = StepwiseSelector(
            direction="both",
            criterion="aic",
            fit_intercept=False,
            engine="python",
            verbose=False,
        )
        rust_selector = StepwiseSelector(
            direction="both",
            criterion="aic",
            fit_intercept=False,
            engine="rust",
            verbose=False,
        )

        python_selector.fit(X, y)
        rust_selector.fit(X, y)

        assert rust_selector.selected_features_ == python_selector.selected_features_

    def test_rust_logit_matches_statsmodels(self):
        """Rust logistic regression diagnostics should match statsmodels."""
        _require_stepwise_rust()
        rng = np.random.default_rng(123)
        n_samples = 600
        X = rng.normal(size=(n_samples, 4))
        beta = np.array([0.2, 1.0, -0.8, 0.5])
        logits = X @ beta
        probabilities = 1 / (1 + np.exp(-logits))
        y = (rng.random(n_samples) < probabilities).astype(float)
        X_with_intercept = np.column_stack([np.ones(n_samples), X])

        statsmodels_result = sm.Logit(y, X_with_intercept).fit(disp=False)
        rust_result = fit_logistic_regression_numpy(X_with_intercept, y)

        np.testing.assert_allclose(
            np.array(rust_result["coefficients"]),
            statsmodels_result.params,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.array(rust_result["p_values"]),
            statsmodels_result.pvalues,
            atol=1e-8,
        )
        assert rust_result["aic"] == pytest.approx(statsmodels_result.aic, abs=1e-8)
        assert rust_result["bic"] == pytest.approx(statsmodels_result.bic, abs=1e-8)
        assert rust_result["converged"] is True

    @pytest.mark.parametrize("direction", ["forward", "both"])
    def test_rust_skips_invalid_candidates_during_selection(
        self, stepwise_data, direction
    ):
        """Rust stepwise should skip invalid candidates instead of crashing."""
        _require_stepwise_rust()
        X, y = stepwise_data
        X = X.copy()
        X["const_bad"] = 1.0
        X["dup_x1"] = X["x1"]
        X["nan_bad"] = np.nan
        X["inf_bad"] = np.inf

        python_selector = StepwiseSelector(
            direction=direction,
            criterion="aic",
            engine="python",
            verbose=False,
        )
        rust_selector = StepwiseSelector(
            direction=direction,
            criterion="aic",
            engine="rust",
            verbose=False,
        )

        python_selector.fit(X, y)
        rust_selector.fit(X, y)

        assert rust_selector.selected_features_ == python_selector.selected_features_
        assert "const_bad" not in rust_selector.selected_features_
        assert "dup_x1" not in rust_selector.selected_features_
        assert "nan_bad" not in rust_selector.selected_features_
        assert "inf_bad" not in rust_selector.selected_features_

    def test_rust_batch_logit_returns_failed_records_for_bad_candidates(self):
        """Bad batch candidates should return sentinel failures, not panic."""
        _require_stepwise_rust()
        n_samples = 24
        y = np.array([0, 1] * (n_samples // 2), dtype=float)
        fixed_x = np.ones((n_samples, 1), dtype=float)
        normal_candidate = np.linspace(-1.0, 1.0, n_samples, dtype=float)
        constant_candidate = np.ones(n_samples, dtype=float)
        nan_candidate = np.full(n_samples, np.nan, dtype=float)
        inf_candidate = np.full(n_samples, np.inf, dtype=float)

        results = batch_fit_logistic_regression_numpy(
            fixed_x,
            [
                normal_candidate,
                constant_candidate,
                nan_candidate,
                inf_candidate,
            ],
            y,
            max_iter=25,
        )

        assert len(results) == 4
        assert results[0]["converged"] is True

        for failed_result in results[1:]:
            assert failed_result["converged"] is False
            assert failed_result["p_value"] == pytest.approx(1.0)
            assert failed_result["aic"] == np.inf
            assert failed_result["bic"] == np.inf

    def test_rust_single_logit_singular_matrix_raises_runtime_error(self):
        """Singular single-model fits should raise a normal Python error."""
        _require_stepwise_rust()
        n_samples = 24
        y = np.array([0, 1] * (n_samples // 2), dtype=float)
        singular_x = np.column_stack([np.ones(n_samples), np.ones(n_samples)])

        with pytest.raises(RuntimeError):
            fit_logistic_regression_numpy(singular_x, y, max_iter=25)


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
