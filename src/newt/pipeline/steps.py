"""Pipeline step objects used by the scorecard pipeline coordinator."""

from typing import List, Optional

import pandas as pd

from newt.config import BINNING, FILTERING, MODELING, SCORECARD
from newt.pipeline.state import PipelineState


class PrefilterStep:
    """Pipeline step for exploratory data analysis and basic feature filtering.

    This step calculates metrics like Information Value (IV), missing rates, and
    feature correlations to perform an initial reduction of the feature space.

    Args:
        iv_threshold: Minimum IV for a feature to be retained.
        missing_threshold: Maximum missing rate for a feature to be retained.
        corr_threshold: Maximum absolute correlation between any two features.
        iv_bins: Number of bins used for initial IV calculation.
        **kwargs: Additional arguments passed to the FeatureSelector.
    """

    def __init__(
        self,
        iv_threshold: float = FILTERING.DEFAULT_IV_THRESHOLD,
        missing_threshold: float = FILTERING.DEFAULT_MISSING_THRESHOLD,
        corr_threshold: float = FILTERING.DEFAULT_CORR_THRESHOLD,
        iv_bins: int = BINNING.DEFAULT_BUCKETS,
        **kwargs,
    ):
        """Initialize PrefilterStep."""
        self.iv_threshold = iv_threshold
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.iv_bins = iv_bins
        self.kwargs = kwargs

    def run(self, state: PipelineState) -> PipelineState:
        """Run pre-filtering and update the pipeline state."""
        from newt.features.selection.selector import FeatureSelector

        selector = FeatureSelector(
            iv_bins=self.iv_bins,
            metrics=["iv", "missing_rate", "correlation"],
            **self.kwargs,
        )
        selector.fit(state.X_current, state.y_train)
        selector.select(
            iv_threshold=self.iv_threshold,
            missing_threshold=self.missing_threshold,
            corr_threshold=self.corr_threshold,
        )

        state.prefilter = selector
        state.X_current = selector.transform(state.X_current)
        if state.X_test_current is not None:
            state.X_test_current = selector.transform(state.X_test_current)

        state.steps.append("prefilter")
        return state


class BinningStep:
    """Pipeline step for fitting discretization rules to numeric features.

    This step uses one of several algorithms (ChiMerge, DT, etc.) to discover
    optimal bin boundaries and applies these boundaries to the dataset.

    Args:
        method: Binning algorithm name ('chi', 'dt', 'opt', 'kmean', etc.).
        n_bins: Target number of bins.
        cols: Specific columns to bin. If None, all numeric columns are used.
        **kwargs: Additional arguments passed to the Binner.
    """

    def __init__(
        self,
        method: str = "chi",
        n_bins: int = BINNING.DEFAULT_N_BINS,
        cols: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize BinningStep."""
        self.method = method
        self.n_bins = n_bins
        self.cols = cols
        self.kwargs = kwargs

    def run(self, state: PipelineState) -> PipelineState:
        """Run binning and update the pipeline state."""
        from newt.features.binning import Binner

        binner = Binner()
        binner.fit(
            state.X_current,
            state.y_train,
            method=self.method,
            n_bins=self.n_bins,
            cols=self.cols,
            **self.kwargs,
        )

        state.binner = binner
        state.X_binned = binner.transform(state.X_current, labels=False)
        if state.X_test_current is not None:
            state.X_test_binned = binner.transform(state.X_test_current, labels=False)

        state.steps.append("bin")
        return state


class WoeTransformStep:
    """Pipeline step for Weight of Evidence (WOE) encoding.

    Converts discrete bin indices into numerical WOE values based on the
    log-odds of the event rate in each bin.

    Args:
        epsilon: Small constant to avoid log(0) and division by zero.
        **kwargs: Additional arguments passed to the WOEEncoder.
    """

    def __init__(self, epsilon: float = BINNING.DEFAULT_EPSILON, **kwargs):
        """Initialize WoeTransformStep."""
        self.epsilon = epsilon
        self.kwargs = kwargs

    def run(self, state: PipelineState) -> PipelineState:
        """Run WOE transformation and update the pipeline state."""
        from newt.features.analysis.woe_calculator import WOEEncoder

        if state.X_binned is None:
            raise ValueError("Must call bin() before woe_transform().")

        state.X_woe = state.X_binned.copy()
        state.woe_encoders = {}

        for column in state.X_binned.columns:
            encoder = WOEEncoder(epsilon=self.epsilon)
            train_values = state.X_binned[column].astype(str)
            encoder.fit(train_values, state.y_train)

            state.woe_encoders[column] = encoder
            state.X_woe[column] = encoder.transform(train_values)

        if state.X_test_binned is not None:
            state.X_test_woe = state.X_test_binned.copy()
            for column in state.X_test_binned.columns:
                if column in state.woe_encoders:
                    test_values = state.X_test_binned[column].astype(str)
                    state.X_test_woe[column] = state.woe_encoders[column].transform(
                        test_values
                    )

        state.X_current = state.X_woe
        if state.X_test_woe is not None:
            state.X_test_current = state.X_test_woe

        state.steps.append("woe_transform")
        return state


class PostfilterStep:
    """Pipeline step for stability (PSI) and multicollinearity (VIF) filtering.

    Ensures that selected features are stable across time/segments and do not
    suffer from high redundancy.

    Args:
        psi_threshold: Maximum PSI value for a feature to be retained.
        vif_threshold: Maximum VIF value for a feature to be retained.
        X_test: Optional test dataset for PSI calculation.
        **kwargs: Additional arguments passed to the PostFilter.
    """

    def __init__(
        self,
        psi_threshold: float = FILTERING.DEFAULT_PSI_THRESHOLD,
        vif_threshold: float = FILTERING.DEFAULT_VIF_THRESHOLD,
        X_test: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        """Initialize PostfilterStep."""
        self.psi_threshold = psi_threshold
        self.vif_threshold = vif_threshold
        self.X_test = X_test
        self.kwargs = kwargs

    def run(self, state: PipelineState) -> PipelineState:
        """Run post-filtering and update the pipeline state."""
        from newt.features.selection.postfilter import PostFilter

        test_data = self.X_test if self.X_test is not None else state.X_test_current
        postfilter = PostFilter(
            psi_threshold=self.psi_threshold,
            vif_threshold=self.vif_threshold,
            **self.kwargs,
        )

        state.postfilter = postfilter
        state.X_current = postfilter.fit_transform(state.X_current, test_data)
        if state.X_test_current is not None:
            state.X_test_current = postfilter.transform(state.X_test_current)

        state.steps.append("postfilter")
        return state


class StepwiseStep:
    """Pipeline step for stepwise feature selection based on statistical criteria.

    Iteratively adds or removes features based on p-values or information
    criteria (AIC/BIC) to find a high-performing subset.

    Args:
        direction: Selection strategy: 'forward', 'backward', or 'both'.
        criterion: The statistical criterion for selection: 'pvalue', 'aic', 'bic'.
        p_enter: p-value threshold for adding a feature.
        p_remove: p-value threshold for removing a feature.
        exclude: List of columns to always keep.
        **kwargs: Additional arguments passed to the StepwiseSelector.
    """

    def __init__(
        self,
        direction: str = "both",
        criterion: str = "aic",
        p_enter: float = MODELING.DEFAULT_P_ENTER,
        p_remove: float = MODELING.DEFAULT_P_REMOVE,
        exclude: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize StepwiseStep."""
        self.direction = direction
        self.criterion = criterion
        self.p_enter = p_enter
        self.p_remove = p_remove
        self.exclude = exclude
        self.kwargs = kwargs

    def run(self, state: PipelineState) -> PipelineState:
        """Run stepwise selection and update the pipeline state."""
        from newt.features.selection.stepwise import StepwiseSelector

        selector = StepwiseSelector(
            direction=self.direction,
            criterion=self.criterion,
            p_enter=self.p_enter,
            p_remove=self.p_remove,
            exclude=self.exclude,
            **self.kwargs,
        )

        state.stepwise = selector
        state.X_current = selector.fit_transform(state.X_current, state.y_train)
        if state.X_test_current is not None:
            state.X_test_current = selector.transform(state.X_test_current)

        state.steps.append("stepwise")
        return state


class ModelingStep:
    """Pipeline step for fitting the final Logistic Regression model.

    Standardizes the model interface and ensures all training artifacts are
    captured for scorecard conversion.

    Args:
        fit_intercept: Whether to calculate the intercept for this model.
        **kwargs: Arguments passed to the LogisticModel.
    """

    def __init__(self, fit_intercept: bool = True, **kwargs):
        """Initialize ModelingStep."""
        self.fit_intercept = fit_intercept
        self.kwargs = kwargs

    def run(self, state: PipelineState) -> PipelineState:
        """Run model training and update the pipeline state."""
        from newt.modeling.logistic import LogisticModel

        model = LogisticModel(fit_intercept=self.fit_intercept, **self.kwargs)
        model.fit(state.X_current, state.y_train)

        state.model = model
        state.steps.append("model")
        return state


class ScorecardStep:
    """Pipeline step for converting a fitted model into an additive scorecard.

    Scales model coefficients based on business requirements like Base Score
    and Points to Double the Odds (PDO).

    Args:
        base_score: Target score at the given base_odds.
        pdo: Points to Double the Odds.
        base_odds: Target odds at the given base_score.
        **kwargs: Optional keyword arguments for Scorecard initialization.
    """

    def __init__(
        self,
        base_score: int = SCORECARD.DEFAULT_BASE_SCORE,
        pdo: int = SCORECARD.DEFAULT_PDO,
        base_odds: float = SCORECARD.DEFAULT_BASE_ODDS,
        **kwargs,
    ):
        """Initialize ScorecardStep."""
        self.base_score = base_score
        self.pdo = pdo
        self.base_odds = base_odds
        self.kwargs = kwargs

    def run(self, state: PipelineState) -> PipelineState:
        """Run scorecard generation and update the pipeline state."""
        from newt.modeling.scorecard import Scorecard

        scorecard = Scorecard(
            base_score=self.base_score, pdo=self.pdo, base_odds=self.base_odds
        )
        scorecard.from_model(state.model, state.binner, state.woe_encoders)

        state.scorecard = scorecard
        state.steps.append("scorecard")
        return state
