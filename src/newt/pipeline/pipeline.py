"""Scorecard pipeline implemented as a thin coordinator around step objects."""

from typing import Any, Dict, List, Optional

import pandas as pd

from newt.config import BINNING, FILTERING, MODELING, SCORECARD
from newt.pipeline.state import PipelineState
from newt.pipeline.steps import (
    BinningStep,
    ModelingStep,
    PostfilterStep,
    PrefilterStep,
    ScorecardStep,
    StepwiseStep,
    WoeTransformStep,
)


class ScorecardPipeline:
    """Chainable pipeline for end-to-end credit scorecard development.

    The ScorecardPipeline provides a fluent, high-level API to orchestrate the entire
    modeling workflow—from initial feature filtering to final scorecard generation.
    It manages internal state transitions and provides access to intermediate artifacts
    (e.g., binning results, WOE encoders) at each step.

    Examples:
        >>> from newt.pipeline import ScorecardPipeline
        >>> pipeline = (
        ...     ScorecardPipeline(X_train, y_train, X_test, y_test)
        ...     .prefilter(iv_threshold=0.02)
        ...     .bin(method='chi', n_bins=5)
        ...     .woe_transform()
        ...     .postfilter(psi_threshold=0.1)
        ...     .build_model()
        ...     .generate_scorecard(base_score=600, pdo=20)
        ... )
        >>> scores = pipeline.score(X_val)
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ):
        """Initialize the pipeline with training and optional testing data.

        Args:
            X: Training feature DataFrame.
            y: Training target Series (binary 0/1).
            X_test: Optional testing feature DataFrame for validation and
                PSI calculation.
            y_test: Optional testing target Series.
        """
        self._state = PipelineState(X, y, X_test, y_test)

    def prefilter(
        self,
        iv_threshold: float = FILTERING.DEFAULT_IV_THRESHOLD,
        missing_threshold: float = FILTERING.DEFAULT_MISSING_THRESHOLD,
        corr_threshold: float = FILTERING.DEFAULT_CORR_THRESHOLD,
        iv_bins: int = BINNING.DEFAULT_BUCKETS,
        **kwargs,
    ) -> "ScorecardPipeline":
        """Apply pre-modeling filters based on EDA metrics.

        Filters features using Information Value (IV), missing rate, and
        feature-to-feature correlation. This step is typically the first in
        the pipeline to reduce dimensionality before expensive operations
        like binning.

        Args:
            iv_threshold: Minimum IV required to keep a feature.
            missing_threshold: Maximum allowed missing rate (0.0 to 1.0).
            corr_threshold: Maximum allowed correlation between feature pairs.
            iv_bins: Number of buckets used for temporary auto-binning
                during IV compute.
            **kwargs: Additional arguments passed to FeatureSelector.

        Returns:
            ScorecardPipeline: The pipeline instance (self) for chaining.

        Examples:
            >>> pipeline.prefilter(iv_threshold=0.05, corr_threshold=0.7)
        """
        step = PrefilterStep(
            iv_threshold=iv_threshold,
            missing_threshold=missing_threshold,
            corr_threshold=corr_threshold,
            iv_bins=iv_bins,
            **kwargs,
        )
        self._state = step.run(self._state)
        return self

    def bin(
        self,
        method: str = "chi",
        n_bins: int = BINNING.DEFAULT_N_BINS,
        cols: Optional[List[str]] = None,
        **kwargs,
    ) -> "ScorecardPipeline":
        """Discretize continuous variables into discrete bins.

        Supported methods include 'chi' (ChiMerge), 'dt' (Decision Tree),
        'opt' (Optimal), 'quantile' (Equal Frequency), 'step' (Equal Width),
        and 'kmean'.

        Args:
            method: Binning algorithm name. Defaults to 'chi'.
            n_bins: Target number of bins for each feature.
            cols: Optional list of features to bin. If None, all numeric
                features are used.
            **kwargs: Additional parameters for the chosen binner (e.g.,
                monotonic=True).

        Returns:
            ScorecardPipeline: The pipeline instance (self) for chaining.

        Examples:
            >>> pipeline.bin(method='opt', n_bins=5, monotonic='auto')
        """
        step = BinningStep(method=method, n_bins=n_bins, cols=cols, **kwargs)
        self._state = step.run(self._state)
        return self

    def woe_transform(
        self,
        epsilon: float = BINNING.DEFAULT_EPSILON,
        **kwargs,
    ) -> "ScorecardPipeline":
        """Apply Weight of Evidence (WOE) encoding to binned features.

        Converts binned categorical/ordinal values into numeric WOE values based on the
        distribution of good and bad labels in each bin.

        Args:
            epsilon: Small constant to prevent log(0) or division by zero.
            **kwargs: Additional arguments passed to WOEEncoder.

        Returns:
            ScorecardPipeline: The pipeline instance (self) for chaining.

        Examples:
            >>> pipeline.woe_transform(epsilon=1e-10)
        """
        step = WoeTransformStep(epsilon=epsilon, **kwargs)
        self._state = step.run(self._state)
        return self

    def postfilter(
        self,
        psi_threshold: float = FILTERING.DEFAULT_PSI_THRESHOLD,
        vif_threshold: float = FILTERING.DEFAULT_VIF_THRESHOLD,
        X_test: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> "ScorecardPipeline":
        """Apply post-transformation filters like PSI stability and VIF
        multicollinearity.

        Typically run after WOE transformation to ensure the selected features are
        stable over time (PSI) and not redundant (VIF).

        Args:
            psi_threshold: Maximum allowed Population Stability Index
                between train/test.
            vif_threshold: Maximum allowed Variance Inflation Factor.
            X_test: Optional override for the test set used for PSI compute.
            **kwargs: Additional parameters passed to PostFilter.

        Returns:
            ScorecardPipeline: The pipeline instance (self) for chaining.

        Examples:
            >>> pipeline.postfilter(psi_threshold=0.1, vif_threshold=5.0)
        """
        step = PostfilterStep(
            psi_threshold=psi_threshold,
            vif_threshold=vif_threshold,
            X_test=X_test,
            **kwargs,
        )
        self._state = step.run(self._state)
        return self

    def stepwise(
        self,
        direction: str = "both",
        criterion: str = "aic",
        p_enter: float = MODELING.DEFAULT_P_ENTER,
        p_remove: float = MODELING.DEFAULT_P_REMOVE,
        exclude: Optional[List[str]] = None,
        **kwargs,
    ) -> "ScorecardPipeline":
        """Perform automated feature selection via stepwise regression.

        Successively adds or removes features based on statistical significance or
        information criteria (AIC/BIC).

        Args:
            direction: Search direction: 'forward', 'backward', or 'both'.
            criterion: Selection criterion: 'p-value', 'aic', or 'bic'.
            p_enter: P-value threshold to enter the model (if using 'p-value').
            p_remove: P-value threshold to be removed from the model.
            exclude: Optional list of features to always keep in the model.
            **kwargs: Additional parameters passed to StepwiseSelector.

        Returns:
            ScorecardPipeline: The pipeline instance (self) for chaining.

        Examples:
            >>> pipeline.stepwise(direction='both', criterion='aic')
        """
        step = StepwiseStep(
            direction=direction,
            criterion=criterion,
            p_enter=p_enter,
            p_remove=p_remove,
            exclude=exclude,
            **kwargs,
        )
        self._state = step.run(self._state)
        return self

    def build_model(
        self,
        fit_intercept: bool = True,
        **kwargs,
    ) -> "ScorecardPipeline":
        """Train the final logistic regression model on selected WOE features.

        Args:
            fit_intercept: Whether to calculate the intercept for this model.
            **kwargs: Additional parameters passed to LogisticModel.

        Returns:
            ScorecardPipeline: The pipeline instance (self) for chaining.

        Examples:
            >>> pipeline.build_model(method='bfgs')
        """
        step = ModelingStep(fit_intercept=fit_intercept, **kwargs)
        self._state = step.run(self._state)
        return self

    def generate_scorecard(
        self,
        base_score: int = SCORECARD.DEFAULT_BASE_SCORE,
        pdo: int = SCORECARD.DEFAULT_PDO,
        base_odds: float = SCORECARD.DEFAULT_BASE_ODDS,
        **kwargs,
    ) -> "ScorecardPipeline":
        """Convert the fitted logistic model into a point-based scorecard.

        Args:
            base_score: The target score at 'base_odds'.
            pdo: Points to Double the Odds.
            base_odds: The odds (Good:Bad) at 'base_score'.
            **kwargs: Additional parameters passed to Scorecard.

        Returns:
            ScorecardPipeline: The pipeline instance (self) for chaining.

        Examples:
            >>> pipeline.generate_scorecard(base_score=600, pdo=20)
        """
        step = ScorecardStep(
            base_score=base_score,
            pdo=pdo,
            base_odds=base_odds,
            **kwargs,
        )
        self._state = step.run(self._state)
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Apply the finished scorecard to new raw data to produce scores.

        Args:
            X: Raw feature DataFrame (un-binned, un-encoded).

        Returns:
            pd.Series: Calculated scores for each row.

        Raises:
            ValueError: If the scorecard has not been generated yet.
        """
        if self.scorecard_ is None:
            raise ValueError("Scorecard not built. Call generate_scorecard() first.")
        return self.scorecard_.score(X)

    @property
    def X_train(self) -> pd.DataFrame:
        return self._state.X_train

    @property
    def y_train(self) -> pd.Series:
        return self._state.y_train

    @property
    def X_test(self) -> Optional[pd.DataFrame]:
        return self._state.X_test

    @property
    def y_test(self) -> Optional[pd.Series]:
        """Get the test target series."""
        return self._state.y_test

    @property
    def X_current(self) -> pd.DataFrame:
        """Get the current training feature set (after transformations)."""
        return self._state.X_current

    @property
    def X_test_current(self) -> Optional[pd.DataFrame]:
        """Get the current test feature set (after transformations)."""
        return self._state.X_test_current

    @property
    def steps_(self) -> List[str]:
        """List of step names that have been executed."""
        return self._state.steps

    @property
    def prefilter_(self) -> Any:
        """The FeatureSelector instance from the prefilter step."""
        return self._state.prefilter

    @prefilter_.setter
    def prefilter_(self, value: Any) -> None:
        self._state.prefilter = value

    @property
    def binner_(self) -> Any:
        """The Binner instance from the bin step."""
        return self._state.binner

    @binner_.setter
    def binner_(self, value: Any) -> None:
        self._state.binner = value

    @property
    def woe_encoders_(self) -> Dict[str, Any]:
        """Dictionary mapping feature names to WOEEncoder instances."""
        return self._state.woe_encoders

    @woe_encoders_.setter
    def woe_encoders_(self, value: Dict[str, Any]) -> None:
        self._state.woe_encoders = value

    @property
    def postfilter_(self) -> Any:
        """The PostFilter instance from the postfilter step."""
        return self._state.postfilter

    @postfilter_.setter
    def postfilter_(self, value: Any) -> None:
        self._state.postfilter = value

    @property
    def stepwise_(self) -> Any:
        """The StepwiseSelector instance from the stepwise step."""
        return self._state.stepwise

    @stepwise_.setter
    def stepwise_(self, value: Any) -> None:
        self._state.stepwise = value

    @property
    def model_(self) -> Any:
        """The fitted LogisticModel instance."""
        return self._state.model

    @model_.setter
    def model_(self, value: Any) -> None:
        self._state.model = value

    @property
    def scorecard_(self) -> Any:
        """The generated Scorecard instance."""
        return self._state.scorecard

    @scorecard_.setter
    def scorecard_(self, value: Any) -> None:
        self._state.scorecard = value

    @property
    def X_binned_(self) -> Optional[pd.DataFrame]:
        """Training data after binning transformation."""
        return self._state.X_binned

    @X_binned_.setter
    def X_binned_(self, value: Optional[pd.DataFrame]) -> None:
        self._state.X_binned = value

    @property
    def X_woe_(self) -> Optional[pd.DataFrame]:
        """Training data after WOE transformation."""
        return self._state.X_woe

    @X_woe_.setter
    def X_woe_(self, value: Optional[pd.DataFrame]) -> None:
        self._state.X_woe = value

    @property
    def X_test_binned_(self) -> Optional[pd.DataFrame]:
        return self._state.X_test_binned

    @X_test_binned_.setter
    def X_test_binned_(self, value: Optional[pd.DataFrame]) -> None:
        self._state.X_test_binned = value

    @property
    def X_test_woe_(self) -> Optional[pd.DataFrame]:
        return self._state.X_test_woe

    @X_test_woe_.setter
    def X_test_woe_(self, value: Optional[pd.DataFrame]) -> None:
        self._state.X_test_woe = value

    @property
    def prefilter_result(self) -> Any:
        """Alias for prefilter_."""
        return self.prefilter_

    @property
    def binner(self) -> Any:
        """Alias for binner_."""
        return self.binner_

    @property
    def woe_encoders(self) -> Dict[str, Any]:
        """Alias for woe_encoders_."""
        return self.woe_encoders_

    @property
    def postfilter_result(self) -> Any:
        """Alias for postfilter_."""
        return self.postfilter_

    @property
    def model(self) -> Any:
        """Alias for model_."""
        return self.model_

    @property
    def scorecard(self) -> Any:
        """Alias for scorecard_."""
        return self.scorecard_

    @property
    def selected_features(self) -> List[str]:
        """Get the list of features currently selected in the pipeline."""
        return self._state.selected_features

    def summary(self) -> Dict[str, Any]:
        """Get pipeline summary."""
        summary = {
            "steps": self.steps_,
            "n_features_initial": len(self.X_train.columns),
            "n_features_final": len(self.X_current.columns),
            "selected_features": self.selected_features,
        }

        if self.prefilter_ is not None:
            summary["prefilter_selected"] = len(self.prefilter_.selected_features_)
            summary["prefilter_removed"] = len(self.prefilter_.removed_features_)

        if self.postfilter_ is not None:
            summary["postfilter_selected"] = len(self.postfilter_.selected_features_)
            summary["postfilter_removed"] = len(self.postfilter_.removed_features_)

        if self.model_ is not None:
            summary["model_fitted"] = True

        if self.scorecard_ is not None:
            summary["scorecard_built"] = True

        return summary
