"""Scorecard pipeline implemented as a thin coordinator around step objects."""

from typing import Any, Dict, List, Optional

import pandas as pd

from newt.config import BINNING, FILTERING, MODELING, SCORECARD
from newt.pipeline.state import PipelineState
from newt.pipeline.steps import (
    BinningStep,
    ModelBuildStep,
    PostfilterStep,
    PrefilterStep,
    ScorecardBuildStep,
    StepwiseSelectionStep,
    WoeTransformStep,
)


class ScorecardPipeline:
    """Chainable pipeline for credit scorecard development."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ):
        self._state = PipelineState(X, y, X_test, y_test)

    def prefilter(
        self,
        iv_threshold: float = FILTERING.DEFAULT_IV_THRESHOLD,
        missing_threshold: float = FILTERING.DEFAULT_MISSING_THRESHOLD,
        corr_threshold: float = FILTERING.DEFAULT_CORR_THRESHOLD,
        iv_bins: int = BINNING.DEFAULT_BUCKETS,
        **kwargs,
    ) -> "ScorecardPipeline":
        """Apply pre-filtering based on IV, missing rate, and correlation."""
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
        """Apply binning to features."""
        step = BinningStep(method=method, n_bins=n_bins, cols=cols, **kwargs)
        self._state = step.run(self._state)
        return self

    def woe_transform(
        self,
        epsilon: float = BINNING.DEFAULT_EPSILON,
        **kwargs,
    ) -> "ScorecardPipeline":
        """Apply WOE transformation to binned features."""
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
        """Apply post-filtering based on PSI and VIF."""
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
        """Apply stepwise regression feature selection."""
        step = StepwiseSelectionStep(
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
        """Build logistic regression model."""
        step = ModelBuildStep(fit_intercept=fit_intercept, **kwargs)
        self._state = step.run(self._state)
        return self

    def generate_scorecard(
        self,
        base_score: int = SCORECARD.DEFAULT_BASE_SCORE,
        pdo: int = SCORECARD.DEFAULT_PDO,
        base_odds: float = SCORECARD.DEFAULT_BASE_ODDS,
        **kwargs,
    ) -> "ScorecardPipeline":
        """Generate scorecard from model."""
        step = ScorecardBuildStep(
            base_score=base_score,
            pdo=pdo,
            base_odds=base_odds,
            **kwargs,
        )
        self._state = step.run(self._state)
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Calculate scores for new data."""
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
        return self._state.y_test

    @property
    def X_current(self) -> pd.DataFrame:
        return self._state.X_current

    @property
    def X_test_current(self) -> Optional[pd.DataFrame]:
        return self._state.X_test_current

    @property
    def steps_(self) -> List[str]:
        return self._state.steps

    @property
    def prefilter_(self) -> Any:
        return self._state.prefilter

    @prefilter_.setter
    def prefilter_(self, value: Any) -> None:
        self._state.prefilter = value

    @property
    def binner_(self) -> Any:
        return self._state.binner

    @binner_.setter
    def binner_(self, value: Any) -> None:
        self._state.binner = value

    @property
    def woe_encoders_(self) -> Dict[str, Any]:
        return self._state.woe_encoders

    @woe_encoders_.setter
    def woe_encoders_(self, value: Dict[str, Any]) -> None:
        self._state.woe_encoders = value

    @property
    def postfilter_(self) -> Any:
        return self._state.postfilter

    @postfilter_.setter
    def postfilter_(self, value: Any) -> None:
        self._state.postfilter = value

    @property
    def stepwise_(self) -> Any:
        return self._state.stepwise

    @stepwise_.setter
    def stepwise_(self, value: Any) -> None:
        self._state.stepwise = value

    @property
    def model_(self) -> Any:
        return self._state.model

    @model_.setter
    def model_(self, value: Any) -> None:
        self._state.model = value

    @property
    def scorecard_(self) -> Any:
        return self._state.scorecard

    @scorecard_.setter
    def scorecard_(self, value: Any) -> None:
        self._state.scorecard = value

    @property
    def X_binned_(self) -> Optional[pd.DataFrame]:
        return self._state.X_binned

    @X_binned_.setter
    def X_binned_(self, value: Optional[pd.DataFrame]) -> None:
        self._state.X_binned = value

    @property
    def X_woe_(self) -> Optional[pd.DataFrame]:
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
        return self.prefilter_

    @property
    def binner(self) -> Any:
        return self.binner_

    @property
    def woe_encoders(self) -> Dict[str, Any]:
        return self.woe_encoders_

    @property
    def postfilter_result(self) -> Any:
        return self.postfilter_

    @property
    def model(self) -> Any:
        return self.model_

    @property
    def scorecard(self) -> Any:
        return self.scorecard_

    @property
    def selected_features(self) -> List[str]:
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
