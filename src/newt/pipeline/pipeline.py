"""
ScorecardPipeline - Chainable pipeline for credit scorecard development.

Provides a fluent API for the complete scorecard modeling workflow.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from newt.config import BINNING, FILTERING, MODELING, SCORECARD


class ScorecardPipeline:
    """
    Chainable pipeline for credit scorecard development.

    Provides a fluent API to chain together the complete workflow:
    1. Pre-filtering (IV, missing rate, correlation)
    2. Binning (various algorithms)
    3. WOE transformation
    4. Post-filtering (PSI, VIF)
    5. Logistic regression modeling
    6. Scorecard generation

    Each step can also be used independently.

    Examples
    --------
    >>> # Full pipeline
    >>> pipeline = (
    ...     ScorecardPipeline(X_train, y_train)
    ...     .prefilter(iv_threshold=0.02, missing_threshold=0.9)
    ...     .bin(method='opt', n_bins=5)
    ...     .woe_transform()
    ...     .postfilter(psi_threshold=0.25, X_test=X_test_woe)
    ...     .build_model()
    ...     .generate_scorecard(base_score=600, pdo=50)
    ... )
    >>> scores = pipeline.score(X_new)

    >>> # Access intermediate results
    >>> pipeline.prefilter_result.report()
    >>> pipeline.binner.export()
    >>> pipeline.model.summary()
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ):
        """
        Initialize ScorecardPipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature data.
        y : pd.Series
            Training target variable (binary 0/1).
        X_test : pd.DataFrame, optional
            Test feature data for PSI calculation.
        y_test : pd.Series, optional
            Test target for model evaluation.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.X_test = X_test.copy() if X_test is not None else None
        self.y_test = y_test.copy() if y_test is not None else None

        # Current state
        self.X_current = self.X_train.copy()
        self.X_test_current = self.X_test.copy() if self.X_test is not None else None

        # Step results
        self.steps_: List[str] = []
        self.prefilter_: Any = None
        self.binner_: Any = None
        self.woe_encoders_: Dict[str, Any] = {}
        self.postfilter_: Any = None
        self.stepwise_: Any = None
        self.model_: Any = None
        self.scorecard_: Any = None

        # Intermediate data
        self.X_binned_: Optional[pd.DataFrame] = None
        self.X_woe_: Optional[pd.DataFrame] = None
        self.X_test_binned_: Optional[pd.DataFrame] = None
        self.X_test_woe_: Optional[pd.DataFrame] = None

    def prefilter(
        self,
        iv_threshold: float = FILTERING.DEFAULT_IV_THRESHOLD,
        missing_threshold: float = FILTERING.DEFAULT_MISSING_THRESHOLD,
        corr_threshold: float = FILTERING.DEFAULT_CORR_THRESHOLD,
        iv_bins: int = BINNING.DEFAULT_BUCKETS,
        **kwargs,
    ) -> "ScorecardPipeline":
        """
        Apply pre-filtering based on IV, missing rate, and correlation.

        Parameters
        ----------
        iv_threshold : float
            Minimum IV threshold. Default 0.02.
        missing_threshold : float
            Maximum missing rate. Default 0.9.
        corr_threshold : float
            Maximum correlation allowed. Default 0.8.
        iv_bins : int
            Bins for IV calculation. Default 10.

        Returns
        -------
        ScorecardPipeline
            Self for chaining.
        """
        from newt.features.selection.selector import FeatureSelector

        self.prefilter_ = FeatureSelector(
            iv_bins=iv_bins,
            metrics=[
                "iv",
                "missing_rate",
                "correlation",
            ],  # Explicitly request required metrics
            **kwargs,
        )

        # Fit to calculate stats
        self.prefilter_.fit(self.X_current, self.y_train)

        # Apply selection
        self.prefilter_.select(
            iv_threshold=iv_threshold,
            missing_threshold=missing_threshold,
            corr_threshold=corr_threshold,
        )

        # Transform data
        self.X_current = self.prefilter_.transform(self.X_current)

        # Apply same filtering to test data
        if self.X_test_current is not None:
            self.X_test_current = self.prefilter_.transform(self.X_test_current)

        self.steps_.append("prefilter")
        return self

    def bin(
        self,
        method: str = "opt",
        n_bins: int = BINNING.DEFAULT_N_BINS,
        cols: Optional[List[str]] = None,
        **kwargs,
    ) -> "ScorecardPipeline":
        """
        Apply binning to features.

        Parameters
        ----------
        method : str
            Binning method: 'chi', 'dt', 'kmean', 'quantile', 'step', 'opt'.
            Default 'opt'.
        n_bins : int
            Number of bins. Default 5.
        cols : List[str], optional
            Columns to bin. Default all numeric.

        Returns
        -------
        ScorecardPipeline
            Self for chaining.
        """
        from newt.features.binning import Binner

        self.binner_ = Binner()
        self.binner_.fit(
            self.X_current,
            self.y_train,
            method=method,
            n_bins=n_bins,
            cols=cols,
            **kwargs,
        )

        self.X_binned_ = self.binner_.transform(self.X_current, labels=False)

        # Transform test data
        if self.X_test_current is not None:
            self.X_test_binned_ = self.binner_.transform(
                self.X_test_current, labels=False
            )

        self.steps_.append("bin")
        return self

    def woe_transform(
        self,
        epsilon: float = BINNING.DEFAULT_EPSILON,
        **kwargs,
    ) -> "ScorecardPipeline":
        """
        Apply WOE transformation to binned features.

        Parameters
        ----------
        epsilon : float
            Smoothing factor. Default 1e-8.

        Returns
        -------
        ScorecardPipeline
            Self for chaining.
        """
        from newt.features.analysis.woe_calculator import WOEEncoder

        if self.X_binned_ is None:
            raise ValueError("Must call bin() before woe_transform().")

        self.X_woe_ = self.X_binned_.copy()

        for col in self.X_binned_.columns:
            encoder = WOEEncoder(epsilon=epsilon)
            # Use binned codes as categories
            X_col = self.X_binned_[col].astype(str)
            encoder.fit(X_col, self.y_train)

            self.woe_encoders_[col] = encoder
            self.X_woe_[col] = encoder.transform(X_col)

        # Transform test data
        if self.X_test_binned_ is not None:
            self.X_test_woe_ = self.X_test_binned_.copy()
            for col in self.X_test_binned_.columns:
                if col in self.woe_encoders_:
                    X_col = self.X_test_binned_[col].astype(str)
                    self.X_test_woe_[col] = self.woe_encoders_[col].transform(X_col)

        self.X_current = self.X_woe_
        if self.X_test_woe_ is not None:
            self.X_test_current = self.X_test_woe_

        self.steps_.append("woe_transform")
        return self

    def postfilter(
        self,
        psi_threshold: float = FILTERING.DEFAULT_PSI_THRESHOLD,
        vif_threshold: float = FILTERING.DEFAULT_VIF_THRESHOLD,
        X_test: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> "ScorecardPipeline":
        """
        Apply post-filtering based on PSI and VIF.

        Parameters
        ----------
        psi_threshold : float
            Maximum PSI threshold. Default 0.25.
        vif_threshold : float
            Maximum VIF threshold. Default 10.0.
        X_test : pd.DataFrame, optional
            Test data for PSI. Uses pipeline's X_test_current if not provided.

        Returns
        -------
        ScorecardPipeline
            Self for chaining.
        """
        from newt.features.selection.postfilter import PostFilter

        test_data = X_test if X_test is not None else self.X_test_current

        self.postfilter_ = PostFilter(
            psi_threshold=psi_threshold,
            vif_threshold=vif_threshold,
            **kwargs,
        )

        self.X_current = self.postfilter_.fit_transform(self.X_current, test_data)

        # Apply same filtering to test data
        if self.X_test_current is not None:
            self.X_test_current = self.postfilter_.transform(self.X_test_current)

        self.steps_.append("postfilter")
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
        """
        Apply stepwise regression feature selection.

        Parameters
        ----------
        direction : str
            Selection direction: 'forward', 'backward', or 'both'.
            Default 'both'.
        criterion : str
            Selection criterion: 'pvalue', 'aic', or 'bic'.
            Default 'aic'.
        p_enter : float
            P-value threshold for entering. Default 0.05.
        p_remove : float
            P-value threshold for removing. Default 0.10.
        exclude : List[str], optional
            Features to force include.

        Returns
        -------
        ScorecardPipeline
            Self for chaining.
        """
        from newt.features.selection.stepwise import StepwiseSelector

        self.stepwise_ = StepwiseSelector(
            direction=direction,
            criterion=criterion,
            p_enter=p_enter,
            p_remove=p_remove,
            exclude=exclude,
            **kwargs,
        )

        self.X_current = self.stepwise_.fit_transform(self.X_current, self.y_train)

        # Apply same filtering to test data
        if self.X_test_current is not None:
            self.X_test_current = self.stepwise_.transform(self.X_test_current)

        self.steps_.append("stepwise")
        return self

    def build_model(
        self,
        fit_intercept: bool = True,
        **kwargs,
    ) -> "ScorecardPipeline":
        """
        Build logistic regression model.

        Parameters
        ----------
        fit_intercept : bool
            Whether to fit intercept. Default True.

        Returns
        -------
        ScorecardPipeline
            Self for chaining.
        """
        from newt.modeling.logistic import LogisticModel

        self.model_ = LogisticModel(fit_intercept=fit_intercept, **kwargs)
        self.model_.fit(self.X_current, self.y_train)

        self.steps_.append("build_model")
        return self

    def generate_scorecard(
        self,
        base_score: int = SCORECARD.DEFAULT_BASE_SCORE,
        pdo: int = SCORECARD.DEFAULT_PDO,
        base_odds: float = SCORECARD.DEFAULT_BASE_ODDS,
        **kwargs,
    ) -> "ScorecardPipeline":
        """
        Generate scorecard from model.

        Parameters
        ----------
        base_score : int
            Base score. Default 600.
        pdo : int
            Points to double odds. Default 50.
        base_odds : float
            Base odds ratio. Default 1/15.

        Returns
        -------
        ScorecardPipeline
            Self for chaining.
        """
        from newt.modeling.scorecard import Scorecard

        if self.model_ is None:
            raise ValueError("Must call build_model() before generate_scorecard().")
        if self.binner_ is None:
            raise ValueError("Must call bin() before generate_scorecard().")

        self.scorecard_ = Scorecard(
            base_score=base_score,
            pdo=pdo,
            base_odds=base_odds,
        )
        self.scorecard_.from_model(self.model_, self.binner_, self.woe_encoders_)

        self.steps_.append("generate_scorecard")
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """
        Calculate scores for new data.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature data.

        Returns
        -------
        pd.Series
            Calculated scores.
        """
        if self.scorecard_ is None:
            raise ValueError("Scorecard not built. Call generate_scorecard() first.")

        return self.scorecard_.score(X)

    # Convenience properties for accessing results
    @property
    def prefilter_result(self) -> Any:
        """Get prefilter result."""
        return self.prefilter_

    @property
    def binner(self) -> Any:
        """Get binner."""
        return self.binner_

    @property
    def woe_encoders(self) -> Dict[str, Any]:
        """Get WOE encoders."""
        return self.woe_encoders_

    @property
    def postfilter_result(self) -> Any:
        """Get postfilter result."""
        return self.postfilter_

    @property
    def model(self) -> Any:
        """Get fitted model."""
        return self.model_

    @property
    def scorecard(self) -> Any:
        """Get scorecard."""
        return self.scorecard_

    @property
    def selected_features(self) -> List[str]:
        """Get final selected features."""
        return self.X_current.columns.tolist()

    def summary(self) -> Dict[str, Any]:
        """
        Get pipeline summary.

        Returns
        -------
        Dict
            Summary of pipeline steps and results.
        """
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
