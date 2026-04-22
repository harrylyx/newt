"""Compatibility facade around feature analysis and feature filtering."""

from typing import List, Optional, Set

import pandas as pd

from newt.config import BINNING, FILTERING
from newt.features.selection.analyzer import FeatureAnalyzer
from newt.features.selection.filtering import FeatureSelectionFilter
from newt.results import FeatureAnalysisResult, FeatureSelectionResult
from newt.utils.decorators import requires_fit


class FeatureSelector:
    """Unified tool for exploratory data analysis (EDA) and feature filtering.

    The FeatureSelector calculates various feature-level metrics (IV, KS, correlation,
    missing rates) and provides a simple interface to filter features based on
    business thresholds.

    Attributes:
        metrics (Set[str]): The set of metrics calculated by the selector.
        eda_summary_ (pd.DataFrame): Summary table of calculated statistics.
        selected_features_ (List[str]): List of column names that passed selection.
        removed_features_ (Dict[str, str]): Mapping of removed features to the reason.
        corr_removed_ (List[str]): List of features removed due to high correlation.

    Examples:
        >>> from newt.features.selection import FeatureSelector
        >>> selector = FeatureSelector(metrics=['iv', 'missing_rate', 'correlation'])
        >>> selector.fit(X_train, y_train)
        >>> selector.select(iv_threshold=0.02, corr_threshold=0.8)
        >>> X_filtered = selector.transform(X_train)
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        iv_bins: int = BINNING.DEFAULT_BUCKETS,
        lift_k: float = 0.1,
        corr_method: str = "pearson",
        engine: str = "auto",
    ):
        """Initialize the FeatureSelector.

        Args:
            metrics: Metrics to calculate. Options: 'iv', 'missing_rate', 'ks',
                'correlation', 'lift'. If None, uses a default set.
            iv_bins: Number of bins for initial IV calculation.
            lift_k: Fraction of population to use for Lift calculation (e.g., top 10%).
            corr_method: Correlation method ('pearson', 'spearman', 'kendall').
            engine: Execution engine ('auto', 'rust', 'python').
        """
        self._analyzer = FeatureAnalyzer(
            metrics=metrics,
            iv_bins=iv_bins,
            lift_k=lift_k,
            corr_method=corr_method,
            engine=engine,
        )
        self._filter = FeatureSelectionFilter(engine=engine)
        self.metrics: Set[str] = set(self._analyzer.metrics)
        self.engine = engine

        self.eda_summary_: pd.DataFrame = pd.DataFrame()
        self.analysis_result_: Optional[FeatureAnalysisResult] = None

        self.selected_features_: List[str] = []
        self.removed_features_: dict = {}
        self.corr_removed_: list = []
        self.selection_result_: Optional[FeatureSelectionResult] = None
        self.is_fitted_: bool = False
        self.is_selected_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureSelector":
        """Calculate feature statistics for the input DataFrame.

        Args:
            X: Input dataset.
            y: Target binary labels. Required for supervised metrics like IV or KS.

        Returns:
            FeatureSelector: The fitted selector instance.
        """
        self.analysis_result_ = self._analyzer.analyze(X, y)
        self.eda_summary_ = self.analysis_result_.summary.copy()
        self.corr_matrix_ = self.analysis_result_.corr_matrix.copy()
        self.is_fitted_ = True

        self.selected_features_ = (
            list(self.eda_summary_["feature"]) if not self.eda_summary_.empty else []
        )
        self.removed_features_ = {}
        self.corr_removed_ = []
        self.selection_result_ = FeatureSelectionResult(
            selected_features=list(self.selected_features_),
        )
        self.is_selected_ = False
        return self

    def select(
        self,
        iv_threshold: float = FILTERING.DEFAULT_IV_THRESHOLD,
        missing_threshold: float = FILTERING.DEFAULT_MISSING_THRESHOLD,
        corr_threshold: float = FILTERING.DEFAULT_CORR_THRESHOLD,
    ) -> "FeatureSelector":
        """Filter features based on thresholds for IV, missing rate, and correlation.

        Args:
            iv_threshold: Minimum Information Value (IV) to keep a feature.
            missing_threshold: Maximum missing rate (fraction) to keep a feature.
            corr_threshold: Maximum absolute correlation coefficient. If a pair
                exceeds this, the one with lower IV is removed.

        Returns:
            FeatureSelector: The selector instance after selection.

        Raises:
            ValueError: If called before fit().
        """
        if not self.is_fitted_:
            raise ValueError("FeatureSelector is not fitted. Call fit() first.")
        if self.analysis_result_ is None:
            raise ValueError("Feature analysis result is missing. Call fit() first.")

        self.selection_result_ = self._filter.select(
            analysis=self.analysis_result_,
            iv_threshold=iv_threshold,
            missing_threshold=missing_threshold,
            corr_threshold=corr_threshold,
        )
        self.selected_features_ = list(self.selection_result_.selected_features)
        self.removed_features_ = dict(self.selection_result_.removed_features)
        self.corr_removed_ = list(self.selection_result_.corr_removed)
        self.is_selected_ = True
        return self

    @requires_fit()
    def report(self) -> pd.DataFrame:
        """Generate a report combining EDA stats and selection status."""
        if self.analysis_result_ is None:
            return pd.DataFrame()

        return self.analysis_result_.report(
            selected_features=self.selected_features_,
            removed_features=self.removed_features_,
        )

    @requires_fit()
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X with only selected features."""
        if self.selection_result_ is None:
            return X[self.selected_features_]
        return self.selection_result_.transform(X)

    @property
    @requires_fit()
    def corr_matrix(self) -> pd.DataFrame:
        """Get the feature-to-feature correlation matrix."""
        if self.analysis_result_ is None:
            return pd.DataFrame()
        return self.analysis_result_.corr_matrix.copy()
