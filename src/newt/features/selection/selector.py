"""Compatibility facade around feature analysis and feature filtering."""

from typing import List, Optional, Set

import pandas as pd

from newt.config import BINNING, FILTERING
from newt.features.selection.analyzer import FeatureAnalyzer
from newt.features.selection.filtering import FeatureSelectionFilter
from newt.results import FeatureAnalysisResult, FeatureSelectionResult
from newt.utils.decorators import requires_fit


class FeatureSelector:
    """Unified feature analysis and selection tool."""

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        iv_bins: int = BINNING.DEFAULT_BUCKETS,
        lift_k: float = 0.1,
        corr_method: str = "pearson",
    ):
        self._analyzer = FeatureAnalyzer(
            metrics=metrics,
            iv_bins=iv_bins,
            lift_k=lift_k,
            corr_method=corr_method,
        )
        self._filter = FeatureSelectionFilter()
        self.metrics: Set[str] = set(self._analyzer.metrics)

        self.eda_summary_: pd.DataFrame = pd.DataFrame()
        self.analysis_result_: Optional[FeatureAnalysisResult] = None

        self.selected_features_: List[str] = []
        self.removed_features_: dict = {}
        self.corr_removed_: list = []
        self.selection_result_: Optional[FeatureSelectionResult] = None
        self.is_fitted_: bool = False
        self.is_selected_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureSelector":
        """Calculate feature statistics."""
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
        """Apply filtering based on calculated statistics."""
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
