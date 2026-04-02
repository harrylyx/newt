"""Stable result and specification objects used across the library."""

from newt.results.scorecard import BinningRuleSpec, FeatureScoreSpec, ScorecardSpec
from newt.results.selection import FeatureAnalysisResult, FeatureSelectionResult
from newt.results.visualization import BinningPlotData

__all__ = [
    "BinningPlotData",
    "BinningRuleSpec",
    "FeatureAnalysisResult",
    "FeatureScoreSpec",
    "FeatureSelectionResult",
    "ScorecardSpec",
]
