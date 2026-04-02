"""Result objects for feature analysis and feature selection."""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import pandas as pd


@dataclass
class FeatureAnalysisResult:
    """Stable container for feature analysis output."""

    summary: pd.DataFrame
    corr_matrix: pd.DataFrame
    metrics: FrozenSet[str]

    def report(
        self,
        selected_features: Optional[List[str]] = None,
        removed_features: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Return the analysis table with selection status columns."""
        if self.summary.empty:
            return pd.DataFrame()

        selected = set(selected_features or self.summary["feature"].tolist())
        removed = removed_features or {}

        report = self.summary.copy()
        report["status"] = report["feature"].apply(
            lambda feature: "selected" if feature in selected else "removed"
        )
        report["reason"] = report["feature"].apply(
            lambda feature: removed.get(feature, "")
        )
        return report


@dataclass
class FeatureSelectionResult:
    """Stable container for feature filtering decisions."""

    selected_features: List[str] = field(default_factory=list)
    removed_features: Dict[str, str] = field(default_factory=dict)
    corr_removed: List[Tuple[str, str, float]] = field(default_factory=list)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe with the selected features only."""
        available = [column for column in self.selected_features if column in X.columns]
        return X[available]
