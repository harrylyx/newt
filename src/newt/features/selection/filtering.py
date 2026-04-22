"""Threshold-based feature filtering helpers."""

from typing import Dict, List, Set, Tuple

import pandas as pd

from newt.features.analysis.correlation import get_high_correlation_pairs
from newt.results import FeatureAnalysisResult, FeatureSelectionResult


class FeatureSelectionFilter:
    """Apply threshold-based filtering to an analysis result."""

    def __init__(self, engine: str = "auto"):
        self.engine = engine

    def select(
        self,
        analysis: FeatureAnalysisResult,
        iv_threshold: float,
        missing_threshold: float,
        corr_threshold: float,
    ) -> FeatureSelectionResult:
        """Filter a set of analyzed features and record the removal reasons."""
        self._validate_required_metrics(analysis.summary)

        summary = analysis.summary.set_index("feature")
        removed_features: Dict[str, str] = {}
        candidates = self._filter_candidates(
            summary,
            iv_threshold=iv_threshold,
            missing_threshold=missing_threshold,
            removed_features=removed_features,
        )

        corr_removed: List[Tuple[str, str, float]] = []
        selected_features = candidates
        if len(candidates) > 1:
            selected_features = self._remove_correlated(
                summary=summary,
                corr_matrix=analysis.corr_matrix,
                candidates=candidates,
                threshold=corr_threshold,
                removed_features=removed_features,
                corr_removed=corr_removed,
            )

        return FeatureSelectionResult(
            selected_features=selected_features,
            removed_features=removed_features,
            corr_removed=corr_removed,
        )

    def _validate_required_metrics(self, summary: pd.DataFrame) -> None:
        """Ensure the thresholding inputs are available."""
        if "missing_rate" not in summary.columns:
            raise ValueError(
                "Metric 'missing_rate' was not calculated. "
                "Cannot filter by missing rate."
            )
        if "iv" not in summary.columns:
            raise ValueError("Metric 'iv' was not calculated. Cannot filter by IV.")

    def _filter_candidates(
        self,
        summary: pd.DataFrame,
        iv_threshold: float,
        missing_threshold: float,
        removed_features: Dict[str, str],
    ) -> List[str]:
        """Filter features by missing rate and IV."""
        candidates: List[str] = []
        for feature in summary.index:
            missing_rate = summary.loc[feature, "missing_rate"]
            iv = summary.loc[feature, "iv"]

            if pd.isna(iv):
                removed_features[feature] = "iv_nan"
                continue

            if missing_rate > missing_threshold:
                removed_features[feature] = f"missing_rate={missing_rate:.3f}"
            elif iv < iv_threshold:
                removed_features[feature] = f"iv={iv:.4f}"
            else:
                candidates.append(feature)

        return candidates

    def _remove_correlated(
        self,
        summary: pd.DataFrame,
        corr_matrix: pd.DataFrame,
        candidates: List[str],
        threshold: float,
        removed_features: Dict[str, str],
        corr_removed: List[Tuple[str, str, float]],
    ) -> List[str]:
        """Remove highly correlated features while keeping the stronger IV."""
        valid_candidates = [column for column in candidates if column in corr_matrix]
        final_selection = [column for column in candidates if column not in corr_matrix]

        sub_matrix = corr_matrix.loc[valid_candidates, valid_candidates]
        high_corr_pairs = get_high_correlation_pairs(
            sub_matrix,
            threshold,
            engine=self.engine,
        )

        to_remove: Set[str] = set()
        corr_reasons: Dict[str, List[Tuple[str, float]]] = {}
        iv_map = summary["iv"].to_dict()

        for pair in high_corr_pairs:
            left = pair["var1"]
            right = pair["var2"]
            corr = pair["correlation"]

            if left in to_remove or right in to_remove:
                if left in to_remove:
                    corr_reasons.setdefault(left, []).append((right, corr))
                if right in to_remove:
                    corr_reasons.setdefault(right, []).append((left, corr))
                continue

            left_iv = iv_map.get(left, 0)
            right_iv = iv_map.get(right, 0)

            if left_iv >= right_iv:
                removed = right
                kept = left
            else:
                removed = left
                kept = right

            to_remove.add(removed)
            corr_removed.append((removed, kept, corr))
            corr_reasons.setdefault(removed, []).append((kept, corr))

        for removed_var, related_vars in corr_reasons.items():
            related_vars.sort(key=lambda item: abs(item[1]), reverse=True)
            reason_parts = [f"{feature}({corr:.3f})" for feature, corr in related_vars]
            removed_features[removed_var] = f"high_corr: {', '.join(reason_parts)}"

        final_selection.extend(
            [column for column in valid_candidates if column not in to_remove]
        )
        return final_selection
