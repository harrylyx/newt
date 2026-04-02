"""Build scorecard specifications from fitted model components."""

from typing import Any, Dict, Tuple

import numpy as np

from newt.results import BinningRuleSpec, FeatureScoreSpec, ScorecardSpec


class ScorecardBuilder:
    """Build a reusable scorecard specification from fitted components."""

    def __init__(self, base_score: int, pdo: int, base_odds: float):
        self.base_score = base_score
        self.pdo = pdo
        self.base_odds = base_odds
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def build(
        self,
        model: Any,
        binner: Any,
        woe_encoder: Any,
    ) -> Tuple[ScorecardSpec, Dict[str, float]]:
        """Build a scorecard spec from fitted model components."""
        intercept, coefficients = self._extract_model_parameters(model)
        intercept_points = self.offset - self.factor * intercept

        feature_scores = {}
        binning_rules = {}
        feature_names = []

        for feature, coefficient in coefficients.items():
            if not self._has_binning_rule(binner, feature):
                continue

            woe_map = self._get_woe_map(woe_encoder, feature)
            if not woe_map:
                continue

            rows = []
            for bin_label, woe in woe_map.items():
                rows.append(
                    {
                        "feature": feature,
                        "bin": str(bin_label),
                        "woe": float(woe),
                        "points": float(-woe * coefficient * self.factor),
                    }
                )

            feature_scores[feature] = FeatureScoreSpec(feature=feature, rows=rows)
            binning_rules[feature] = BinningRuleSpec(
                feature=feature,
                splits=self._get_splits(binner, feature),
                missing_label=getattr(binner, "_missing_label", "Missing"),
            )
            feature_names.append(feature)

        spec = ScorecardSpec(
            base_score=self.base_score,
            pdo=self.pdo,
            base_odds=self.base_odds,
            factor=self.factor,
            offset=self.offset,
            intercept_points=intercept_points,
            feature_names=feature_names,
            feature_scores=feature_scores,
            binning_rules=binning_rules,
        )
        return spec, coefficients

    def _extract_model_parameters(self, model: Any) -> Tuple[float, Dict[str, float]]:
        """Extract intercept and feature coefficients from supported models."""
        if hasattr(model, "to_dict"):
            model_dict = model.to_dict()
        elif isinstance(model, dict):
            model_dict = model
        else:
            raise ValueError("Model must be LogisticModel or dict with coefficients.")

        intercept = float(model_dict.get("intercept", 0.0))
        coefficients = {
            feature: float(coef)
            for feature, coef in model_dict.get("coefficients", {}).items()
        }
        return intercept, coefficients

    def _has_binning_rule(self, binner: Any, feature: str) -> bool:
        """Check whether a feature has fitted binning rules."""
        if hasattr(binner, "rules_") and feature in binner.rules_:
            return True
        if hasattr(binner, "binners_") and feature in binner.binners_:
            return True
        return False

    def _get_splits(self, binner: Any, feature: str):
        """Get fitted splits for a feature."""
        if hasattr(binner, "get_splits"):
            return binner.get_splits(feature)
        if hasattr(binner, "rules_"):
            return list(binner.rules_.get(feature, []))
        if hasattr(binner, "binners_") and feature in binner.binners_:
            return list(getattr(binner.binners_[feature], "splits_", []))
        return []

    def _get_woe_map(self, woe_encoder: Any, feature: str) -> Dict[str, float]:
        """Get the WOE mapping for a feature."""
        if isinstance(woe_encoder, dict) and feature in woe_encoder:
            return dict(woe_encoder[feature].woe_map_)
        if hasattr(woe_encoder, "woe_map_"):
            return dict(woe_encoder.woe_map_)
        return {}
