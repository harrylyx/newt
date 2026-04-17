"""Build scorecard specifications from fitted model components."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

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
    ) -> Tuple[
        ScorecardSpec,
        Dict[str, float],
        Dict[str, Dict[str, float]],
        Dict[str, float],
        Dict[str, Any],
    ]:
        """Build a scorecard spec from fitted model components."""
        intercept, coefficients = self._extract_model_parameters(model)
        feature_statistics = self._extract_feature_statistics(model, coefficients)
        model_statistics = self._extract_model_statistics(model)
        lr_parameters = self._extract_lr_parameters(model)
        intercept_points = self.offset - self.factor * intercept

        feature_scores = {}
        binning_rules = {}
        feature_names = []

        for feature, coefficient in coefficients.items():
            if not self._has_binning_rule(binner, feature):
                continue

            woe_map = self._get_woe_map(binner, feature)
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
            feature_statistics=feature_statistics,
            model_statistics=model_statistics,
            lr_parameters=lr_parameters,
        )
        return spec, coefficients, feature_statistics, model_statistics, lr_parameters

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

    def _get_woe_map(self, binner: Any, feature: str) -> Dict[str, float]:
        """Get the WOE mapping for a feature."""
        if hasattr(binner, "get_woe_map"):
            return dict(binner.get_woe_map(feature))
        if hasattr(binner, "woe_maps_") and feature in binner.woe_maps_:
            return dict(binner.woe_maps_[feature])
        return {}

    def _extract_feature_statistics(
        self,
        model: Any,
        coefficients: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Extract feature-level logistic statistics when available."""
        coef_frame = getattr(model, "coefficients_", pd.DataFrame())
        if not isinstance(coef_frame, pd.DataFrame) or coef_frame.empty:
            return {}
        if "feature" not in coef_frame.columns:
            return {}

        supported_columns = [
            "coefficient",
            "std_error",
            "z_value",
            "p_value",
            "ci_lower",
            "ci_upper",
            "odds_ratio",
        ]
        available_columns = [
            col for col in supported_columns if col in coef_frame.columns
        ]
        if not available_columns:
            return {}

        feature_frame = coef_frame.loc[
            coef_frame["feature"].isin(coefficients.keys())
        ].copy()
        if feature_frame.empty:
            return {}

        stats: Dict[str, Dict[str, float]] = {}
        for _, row in feature_frame.iterrows():
            feature = str(row["feature"])
            feature_stats: Dict[str, float] = {}
            for column in available_columns:
                value = row.get(column)
                numeric = self._as_finite_float(value)
                if numeric is None:
                    continue
                feature_stats[column] = numeric
            if feature_stats:
                stats[feature] = feature_stats
        return stats

    def _extract_model_statistics(self, model: Any) -> Dict[str, float]:
        """Extract model-level logistic summary statistics when available."""
        result = getattr(model, "result_", None)
        if result is None:
            return {}

        mapping = {
            "aic": "aic",
            "bic": "bic",
            "llf": "log_likelihood",
            "prsquared": "pseudo_r2",
            "nobs": "nobs",
        }
        output: Dict[str, float] = {}
        for attr_name, output_name in mapping.items():
            value = getattr(result, attr_name, None)
            numeric = self._as_finite_float(value)
            if numeric is None:
                continue
            output[output_name] = numeric
        return output

    def _extract_lr_parameters(self, model: Any) -> Dict[str, Any]:
        """Extract a compact, report-friendly LR parameter snapshot."""
        preferred = ("fit_intercept", "method", "maxiter", "regularization", "alpha")
        snapshot: Dict[str, Any] = {}

        for name in preferred:
            if not hasattr(model, name):
                continue
            value = self._as_supported_lr_param(getattr(model, name))
            if value is None:
                continue
            snapshot[name] = value

        extra_kwargs = getattr(model, "extra_kwargs", {})
        if isinstance(extra_kwargs, dict):
            for name, raw_value in extra_kwargs.items():
                key = str(name)
                if key in snapshot:
                    continue
                value = self._as_supported_lr_param(raw_value)
                if value is None:
                    continue
                snapshot[key] = value

        return snapshot

    def _as_finite_float(self, value: Any) -> Optional[float]:
        """Normalize scalar numeric values and reject non-finite entries."""
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    def _as_supported_lr_param(self, value: Any) -> Optional[Any]:
        """Keep only scalar values that are safe to serialize in scorecard payloads."""
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            if not np.isfinite(value):
                return None
            return float(value)
        if isinstance(value, str):
            return value
        return None
