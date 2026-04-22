"""Result and specification objects for scorecard building and scoring."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric))


def _is_supported_lr_param(value: Any) -> bool:
    if isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return False


def _normalize_points_decimals(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("points_decimals must be a non-negative integer or None")
    if value < 0:
        raise ValueError("points_decimals must be a non-negative integer or None")
    return int(value)


def _extract_bin_left_boundary(bin_label: Any) -> float:
    text = str(bin_label).strip()
    if text == "":
        return float("inf")
    if "," not in text:
        return float("inf")
    left_text = (
        text.replace("[", "").replace("(", "").split(",", maxsplit=1)[0].strip().lower()
    )
    if left_text in {"-inf", "-infinity"}:
        return float("-inf")
    if left_text in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    try:
        return float(left_text)
    except (TypeError, ValueError):
        return float("inf")


@dataclass
class BinningRuleSpec:
    """Serializable description of how a feature is binned."""

    feature: str
    splits: List[float] = field(default_factory=list)
    missing_label: str = "Missing"

    def bin_series(self, values: pd.Series) -> pd.Series:
        """Apply the stored split rules to a raw feature series."""
        binned = pd.Series(index=values.index, dtype=object)
        valid_mask = values.notna()

        if valid_mask.any():
            bins = [-np.inf] + list(self.splits) + [np.inf]
            cut = pd.cut(values[valid_mask], bins=bins, include_lowest=True)
            binned[valid_mask] = cut.astype(str)

        binned[~valid_mask] = self.missing_label
        return binned

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the binning rule."""
        return {
            "feature": self.feature,
            "splits": [float(split) for split in self.splits],
            "missing_label": self.missing_label,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BinningRuleSpec":
        """Deserialize a binning rule."""
        return cls(
            feature=payload["feature"],
            splits=[float(split) for split in payload.get("splits", [])],
            missing_label=payload.get("missing_label", "Missing"),
        )


@dataclass
class FeatureScoreSpec:
    """Serializable score table for a single feature."""

    feature: str
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Return the score rows as a dataframe."""
        return pd.DataFrame(self.rows)

    def score_map(self) -> Dict[str, float]:
        """Return a mapping from bin label to points."""
        return {str(row["bin"]): float(row["points"]) for row in self.rows}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the feature score table."""
        serialized_rows: List[Dict[str, Any]] = []
        for row in self.rows:
            serialized_rows.append(
                {
                    "feature": row["feature"],
                    "bin": str(row["bin"]),
                    "woe": float(row["woe"]),
                    "points": float(row["points"]),
                }
            )
        return {
            "feature": self.feature,
            "rows": serialized_rows,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FeatureScoreSpec":
        """Deserialize a feature score table."""
        return cls(
            feature=payload["feature"],
            rows=list(payload.get("rows", [])),
        )


@dataclass
class ScorecardSpec:
    """Serializable scorecard specification."""

    base_score: int
    pdo: int
    base_odds: float
    factor: float
    offset: float
    intercept_points: float
    points_decimals: Optional[int] = None
    feature_names: List[str] = field(default_factory=list)
    feature_scores: Dict[str, FeatureScoreSpec] = field(default_factory=dict)
    binning_rules: Dict[str, BinningRuleSpec] = field(default_factory=dict)
    feature_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model_statistics: Dict[str, float] = field(default_factory=dict)
    lr_parameters: Dict[str, Any] = field(default_factory=dict)

    def export(self) -> pd.DataFrame:
        """Export the complete scorecard to a dataframe."""
        records: List[Dict[str, Any]] = [
            {
                "feature": "Intercept",
                "bin": "-",
                "woe": 0.0,
                "points": float(self.intercept_points),
            }
        ]

        for feature in self._ordered_feature_names():
            feature_spec = self.feature_scores.get(feature)
            if feature_spec is None:
                continue
            records.extend(feature_spec.rows)

        return pd.DataFrame(records)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the scorecard specification."""
        payload = {
            "base_score": self.base_score,
            "pdo": self.pdo,
            "base_odds": float(self.base_odds),
            "factor": float(self.factor),
            "offset": float(self.offset),
            "intercept_points": float(self.intercept_points),
            "feature_names": list(self.feature_names),
            "features": {
                feature: self.feature_scores[feature].to_dict()["rows"]
                for feature in self._ordered_feature_names()
                if feature in self.feature_scores
            },
            "binning_rules": {
                feature: rule.to_dict() for feature, rule in self.binning_rules.items()
            },
            "feature_statistics": {
                feature: {
                    metric: float(value)
                    for metric, value in stats.items()
                    if _is_finite_number(value)
                }
                for feature, stats in self.feature_statistics.items()
            },
            "model_statistics": {
                metric: float(value)
                for metric, value in self.model_statistics.items()
                if _is_finite_number(value)
            },
            "lr_parameters": {
                str(name): value
                for name, value in self.lr_parameters.items()
                if _is_supported_lr_param(value)
            },
        }
        if self.points_decimals is not None:
            payload["points_decimals"] = int(self.points_decimals)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScorecardSpec":
        """Deserialize a scorecard specification."""
        feature_scores = {
            feature: FeatureScoreSpec(feature=feature, rows=list(rows))
            for feature, rows in payload.get("features", {}).items()
        }
        binning_rules = {
            feature: BinningRuleSpec.from_dict(rule)
            for feature, rule in payload.get("binning_rules", {}).items()
        }
        return cls(
            base_score=int(payload["base_score"]),
            pdo=int(payload["pdo"]),
            base_odds=float(payload["base_odds"]),
            factor=float(payload["factor"]),
            offset=float(payload["offset"]),
            intercept_points=float(payload["intercept_points"]),
            points_decimals=_normalize_points_decimals(payload.get("points_decimals")),
            feature_names=list(payload.get("feature_names", [])),
            feature_scores=feature_scores,
            binning_rules=binning_rules,
            feature_statistics={
                str(feature): {
                    str(metric): float(value)
                    for metric, value in dict(stats).items()
                    if _is_finite_number(value)
                }
                for feature, stats in dict(
                    payload.get("feature_statistics", {})
                ).items()
            },
            model_statistics={
                str(metric): float(value)
                for metric, value in dict(payload.get("model_statistics", {})).items()
                if _is_finite_number(value)
            },
            lr_parameters={
                str(name): value
                for name, value in dict(payload.get("lr_parameters", {})).items()
                if _is_supported_lr_param(value)
            },
        )

    def normalize_points_precision(self) -> None:
        """Apply optional points rounding precision to intercept and bin rows."""
        decimals = _normalize_points_decimals(self.points_decimals)
        self.points_decimals = decimals
        if decimals is None:
            return

        self.intercept_points = float(np.round(self.intercept_points, decimals))
        for feature_spec in self.feature_scores.values():
            normalized_rows: List[Dict[str, Any]] = []
            for row in feature_spec.rows:
                normalized = dict(row)
                if "points" in normalized and _is_finite_number(normalized["points"]):
                    normalized["points"] = float(
                        np.round(normalized["points"], decimals)
                    )
                normalized_rows.append(normalized)
            feature_spec.rows = normalized_rows

    def normalize_feature_row_order(self) -> None:
        """Sort feature rows by bin boundary with Missing at the end."""
        for feature in self._ordered_feature_names():
            feature_spec = self.feature_scores.get(feature)
            if feature_spec is None:
                continue
            missing_label = "Missing"
            rule = self.binning_rules.get(feature)
            if rule is not None:
                missing_label = str(rule.missing_label)
            feature_spec.rows = sorted(
                [dict(row) for row in feature_spec.rows],
                key=lambda row: self._feature_row_sort_key(row, missing_label),
            )

    def _ordered_feature_names(self) -> List[str]:
        """Return feature names ordered by model list then any residual keys."""
        ordered: List[str] = []
        seen = set()
        for feature in self.feature_names:
            if feature in self.feature_scores and feature not in seen:
                ordered.append(feature)
                seen.add(feature)
        for feature in self.feature_scores.keys():
            if feature not in seen:
                ordered.append(feature)
                seen.add(feature)
        return ordered

    def _feature_row_sort_key(
        self,
        row: Dict[str, Any],
        missing_label: str,
    ) -> tuple:
        label = str(row.get("bin", ""))
        is_missing = 1 if label == str(missing_label) else 0
        left = _extract_bin_left_boundary(label)
        return (is_missing, left, label)
