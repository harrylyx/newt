"""Result and specification objects for scorecard building and scoring."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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
    feature_names: List[str] = field(default_factory=list)
    feature_scores: Dict[str, FeatureScoreSpec] = field(default_factory=dict)
    binning_rules: Dict[str, BinningRuleSpec] = field(default_factory=dict)

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

        for feature in self.feature_names:
            feature_spec = self.feature_scores.get(feature)
            if feature_spec is None:
                continue
            records.extend(feature_spec.rows)

        return pd.DataFrame(records)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the scorecard specification."""
        return {
            "base_score": self.base_score,
            "pdo": self.pdo,
            "base_odds": float(self.base_odds),
            "factor": float(self.factor),
            "offset": float(self.offset),
            "intercept_points": float(self.intercept_points),
            "feature_names": list(self.feature_names),
            "features": {
                feature: score_spec.to_dict()["rows"]
                for feature, score_spec in self.feature_scores.items()
            },
            "binning_rules": {
                feature: rule.to_dict()
                for feature, rule in self.binning_rules.items()
            },
        }

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
            feature_names=list(payload.get("feature_names", [])),
            feature_scores=feature_scores,
            binning_rules=binning_rules,
        )
