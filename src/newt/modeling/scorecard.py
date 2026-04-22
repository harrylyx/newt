"""Scorecard facade that builds and scores reusable specifications."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from newt.config import SCORECARD
from newt.modeling.scorecard_builder import ScorecardBuilder
from newt.modeling.scorecard_scorer import ScorecardScorer
from newt.modeling.scorecard_sql_builder import ScorecardSQLBuilder
from newt.results import ScorecardSpec


class Scorecard:
    """Scorecard generator from logistic regression model coefficients.

    The Scorecard class converts the continuous probability output of a logistic
    regression model into an additive point-based scoring system. It manages
    scaliing parameters (base score, PDO) and provides methods for scoring new data,
    exporting definitions, and summarizing findings.

    Attributes:
        base_score (int): The target score at 'base_odds'.
        pdo (int): Points to Double the Odds.
        base_odds (float): The odds (Good:Bad) at 'base_score'.
        factor (float): Calculated scaling factor.
        offset (float): Calculated scaling offset.
    """

    SERIALIZATION_VERSION = 1

    def __init__(
        self,
        base_score: int = SCORECARD.DEFAULT_BASE_SCORE,
        pdo: int = SCORECARD.DEFAULT_PDO,
        base_odds: float = SCORECARD.DEFAULT_BASE_ODDS,
        points_decimals: Optional[int] = None,
    ):
        """Initialize the Scorecard instance.

        Args:
            base_score: Target score at the given base_odds.
            pdo: Points to Double the Odds (PDO).
            base_odds: Target odds at the given base_score.
            points_decimals: Optional decimal precision for scorecard points.
        """
        self.points_decimals = self._validate_points_decimals(points_decimals)
        self.base_score = base_score
        self.pdo = pdo
        self.base_odds = base_odds

        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

        self.scorecard_: Dict[str, pd.DataFrame] = {}
        self.intercept_points_: float = 0.0
        self.feature_names_: List[str] = []
        self.is_built_: bool = False

        self.spec_: Optional[ScorecardSpec] = None
        self.scorer_: Optional[ScorecardScorer] = None
        self._binner = None
        self._model_coefs: Dict[str, float] = {}
        self.feature_statistics_: pd.DataFrame = pd.DataFrame()
        self.model_statistics_: Dict[str, float] = {}
        self.lr_model_: Optional[Any] = None
        self.lr_parameters_: Dict[str, Any] = {}
        self.lr_snapshot_: Dict[str, Any] = {}

    def from_model(
        self,
        model: Any,
        binner: Any,
        *,
        keep_training_artifacts: bool = False,
    ) -> "Scorecard":
        """Build a scorecard from a fitted model and its binning/encoding artifacts.

        Args:
            model: A fitted model object (scikit-learn, statsmodels, or dict).
            binner: A fitted Binner instance.
            keep_training_artifacts: Whether to keep direct runtime references
                to the original model and binner objects.

        Returns:
            Scorecard: The built Scorecard instance.

        Examples:
            >>> scorecard = Scorecard(base_score=600, pdo=20)
            >>> scorecard.from_model(lr_model, binner)
        """
        builder = ScorecardBuilder(
            base_score=self.base_score,
            pdo=self.pdo,
            base_odds=self.base_odds,
        )
        (
            spec,
            model_coefs,
            feature_statistics,
            model_statistics,
            lr_parameters,
        ) = builder.build(model, binner)

        if keep_training_artifacts:
            self._binner = binner
            self.lr_model_ = model if not isinstance(model, dict) else None
        else:
            self._binner = None
            self.lr_model_ = None

        self._model_coefs = dict(model_coefs)
        spec.lr_parameters = self._build_enriched_lr_parameters(
            lr_parameters=lr_parameters,
            model_coefs=model_coefs,
            summary_text=self._extract_model_summary_text(model),
            intercept=self._estimate_intercept(spec, model),
        )
        spec.points_decimals = self.points_decimals
        self._normalize_scorecard_spec(spec)
        scorecard = self._load_spec(spec)
        scorecard.lr_snapshot_ = self._build_lr_snapshot(
            spec=spec,
            model_coefs=model_coefs,
            feature_statistics=feature_statistics,
            model_statistics=model_statistics,
        )
        return scorecard

    def from_dict(self, payload: Dict[str, Any]) -> "Scorecard":
        """Restore a scorecard from a serialized specification.

        Args:
            payload: A dictionary representing a serialized ScorecardSpec.

        Returns:
            Scorecard: The restored Scorecard instance.
        """
        spec = ScorecardSpec.from_dict(payload)
        self.lr_model_ = None
        self._binner = None
        self.points_decimals = self._validate_points_decimals(spec.points_decimals)
        self._normalize_scorecard_spec(spec)
        scorecard = self._load_spec(spec)
        scorecard.lr_snapshot_ = self._normalize_lr_snapshot(payload.get("lr_snapshot"))
        return scorecard

    def _load_spec(self, spec: ScorecardSpec) -> "Scorecard":
        """Internal helper to load a specification into the facade properties."""
        self.spec_ = spec
        self.scorer_ = ScorecardScorer(spec)
        self.base_score = spec.base_score
        self.pdo = spec.pdo
        self.base_odds = spec.base_odds
        self.factor = spec.factor
        self.offset = spec.offset
        self.intercept_points_ = spec.intercept_points
        self.points_decimals = self._validate_points_decimals(spec.points_decimals)
        self.feature_names_ = list(spec.feature_names)
        self.scorecard_ = {
            feature: feature_spec.to_frame()
            for feature, feature_spec in spec.feature_scores.items()
        }
        if spec.feature_statistics:
            self.feature_statistics_ = (
                pd.DataFrame.from_dict(spec.feature_statistics, orient="index")
                .reset_index()
                .rename(columns={"index": "feature"})
            )
        else:
            self.feature_statistics_ = pd.DataFrame()
        self.model_statistics_ = dict(spec.model_statistics)
        self.lr_parameters_ = dict(spec.lr_parameters)
        self.lr_snapshot_ = {}
        self.is_built_ = True
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Calculate scores for input raw data.

        Args:
            X: Input DataFrame containing raw (un-binned) features.

        Returns:
            pd.Series: Calculated scores for each row.

        Raises:
            ValueError: If the scorecard has not been built.
        """
        if not self.is_built_ or self.scorer_ is None:
            raise ValueError("Scorecard is not built. Call from_model() first.")
        scores = self.scorer_.score(X)
        if self.points_decimals is not None:
            rounded = np.round(scores.to_numpy(dtype=float), self.points_decimals)
            return pd.Series(rounded, index=scores.index, name=scores.name)
        return scores

    def export(self) -> pd.DataFrame:
        """Export the scorecard as a single flat DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing bin ranges and corresponding points
                for all features.
        """
        if not self.is_built_ or self.spec_ is None:
            raise ValueError("Scorecard is not built. Call from_model() first.")
        return self.spec_.export()

    def to_dict(self) -> Dict[str, Any]:
        """Export the scorecard specification as a serializable dictionary.

        Returns:
            Dict[str, Any]: The scorecard definition payload.
        """
        if not self.is_built_ or self.spec_ is None:
            raise ValueError("Scorecard is not built. Call from_model() first.")
        payload = self.spec_.to_dict()
        if self.lr_snapshot_:
            payload["lr_snapshot"] = self._normalize_lr_snapshot(self.lr_snapshot_)
        return payload

    def dump(self, path: Union[str, Path]) -> None:
        """Dump scorecard payload to a JSON file."""
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Scorecard":
        """Load scorecard from a JSON file."""
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return cls().from_dict(payload)

    def to_sql(
        self,
        table_name: str = "input_table",
        score_alias: str = "score",
        include_breakdown: bool = False,
    ) -> str:
        """Render the scorecard as an ANSI SQL scoring query.

        Args:
            table_name: Source table name used in the FROM clause.
            score_alias: Alias of the output score column.
            include_breakdown: Whether to include per-feature points columns.

        Returns:
            str: ANSI SQL query for score calculation.
        """
        if not self.is_built_ or self.spec_ is None:
            raise ValueError("Scorecard is not built. Call from_model() first.")

        return ScorecardSQLBuilder(self.spec_).build(
            table_name=table_name,
            score_alias=score_alias,
            include_breakdown=include_breakdown,
        )

    def summary(self) -> str:
        """Generate a human-readable summary of the scorecard configuration and points.

        Returns:
            str: The summary text.
        """
        if not self.is_built_ or self.spec_ is None:
            raise ValueError("Scorecard is not built. Call from_model() first.")

        lines = [
            "=" * 50,
            "Scorecard Summary",
            "=" * 50,
            f"Base Score: {self.base_score}",
            f"PDO: {self.pdo}",
            f"Base Odds: {self.base_odds:.4f}",
            f"Factor: {self.factor:.4f}",
            f"Offset: {self.offset:.4f}",
            f"Intercept Points: {self.intercept_points_:.2f}",
            f"Number of Features: {len(self.feature_names_)}",
            "-" * 50,
            "Features:",
        ]

        for feature in self.feature_names_:
            if feature in self.scorecard_:
                n_bins = len(self.scorecard_[feature])
                min_pts = self.scorecard_[feature]["points"].min()
                max_pts = self.scorecard_[feature]["points"].max()
                pts_range = f"[{min_pts:.1f}, {max_pts:.1f}]"
                lines.append(f"  {feature}: {n_bins} bins, points range {pts_range}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def _build_enriched_lr_parameters(
        self,
        lr_parameters: Dict[str, Any],
        model_coefs: Dict[str, float],
        summary_text: str,
        intercept: float,
    ) -> Dict[str, Any]:
        """Build compact scalar LR metadata for ScorecardSpec persistence."""
        enriched: Dict[str, Any] = {}
        for key, value in dict(lr_parameters).items():
            normalized = self._as_supported_lr_scalar(value)
            if normalized is None:
                continue
            enriched[str(key)] = normalized

        intercept_value = self._as_finite_float(intercept)
        if intercept_value is not None:
            enriched["intercept"] = intercept_value

        if summary_text:
            enriched["summary_text"] = summary_text

        for feature, coefficient in dict(model_coefs).items():
            numeric = self._as_finite_float(coefficient)
            if numeric is None:
                continue
            enriched[f"coef__{feature}"] = numeric
        return enriched

    def _extract_model_summary_text(self, model: Any) -> str:
        """Extract summary text from a fitted model when available."""
        if isinstance(model, dict):
            value = model.get("summary_text")
            return value if isinstance(value, str) else ""
        if hasattr(model, "summary") and callable(model.summary):
            try:
                value = model.summary()
            except Exception:
                return ""
            return value if isinstance(value, str) else str(value)
        return ""

    def _estimate_intercept(self, spec: ScorecardSpec, model: Any) -> float:
        """Estimate intercept from model payload or score scaling parameters."""
        if isinstance(model, dict):
            numeric = self._as_finite_float(model.get("intercept"))
            if numeric is not None:
                return numeric
        elif hasattr(model, "to_dict") and callable(model.to_dict):
            try:
                payload = model.to_dict()
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                numeric = self._as_finite_float(payload.get("intercept"))
                if numeric is not None:
                    return numeric

        if spec.factor == 0:
            return 0.0
        return float((spec.offset - spec.intercept_points) / spec.factor)

    def _build_lr_snapshot(
        self,
        spec: ScorecardSpec,
        model_coefs: Dict[str, float],
        feature_statistics: Dict[str, Dict[str, float]],
        model_statistics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Build lightweight LR snapshot without training samples."""
        snapshot: Dict[str, Any] = {
            "schema_version": self.SERIALIZATION_VERSION,
            "fit_intercept": bool(self.lr_parameters_.get("fit_intercept", True)),
            "method": self.lr_parameters_.get("method"),
            "maxiter": self.lr_parameters_.get("maxiter"),
            "regularization": self.lr_parameters_.get("regularization"),
            "alpha": self.lr_parameters_.get("alpha"),
            "intercept": self._as_finite_float(self.lr_parameters_.get("intercept")),
            "coefficients": {
                str(feature): float(coef)
                for feature, coef in dict(model_coefs).items()
                if self._as_finite_float(coef) is not None
            },
            "feature_names": list(spec.feature_names),
            "feature_statistics": self._normalize_feature_statistics(
                feature_statistics
            ),
            "model_statistics": self._normalize_model_statistics(model_statistics),
            "summary_text": str(self.lr_parameters_.get("summary_text", "") or ""),
        }
        if snapshot["intercept"] is None:
            snapshot["intercept"] = self._estimate_intercept(spec, {})
        return self._normalize_lr_snapshot(snapshot)

    def _normalize_feature_statistics(self, raw: Any) -> Dict[str, Dict[str, float]]:
        """Normalize nested feature statistics dictionary."""
        if not isinstance(raw, dict):
            return {}
        output: Dict[str, Dict[str, float]] = {}
        for feature, stats in raw.items():
            if not isinstance(stats, dict):
                continue
            normalized_stats: Dict[str, float] = {}
            for metric, value in stats.items():
                numeric = self._as_finite_float(value)
                if numeric is None:
                    continue
                normalized_stats[str(metric)] = numeric
            if normalized_stats:
                output[str(feature)] = normalized_stats
        return output

    def _normalize_model_statistics(self, raw: Any) -> Dict[str, float]:
        """Normalize model-level statistics dictionary."""
        if not isinstance(raw, dict):
            return {}
        output: Dict[str, float] = {}
        for metric, value in raw.items():
            numeric = self._as_finite_float(value)
            if numeric is None:
                continue
            output[str(metric)] = numeric
        return output

    def _normalize_lr_snapshot(self, raw: Any) -> Dict[str, Any]:
        """Normalize persisted LR snapshot payload."""
        if not isinstance(raw, dict):
            return {}

        coefficients = raw.get("coefficients", {})
        if not isinstance(coefficients, dict):
            coefficients = {}
        feature_names = raw.get("feature_names", [])
        if not isinstance(feature_names, list):
            feature_names = []
        summary_text = raw.get("summary_text", "")
        if not isinstance(summary_text, str):
            summary_text = str(summary_text)

        normalized = {
            "schema_version": int(
                raw.get("schema_version", self.SERIALIZATION_VERSION)
            ),
            "fit_intercept": bool(raw.get("fit_intercept", True)),
            "method": self._as_supported_lr_scalar(raw.get("method")),
            "maxiter": self._as_supported_lr_scalar(raw.get("maxiter")),
            "regularization": self._as_supported_lr_scalar(raw.get("regularization")),
            "alpha": self._as_supported_lr_scalar(raw.get("alpha")),
            "intercept": self._as_finite_float(raw.get("intercept")),
            "coefficients": {
                str(feature): float(value)
                for feature, value in coefficients.items()
                if self._as_finite_float(value) is not None
            },
            "feature_names": [str(feature) for feature in feature_names],
            "feature_statistics": self._normalize_feature_statistics(
                raw.get("feature_statistics", {})
            ),
            "model_statistics": self._normalize_model_statistics(
                raw.get("model_statistics", {})
            ),
            "summary_text": summary_text,
        }
        return normalized

    def _as_supported_lr_scalar(self, value: Any) -> Optional[Any]:
        """Keep scalar values that are safe to persist in ScorecardSpec."""
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

    def _as_finite_float(self, value: Any) -> Optional[float]:
        """Convert value to finite float when possible."""
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    def _validate_points_decimals(self, value: Optional[int]) -> Optional[int]:
        """Validate optional score decimal precision."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("points_decimals must be a non-negative integer or None")
        if value < 0:
            raise ValueError("points_decimals must be a non-negative integer or None")
        return int(value)

    def _normalize_scorecard_spec(self, spec: ScorecardSpec) -> None:
        """Normalize scorecard rows for stable ordering and optional precision."""
        spec.points_decimals = self.points_decimals
        spec.normalize_feature_row_order()
        spec.normalize_points_precision()
