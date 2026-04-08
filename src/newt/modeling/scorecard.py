"""Scorecard facade that builds and scores reusable specifications."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from newt.config import SCORECARD
from newt.modeling.scorecard_builder import ScorecardBuilder
from newt.modeling.scorecard_scorer import ScorecardScorer
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

    def __init__(
        self,
        base_score: int = SCORECARD.DEFAULT_BASE_SCORE,
        pdo: int = SCORECARD.DEFAULT_PDO,
        base_odds: float = SCORECARD.DEFAULT_BASE_ODDS,
    ):
        """Initialize the Scorecard instance.

        Args:
            base_score: Target score at the given base_odds.
            pdo: Points to Double the Odds (PDO).
            base_odds: Target odds at the given base_score.
        """
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
        self._woe_encoder = None
        self._model_coefs: Dict[str, float] = {}

    def from_model(
        self,
        model: Any,
        binner: Any,
        woe_encoder: Any,
    ) -> "Scorecard":
        """Build a scorecard from a fitted model and its binning/encoding artifacts.

        Args:
            model: A fitted model object (scikit-learn, statsmodels, or dict).
            binner: A fitted Binner instance or rules dictionary.
            woe_encoder: A fitted WOEEncoder instance or mapping dictionary.

        Returns:
            Scorecard: The built Scorecard instance.

        Examples:
            >>> scorecard = Scorecard(base_score=600, pdo=20)
            >>> scorecard.from_model(lr_model, binner, woe_encoders)
        """
        builder = ScorecardBuilder(
            base_score=self.base_score,
            pdo=self.pdo,
            base_odds=self.base_odds,
        )
        spec, model_coefs = builder.build(model, binner, woe_encoder)

        self._binner = binner
        self._woe_encoder = woe_encoder
        self._model_coefs = model_coefs

        return self._load_spec(spec)

    def from_dict(self, payload: Dict[str, Any]) -> "Scorecard":
        """Restore a scorecard from a serialized specification.

        Args:
            payload: A dictionary representing a serialized ScorecardSpec.

        Returns:
            Scorecard: The restored Scorecard instance.
        """
        spec = ScorecardSpec.from_dict(payload)
        return self._load_spec(spec)

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
        self.feature_names_ = list(spec.feature_names)
        self.scorecard_ = {
            feature: feature_spec.to_frame()
            for feature, feature_spec in spec.feature_scores.items()
        }
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
        return self.scorer_.score(X)

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
        return self.spec_.to_dict()

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
