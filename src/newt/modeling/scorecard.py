"""
Scorecard generation from logistic regression model.

Converts WOE-based logistic regression coefficients into a scorecard.
"""

from typing import Any, Dict, List


import numpy as np
import pandas as pd


class Scorecard:
    """
    Scorecard generator from logistic regression model.

    Converts a fitted logistic regression model with WOE features
    into a traditional credit scorecard with points for each bin.

    The scoring formula follows the standard approach:
    Score = Base Score + Sum(Points for each binned feature)

    Where:
    - Base Score = Base Odds * Factor + Offset
    - Points = -(WOE * coefficient) * Factor

    Examples
    --------
    >>> scorecard = Scorecard(base_score=600, pdo=50, base_odds=1/15)
    >>> scorecard.from_model(model, binner, woe_encoder)
    >>> scores = scorecard.score(X)
    >>> print(scorecard.export())
    """

    def __init__(
        self,
        base_score: int = 600,
        pdo: int = 50,
        base_odds: float = 1 / 15,
    ):
        """
        Initialize Scorecard.

        Parameters
        ----------
        base_score : int
            Base score at base odds. Default 600.
        pdo : int
            Points to double the odds. Default 50.
        base_odds : float
            Base odds (good/bad ratio) at base_score. Default 1/15 (~6.67% bad rate).
        """
        self.base_score = base_score
        self.pdo = pdo
        self.base_odds = base_odds

        # Calculated scaling factors
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

        # Scorecard components
        self.scorecard_: Dict[str, pd.DataFrame] = {}
        self.intercept_points_: float = 0.0
        self.feature_names_: List[str] = []
        self.is_built_: bool = False

        # Store references
        self._binner = None
        self._woe_encoder = None
        self._model_coefs: Dict[str, float] = {}

    def from_model(
        self,
        model: Any,  # LogisticModel or dict with coefficients
        binner: Any,  # Binner object
        woe_encoder: Any,  # WOEEncoder object or dict of WOEEncoders
    ) -> "Scorecard":
        """
        Build scorecard from fitted model, binner, and WOE encoder.

        Parameters
        ----------
        model : LogisticModel or Dict
            Fitted logistic model or dict with 'intercept' and 'coefficients'.
        binner : Binner
            Fitted Binner object with binning rules.
        woe_encoder : WOEEncoder or Dict[str, WOEEncoder]
            Fitted WOE encoder(s). If dict, keys are feature names.

        Returns
        -------
        Scorecard
            Built scorecard instance.
        """
        self._binner = binner
        self._woe_encoder = woe_encoder

        # Extract model coefficients
        if hasattr(model, "to_dict"):
            model_dict = model.to_dict()
            intercept = model_dict.get("intercept", 0.0)
            coefficients = model_dict.get("coefficients", {})
        elif isinstance(model, dict):
            intercept = model.get("intercept", 0.0)
            coefficients = model.get("coefficients", {})
        else:
            raise ValueError("Model must be LogisticModel or dict with coefficients.")

        self._model_coefs = coefficients
        self.feature_names_ = list(coefficients.keys())

        # Calculate intercept points
        self.intercept_points_ = self.offset - self.factor * intercept

        # Build scorecard for each feature
        for feature, coef in coefficients.items():
            self._build_feature_scorecard(feature, coef, binner, woe_encoder)

        self.is_built_ = True
        return self

    def _build_feature_scorecard(
        self,
        feature: str,
        coefficient: float,
        binner: Any,
        woe_encoder: Any,
    ):
        """Build scorecard for a single feature."""
        # Get binning rules (splits used to validate feature exists in binner)
        if hasattr(binner, "binners_") and feature in binner.binners_:
            feature_binner = binner.binners_[feature]
            _ = feature_binner.splits_  # Validate feature exists  # noqa: F841
        elif hasattr(binner, "rules_") and feature in binner.rules_:
            _ = binner.rules_[feature]  # Validate feature exists  # noqa: F841
        else:
            # Feature not in binner, skip
            return

        # Get WOE mapping
        if isinstance(woe_encoder, dict) and feature in woe_encoder:
            woe_map = woe_encoder[feature].woe_map_
        elif hasattr(woe_encoder, "woe_map_"):
            woe_map = woe_encoder.woe_map_
        else:
            # Cannot get WOE, skip
            return

        # Build records for each bin
        records = []

        for bin_label, woe in woe_map.items():
            # Points = -(WOE * coefficient) * Factor
            points = -woe * coefficient * self.factor

            records.append(
                {
                    "feature": feature,
                    "bin": str(bin_label),
                    "woe": woe,
                    "points": points,
                }
            )

        self.scorecard_[feature] = pd.DataFrame(records)

    def _create_bin_labels(self, splits: List[float]) -> List[str]:
        """Create human-readable bin labels from splits."""
        if not splits:
            return ["(-inf, inf]"]

        labels = []
        boundaries = [-np.inf] + splits + [np.inf]

        for i in range(len(boundaries) - 1):
            lower = boundaries[i]
            upper = boundaries[i + 1]

            if lower == -np.inf:
                label = f"(-inf, {upper}]"
            elif upper == np.inf:
                label = f"({lower}, inf)"
            else:
                label = f"({lower}, {upper}]"

            labels.append(label)

        return labels

    def score(self, X: pd.DataFrame) -> pd.Series:
        """
        Calculate scores for input data.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature data (not binned, not WOE transformed).

        Returns
        -------
        pd.Series
            Calculated scores.
        """
        if not self.is_built_:
            raise ValueError("Scorecard is not built. Call from_model() first.")

        scores = np.full(len(X), self.intercept_points_)

        for feature in self.feature_names_:
            if feature not in X.columns:
                continue

            feature_scores = self._score_feature(X[feature], feature)
            scores += feature_scores

        return pd.Series(scores, index=X.index, name="score")

    def _score_feature(self, x: pd.Series, feature: str) -> np.ndarray:
        """Calculate scores for a single feature."""
        # Bin the data
        if self._binner is not None and hasattr(self._binner, "binners_"):
            if feature in self._binner.binners_:
                binned = self._binner.binners_[feature].transform(x)
                binned = binned.astype(str)
            else:
                return np.zeros(len(x))
        else:
            return np.zeros(len(x))

        # Map to scores
        if feature in self.scorecard_:
            score_map = dict(
                zip(
                    self.scorecard_[feature]["bin"],
                    self.scorecard_[feature]["points"],
                )
            )
            scores = binned.map(score_map).fillna(0).values
            return scores

        return np.zeros(len(x))

    def export(self) -> pd.DataFrame:
        """
        Export scorecard as a single DataFrame.

        Returns
        -------
        pd.DataFrame
            Complete scorecard with all features.
        """
        if not self.is_built_:
            raise ValueError("Scorecard is not built. Call from_model() first.")

        all_records = []

        # Add intercept
        all_records.append(
            {
                "feature": "Intercept",
                "bin": "-",
                "woe": 0.0,
                "points": self.intercept_points_,
            }
        )

        # Add all features
        for feature, df in self.scorecard_.items():
            for _, row in df.iterrows():
                all_records.append(row.to_dict())

        return pd.DataFrame(all_records)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export scorecard configuration as dictionary.

        Returns
        -------
        Dict
            Scorecard configuration and points.
        """
        if not self.is_built_:
            raise ValueError("Scorecard is not built. Call from_model() first.")

        return {
            "base_score": self.base_score,
            "pdo": self.pdo,
            "base_odds": self.base_odds,
            "factor": self.factor,
            "offset": self.offset,
            "intercept_points": self.intercept_points_,
            "features": {
                feature: df.to_dict(orient="records")
                for feature, df in self.scorecard_.items()
            },
        }

    def summary(self) -> str:
        """
        Get scorecard summary.

        Returns
        -------
        str
            Summary string.
        """
        if not self.is_built_:
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
