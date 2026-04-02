"""Apply reusable scorecard specifications to raw input data."""

import numpy as np
import pandas as pd

from newt.results import FeatureScoreSpec, ScorecardSpec


class ScorecardScorer:
    """Score raw input data with a reusable scorecard spec."""

    def __init__(self, spec: ScorecardSpec):
        self.spec = spec

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Calculate scores for input data."""
        scores = np.full(len(X), self.spec.intercept_points, dtype=float)

        for feature in self.spec.feature_names:
            if feature not in X.columns:
                continue

            feature_spec = self.spec.feature_scores.get(feature)
            binning_rule = self.spec.binning_rules.get(feature)
            if feature_spec is None or binning_rule is None:
                continue

            scores += self._score_feature(X[feature], feature_spec, binning_rule)

        return pd.Series(scores, index=X.index, name="score")

    def _score_feature(
        self,
        values: pd.Series,
        feature_spec: FeatureScoreSpec,
        binning_rule,
    ) -> np.ndarray:
        """Calculate scores for a single feature."""
        binned = binning_rule.bin_series(values)
        score_map = feature_spec.score_map()
        scores = binned.astype(str).map(score_map).fillna(0.0)
        return scores.to_numpy(dtype=float)
