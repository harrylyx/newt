"""
Batch WOE transformer for multi-column operations.
"""

from typing import Any, Dict

import pandas as pd

from .woe_calculator import WOEEncoder


class WOETransformer:
    """
    Multi-column WOE transformer.

    Transforms multiple binned features into WOE values based on a target variable.
    """

    def __init__(self):
        self.woe_maps_: Dict[str, Dict[Any, float]] = {}
        self.ivs_: Dict[str, float] = {}
        self.encoders_: Dict[str, WOEEncoder] = {}
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        """
        Fit the transformer to multiple features.

        Args:
            X: Input DataFrame (binned/categorical features).
            y: Binary target series.

        Returns:
            WOETransformer: Fitted transformer.
        """
        self.woe_maps_ = {}
        self.ivs_ = {}
        self.encoders_ = {}
        self.is_fitted_ = False

        for col in X.columns:
            encoder = WOEEncoder()
            encoder.fit(X[col], y)
            self.encoders_[col] = encoder
            self.woe_maps_[col] = encoder.woe_map_
            self.ivs_[col] = encoder.iv_

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply WOE transformation to multiple columns.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: WOE-encoded DataFrame.
        """
        X_new = X.copy()
        for col, encoder in self.encoders_.items():
            if col in X_new.columns:
                X_new[col] = encoder.transform(X_new[col])
        return X_new

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    @property
    def iv_table(self) -> pd.DataFrame:
        """Get IV values for all features as a sorted DataFrame."""
        records = [{"feature": feat, "iv": iv} for feat, iv in self.ivs_.items()]
        if not records:
            return pd.DataFrame(columns=["feature", "iv"])
        return pd.DataFrame(records).sort_values("iv", ascending=False)

    def export(self) -> Dict[str, Dict[str, Any]]:
        """Export WOE mappings and IVs."""
        return {
            feat: {"woe": self.woe_maps_[feat], "iv": self.ivs_[feat]}
            for feat in self.woe_maps_
        }

    def load(self, rules: Dict[str, Dict[str, Any]]) -> "WOETransformer":
        """Load WOE mappings and IVs."""
        self.woe_maps_ = {}
        self.ivs_ = {}
        self.encoders_ = {}

        for col, config in rules.items():
            if "woe_map" in config and "woe" not in config:
                raise ValueError(
                    "Legacy key 'woe_map' is unsupported. "
                    "Use 'woe' in WOETransformer.load()."
                )

            woe_map = config.get("woe", {})
            iv = config.get("iv", 0.0)

            self.woe_maps_[col] = woe_map
            self.ivs_[col] = iv

            encoder = WOEEncoder()
            encoder.woe_map_ = woe_map
            encoder.iv_ = iv
            encoder.is_fitted_ = True
            self.encoders_[col] = encoder

        self.is_fitted_ = True
        return self
