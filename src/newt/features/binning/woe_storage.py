"""
WOE Encoder storage and management.

Manages WOE encoders for scorecard generation.
"""

from typing import Any, Dict, Optional

import pandas as pd


class WOEStorage:
    """
    Storage and management for WOE encoders.

    Stores WOE encoders computed during binning for later use
    in scorecard generation.
    """

    def __init__(self):
        self.encoders_: Dict[str, Any] = {}

    def store(self, feature: str, encoder: Any):
        """
        Store a WOE encoder for a feature.

        Parameters
        ----------
        feature : str
            Feature name.
        encoder : WOEEncoder
            Fitted WOE encoder.
        """
        self.encoders_[feature] = encoder

    def get(self, feature: str) -> Optional[Any]:
        """
        Get WOE encoder for a feature.

        Parameters
        ----------
        feature : str
            Feature name.

        Returns
        -------
        WOEEncoder or None
            Fitted WOE encoder if exists.
        """
        return self.encoders_.get(feature)

    def get_woe_map(self, feature: str) -> Dict[Any, float]:
        """
        Get WOE mapping dictionary for a feature.

        Parameters
        ----------
        feature : str
            Feature name.

        Returns
        -------
        Dict[Any, float]
            Mapping from bin label to WOE value.
        """
        encoder = self.get(feature)
        if encoder is None:
            return {}
        return getattr(encoder, "woe_map_", {})

    def get_iv(self, feature: str) -> float:
        """
        Get IV value for a feature.

        Parameters
        ----------
        feature : str
            Feature name.

        Returns
        -------
        float
            IV value, or 0.0 if not found.
        """
        encoder = self.get(feature)
        if encoder is None:
            return 0.0
        return getattr(encoder, "iv_", 0.0)

    def get_all_iv(self) -> pd.DataFrame:
        """
        Get IV values for all stored features.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and IV values.
        """
        records = []
        for feature, encoder in self.encoders_.items():
            iv = getattr(encoder, "iv_", 0.0)
            records.append({"feature": feature, "iv": iv})

        if not records:
            return pd.DataFrame(columns=["feature", "iv"])

        return pd.DataFrame(records).sort_values("iv", ascending=False)

    def remove(self, feature: str):
        """Remove encoder for a feature."""
        if feature in self.encoders_:
            del self.encoders_[feature]

    def clear(self):
        """Clear all stored encoders."""
        self.encoders_ = {}

    def features(self):
        """Get list of features with stored encoders."""
        return list(self.encoders_.keys())

    def __contains__(self, feature: str) -> bool:
        return feature in self.encoders_

    def __len__(self) -> int:
        return len(self.encoders_)

    def __iter__(self):
        return iter(self.encoders_)
