"""Result objects for visualization helpers."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class BinningPlotData:
    """Plot-ready binning statistics for a single feature."""

    feature: str
    stats: pd.DataFrame

    @classmethod
    def from_binner(cls, binner: object, feature: str) -> "BinningPlotData":
        """Build plot data from a fitted binner."""
        if feature not in binner.binners_:
            raise ValueError(f"Feature '{feature}' not found in binner.")

        if feature not in binner.stats_:
            binner._calculate_and_store_stats(feature)

        return cls(feature=feature, stats=binner.stats_[feature].copy())
