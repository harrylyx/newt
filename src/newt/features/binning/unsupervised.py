import pandas as pd
import numpy as np
from typing import List, Optional
from .base import BaseBinner

class EqualWidthBinner(BaseBinner):
    """
    Bins continuous data into intervals of equal size (width).
    """
    def _fit_splits(self, X: pd.Series, y: Optional[pd.Series] = None) -> List[float]:
        # Use pd.cut with retbins to get splits including edges
        _, bins = pd.cut(X, bins=self.n_bins, retbins=True)
        # bins includes min and max. We only need internal splits.
        # bins is array([min, s1, s2, ..., max])
        # We need [s1, s2, ..., sn-1]
        # BaseBinner transforms using [-inf] + splits + [inf]
        # So we return the internal boundaries.
        if len(bins) <= 2:
            return []
        return list(bins[1:-1])

class EqualFrequencyBinner(BaseBinner):
    """
    Bins continuous data into intervals with equal number of samples (quantiles).
    """
    def _fit_splits(self, X: pd.Series, y: Optional[pd.Series] = None) -> List[float]:
        # Use pd.qcut
        try:
            _, bins = pd.qcut(X, q=self.n_bins, duplicates='drop', retbins=True)
        except Exception:
            # Fallback to cut if qcut fails (e.g. all same values)
            _, bins = pd.cut(X, bins=self.n_bins, retbins=True)
            
        if len(bins) <= 2:
            return []
        return list(bins[1:-1])
