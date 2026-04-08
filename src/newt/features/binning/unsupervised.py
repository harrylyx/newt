from typing import List, Optional

import pandas as pd
from sklearn.cluster import KMeans

from .base import BaseBinner


class EqualWidthBinner(BaseBinner):
    """Discretizes continuous data into intervals of equal width.

    This method divides the range of values into 'n_bins' equal-sized intervals.
    Useful for uniform distributions or when the physical scale of the feature
    is the primary concern.

    Examples:
        >>> binner = EqualWidthBinner(n_bins=5)
        >>> binner.fit(X_series)
    """

    def __init__(self, **kwargs):
        """Initialize EqualWidthBinner.

        Args:
            **kwargs: Arguments passed to BaseBinner.
        """
        super().__init__(**kwargs)

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
    """Discretizes continuous data into intervals with an equal number of samples.

    Also known as quantile binning. This method ensures that each bin contains
    approximately the same number of observations.

    Examples:
        >>> binner = EqualFrequencyBinner(n_bins=5)
        >>> binner.fit(X_series)
    """

    def __init__(self, **kwargs):
        """Initialize EqualFrequencyBinner.

        Args:
            **kwargs: Arguments passed to BaseBinner.
        """
        super().__init__(**kwargs)

    def _fit_splits(self, X: pd.Series, y: Optional[pd.Series] = None) -> List[float]:
        # Use pd.qcut
        try:
            _, bins = pd.qcut(X, q=self.n_bins, duplicates="drop", retbins=True)
        except Exception:
            # Fallback to cut if qcut fails (e.g. all same values)
            _, bins = pd.cut(X, bins=self.n_bins, retbins=True)

        if len(bins) <= 2:
            return []
        return list(bins[1:-1])


class KMeansBinner(BaseBinner):
    """Discretizes continuous data using K-Means clustering.

    This method finds 'n_bins' clusters in the 1D space and chooses boundaries
    as the midpoints between adjacent cluster centers.

    Examples:
        >>> binner = KMeansBinner(n_bins=5)
        >>> binner.fit(X_series)
    """

    def __init__(self, **kwargs):
        """Initialize KMeansBinner.

        Args:
            **kwargs: Arguments passed to BaseBinner.
        """
        super().__init__(**kwargs)

    def _fit_splits(self, X: pd.Series, y: Optional[pd.Series] = None) -> List[float]:
        # Reshape for sklearn
        mask = ~X.isna()
        X_clean = X[mask].values.reshape(-1, 1)

        if len(X_clean) < self.n_bins:
            # Not enough data
            return []

        kmeans = KMeans(n_clusters=self.n_bins, random_state=42, n_init=10)
        kmeans.fit(X_clean)

        # The splits are usually defined as the midpoints between cluster centers.
        centers = sorted(kmeans.cluster_centers_.flatten())
        splits = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]
        return splits
