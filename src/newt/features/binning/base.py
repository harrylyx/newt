import pandas as pd
import numpy as np
from typing import List, Optional
from abc import ABC, abstractmethod


class BaseBinner(ABC):
    """
    Base class for feature binning.
    Supports monotonicity adjustment and custom splits.
    """

    def __init__(self, n_bins: int = 5, force_monotonic: bool = False):
        self.n_bins = n_bins
        self.force_monotonic = force_monotonic
        self.splits_ = []  # List of break points (float)
        self.is_fitted_ = False

    @abstractmethod
    def _fit_splits(
        self, X: pd.Series, y: Optional[pd.Series] = None
    ) -> List[float]:
        """Calculate initial splits."""
        pass

    def fit(self, X: pd.Series, y: Optional[pd.Series] = None) -> "BaseBinner":
        """
        Fit the binner to the data.
        """
        X = X.copy()
        if y is not None:
            y = y.copy()

        # 1. Calculate initial splits
        splits = self._fit_splits(X, y)
        splits = sorted(list(set(splits)))  # Ensure unique and sorted

        # 2. Monotonicity adjustment if requested
        if self.force_monotonic and y is not None:
            splits = self._adjust_monotonicity(X, y, splits)

        # 3. Finalize splits
        # splits should define the upper bounds of bins (excluding infinity)
        # We usually store internal boundaries.
        # pd.cut uses bins argument.
        self.splits_ = splits
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """
        Bin the data using fitted splits.
        Returns the bin index or interval?
        Usually returning the bin index (categorical) or Interval index is useful.
        Let's return IntervalIndex for clarity, or integer codes?
        IntervalIndex is safer for mapping.
        """
        if not self.is_fitted_:
            raise ValueError("Binner is not fitted.")

        # Construct bins for pd.cut
        # internal splits + -inf/inf
        if not self.splits_:
            # No splits -> single bin?
            bins = [-np.inf, np.inf]
        else:
            bins = [-np.inf] + self.splits_ + [np.inf]

        # Use pd.cut
        return pd.cut(X, bins=bins, include_lowest=True)

    def _adjust_monotonicity(
        self, X: pd.Series, y: pd.Series, splits: List[float]
    ) -> List[float]:
        """
        Iteratively merge bins to ensure monotonic event rate.
        """
        if not splits:
            return []

        # Loop until monotonic or single bin
        current_splits = splits.copy()

        while len(current_splits) > 0:
            # Create bins from current splits
            bins = [-np.inf] + current_splits + [np.inf]
            binned_X = pd.cut(X, bins=bins, include_lowest=True)

            # Calculate event rate
            stats = pd.DataFrame({"bin": binned_X, "target": y})
            grouped = stats.groupby("bin", observed=True)["target"].mean()

            # Check monotonicity
            rates = grouped.values
            if len(rates) < 2:
                break

            # Check increasing or decreasing
            # We assume Spearman correlation determines direction,
            # or we check if it's strictly monotonic.
            # Simple check: calculate Spearman
            # If correlation is low, we might be far from monotonic.

            # Identify violation
            # For simplicity, we just check if sorted(rates) == rates.
            is_increasing = np.all(np.diff(rates) >= 0)
            is_decreasing = np.all(np.diff(rates) <= 0)

            if is_increasing or is_decreasing:
                break  # Satisfied

            # Not monotonic. Find the pair causing violation (simplest heuristic)
            # We need to decide direction. Usually determined by overall trend.
            # Let's use correlation with bin index to decide intended direction.
            # Assuming bin index 0..N
            # If target vs X is positive corr, we want increasing rates.

            # corr, _ = scipy.stats.spearmanr(...) # Deprecated input
            # Use X vs y correlation?
            # Let's assume direction from first and last bin comparison
            direction = 1 if rates[-1] > rates[0] else -1

            # Find first violation
            violation_idx = -1
            if direction == 1:  # Expect increasing
                for i in range(len(rates) - 1):
                    if rates[i] > rates[i + 1]:
                        violation_idx = i
                        break
            else:  # Expect decreasing
                for i in range(len(rates) - 1):
                    if rates[i] < rates[i + 1]:
                        violation_idx = i
                        break

            if violation_idx != -1:
                # Merge bin i and i+1
                # This means removing the split between them.
                # bins are defined by boundaries: -inf, s0, s1, ... sn, inf
                # rates[0] corresponds to (-inf, s0]
                # rates[i] corresponds to (si-1, si]
                # violation at i means bin i and bin i+1 need merge.
                # bin i: boundary at index i (in splits list).
                # Wait, splits[i] is the UPPER bound of bin i (if 0-indexed).
                # bins: -inf (idx -1), split0 (idx 0), split1 (idx 1)...
                # bin 0 uses split0 as upper.
                # bin i uses split_i as upper.
                # removing split_i merges bin i and bin i+1.

                # So we remove current_splits[violation_idx]
                current_splits.pop(violation_idx)
            else:
                # Should have found violation if not monotonic
                # Could be noise or flat regions.
                break

        return current_splits

    def set_splits(self, splits: List[float]):
        """Manually set splits."""
        self.splits_ = sorted(list(set(splits)))
        self.is_fitted_ = True
