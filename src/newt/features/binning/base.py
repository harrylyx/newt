from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from newt.config import BINNING

# Valid monotonic trend values
MONOTONIC_TRENDS = frozenset(["ascending", "descending", "auto"])


class BaseBinner(ABC):
    """Abstract base class for all feature discretization (binning) algorithms.

    A binner discovers optimal split points for continuous features and transforms
    them into discrete categories (bins). It supports automatic monotonicity
    adjustment to ensure binned features have a logical relationship with the target.

    Attributes:
        n_bins (int): Target number of bins.
        monotonic (Union[bool, str, None]): Monotonicity constraint setting.
        splits_ (List[float]): The internal split boundaries found during fit().
        is_fitted_ (bool): Whether the binner has been fitted.
    """

    def __init__(
        self,
        n_bins: int = BINNING.DEFAULT_N_BINS,
        monotonic: Union[bool, str, None] = None,
        **kwargs,
    ):
        """Initialize the BaseBinner.

        Args:
            n_bins: Maximum number of bins to discover.
            monotonic: Monotonicity constraint. Options: 'ascending', 'descending',
                'auto', True (alias for 'auto'), or None/False (no constraint).
            **kwargs: Additional keyword arguments.
        """
        self.n_bins = n_bins
        self.monotonic = monotonic
        self.splits_ = []  # List of break points (float)
        self.is_fitted_ = False

    @abstractmethod
    def _fit_splits(self, X: pd.Series, y: Optional[pd.Series] = None) -> List[float]:
        """Calculate the initial split points (to be implemented by subclasses).

        Args:
            X: Input numerical series.
            y: Optional target labels for supervised binning.

        Returns:
            List[float]: Internal split boundaries.
        """
        pass

    def fit(self, X: pd.Series, y: Optional[pd.Series] = None) -> "BaseBinner":
        """Fit the binner to find optimal discretization rules.

        Args:
            X: Numerical series to discretize.
            y: Optional binary target labels. Required if monotonicity or
                supervised methods are used.

        Returns:
            BaseBinner: The fitted binner instance.
        """
        X = X.copy()
        if y is not None:
            y = y.copy()

        # 1. Calculate initial splits
        splits = self._fit_splits(X, y)
        splits = sorted(list(set(splits)))  # Ensure unique and sorted

        # 2. Monotonicity adjustment if requested
        if self.monotonic and y is not None:
            splits = self._adjust_monotonicity(X, y, splits)

        # 3. Finalize splits
        self.splits_ = splits
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Apply the learned split rules to discretize new data.

        Args:
            X: Numerical series to transform.

        Returns:
            pd.Series: Categorical series containing Interval objects.

        Raises:
            ValueError: If the binner has not been fitted.
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

        Uses PAVA (Pool Adjacent Violators Algorithm) approach.
        Direction is determined by:
        - monotonic="ascending": force increasing bad rate
        - monotonic="descending": force decreasing bad rate
        - monotonic=True or "auto": auto-detect from data
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
            is_increasing = np.all(np.diff(rates) >= 0)
            is_decreasing = np.all(np.diff(rates) <= 0)

            if is_increasing or is_decreasing:
                break  # Satisfied

            # Determine intended direction
            if isinstance(self.monotonic, str):
                if self.monotonic == "ascending":
                    direction = 1
                elif self.monotonic == "descending":
                    direction = -1
                else:  # "auto"
                    direction = 1 if rates[-1] > rates[0] else -1
            else:
                # monotonic=True, auto-detect
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
                # Merge bin i and i+1 by removing the split between them
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
