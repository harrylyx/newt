from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

from newt._native import load_native_module

try:
    from optbinning import OptimalBinning
except ImportError:
    OptimalBinning = None

from .base import BaseBinner


def _load_rust_engine():
    """Import the compiled Rust extension."""
    return load_native_module()


def _calculate_cut_points_from_bins(bins) -> List[float]:
    """Convert ordered bin start values into split points for ``pd.cut``."""
    if len(bins) < 2:
        return []

    return [(bins[i][0] + bins[i + 1][0]) / 2 for i in range(len(bins) - 1)]


class DecisionTreeBinner(BaseBinner):
    """Discretizes continuous data using a Decision Tree to find optimal splits.

    Uses a classification tree to split the feature based on its relationship with
    the target variable. This method naturally finds boundaries that maximize
    separation between classes.

    Examples:
        >>> binner = DecisionTreeBinner(n_bins=5, min_samples_leaf=0.1)
        >>> binner.fit(X_series, y_series)
        >>> print(binner.splits_)
    """

    def __init__(
        self,
        n_bins: int = 5,
        monotonic: Union[bool, str, None] = None,
        min_samples_leaf: float = 0.05,
        **kwargs,
    ):
        """Initialize DecisionTreeBinner.

        Args:
            n_bins: Maximum number of bins (max_leaf_nodes).
            monotonic: Enforce monotonic trend.
            min_samples_leaf: Minimum fraction of samples required in a leaf.
            **kwargs: Arguments passed to BaseBinner.
        """
        super().__init__(n_bins=n_bins, monotonic=monotonic, **kwargs)
        self.min_samples_leaf = min_samples_leaf

    def _fit_splits(self, X: pd.Series, y: Optional[pd.Series] = None) -> List[float]:
        if y is None:
            raise ValueError("DecisionTreeBinner requires target 'y'.")

        # Remove NaNs for tree training
        mask = (~X.isna()) & (~y.isna())
        X_clean = X[mask].values.reshape(-1, 1)
        y_clean = y[mask].values

        if len(X_clean) == 0:
            return []

        clf = DecisionTreeClassifier(
            max_leaf_nodes=self.n_bins,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        clf.fit(X_clean, y_clean)

        # Extract thresholds
        # The tree stores thresholds in tree_.threshold
        # Only non-leaf nodes have valid thresholds (others are -2)
        thresholds = clf.tree_.threshold
        splits = [t for t in thresholds if t != -2]
        return sorted(splits)


class ChiMergeBinner(BaseBinner):
    """Discretizes continuous data using the ChiMerge algorithm.

    ChiMerge is a bottom-up merging algorithm that starts with each unique value
    as a bin and iteratively merges adjacent bins if they are statistically
    similar (based on Chi-square test).

    Examples:
        >>> binner = ChiMergeBinner(n_bins=5, alpha=0.05)
        >>> binner.fit(X_series, y_series)
    """

    def __init__(
        self,
        n_bins: int = 5,
        monotonic: Union[bool, str, None] = None,
        alpha: float = 0.05,
        min_samples: float = 0.05,
        **kwargs,
    ):
        """Initialize ChiMergeBinner.

        Args:
            n_bins: Target number of bins.
            monotonic: Enforce monotonic trend.
            alpha: Significance level for Chi-square test (merges if p > alpha).
            min_samples: Minimum fraction of samples per bin (not yet enforced
                in core loop).
            **kwargs: Arguments passed to BaseBinner.
        """
        super().__init__(n_bins=n_bins, monotonic=monotonic, **kwargs)
        self.alpha = alpha
        self.min_samples = min_samples

    def _fit_splits(self, X: pd.Series, y: Optional[pd.Series] = None) -> List[float]:
        """
        Fast ChiMerge Implementation.
        """
        if y is None:
            raise ValueError("ChiMergeBinner requires target 'y'.")

        # 1. Prepare data
        df = pd.DataFrame({"X": X, "y": y}).dropna()
        X_arr = df["X"].to_numpy(dtype=np.float64)
        y_arr = df["y"].to_numpy(dtype=np.int64)

        if len(X_arr) == 0:
            return []

        threshold = float(stats.chi2.ppf(1 - self.alpha, 1))

        # 2. Try Rust engine first
        rust_module = _load_rust_engine()
        if rust_module and hasattr(rust_module, "calculate_chi_merge_numpy"):
            try:
                splits = rust_module.calculate_chi_merge_numpy(
                    X_arr,
                    y_arr,
                    self.n_bins,
                    threshold,
                )
                return sorted(splits)
            except Exception:
                # Fallback to Python if Rust fails
                pass

        # 3. Initial binning for Python fallback
        sort_idx = np.argsort(X_arr)
        X_sorted = X_arr[sort_idx]
        y_sorted = y_arr[sort_idx]
        unique_vals, counts = np.unique(X_sorted, return_counts=True)

        event_counts = []
        start = 0
        for count in counts:
            end = start + count
            event_counts.append(np.sum(y_sorted[start:end]))
            start = end

        bins = list(zip(unique_vals, counts, event_counts))

        # 4. Merge iterations (Python fallback)
        chi_squares = self._compute_chi_squares(bins)
        max_bins = self.n_bins

        while len(bins) > max_bins:
            if len(chi_squares) == 0:
                break

            min_chi2 = np.min(chi_squares)
            if min_chi2 >= threshold:
                break

            min_idx = np.argmin(chi_squares)
            bins = self._merge_bins(bins, min_idx)
            chi_squares = self._compute_chi_squares(bins)

        # 5. Extract splits
        return _calculate_cut_points_from_bins(bins)

    def _compute_chi_squares(self, bins):
        if len(bins) < 2:
            return np.array([])

        n_bins = len(bins)
        chi_squares = np.zeros(n_bins - 1)

        for i in range(n_bins - 1):
            n1, e1 = bins[i][1], bins[i][2]
            n2, e2 = bins[i + 1][1], bins[i + 1][2]

            total_n = n1 + n2
            total_e = e1 + e2
            total_ne = total_n - total_e

            if total_n == 0:
                chi_squares[i] = 0
                continue

            e1_expected = n1 * total_e / total_n
            e2_expected = n2 * total_e / total_n
            ne1_expected = n1 * total_ne / total_n
            ne2_expected = n2 * total_ne / total_n

            # Add eps to avoid div by zero
            e1_expected = max(e1_expected, 1e-9)
            e2_expected = max(e2_expected, 1e-9)
            ne1_expected = max(ne1_expected, 1e-9)
            ne2_expected = max(ne2_expected, 1e-9)

            chi2 = (
                (abs(e1 - e1_expected) - 0.5) ** 2 / e1_expected
                + (abs(e2 - e2_expected) - 0.5) ** 2 / e2_expected
                + (abs(n1 - e1 - ne1_expected) - 0.5) ** 2 / ne1_expected
                + (abs(n2 - e2 - ne2_expected) - 0.5) ** 2 / ne2_expected
            )
            chi_squares[i] = chi2

        return chi_squares

    def _merge_bins(self, bins, idx):
        val1, n1, e1 = bins[idx]
        val2, n2, e2 = bins[idx + 1]

        merged = (val1, n1 + n2, e1 + e2)
        new_bins = bins[:idx] + [merged] + bins[idx + 2 :]
        return new_bins


class OptBinningBinner(BaseBinner):
    """Discretizes continuous data using the `optbinning` library.

    Provides a wrapper for the Optimal Binning algorithm which uses constrained
    programming to find splits that optimize information value (IV).

    Note: Requires `optbinning` and is only available on Python < 3.12.

    Examples:
        >>> binner = OptBinningBinner(n_bins=5, monotonic='ascending')
        >>> binner.fit(X, y)
    """

    def __init__(
        self,
        n_bins: int = 5,
        monotonic: Union[bool, str, None] = None,
        **kwargs,
    ):
        """Initialize OptBinningBinner.

        Args:
            n_bins: Maximum number of bins.
            monotonic: Monotonic constraint setting.
            **kwargs: Arguments passed to `optbinning.OptimalBinning`.
        """
        # OptBinning handles monotonicity internally, so we don't pass to base
        super().__init__(n_bins=n_bins, monotonic=None)
        self.monotonic_setting = monotonic
        self.kwargs = kwargs

    def _fit_splits(self, X: pd.Series, y: Optional[pd.Series] = None) -> List[float]:
        if OptimalBinning is None:
            raise ImportError(
                "optbinning is not installed. "
                "Install the optional dependency with "
                '`pip install "newt[optbinning]"`.'
            )

        if y is None:
            raise ValueError("OptBinningBinner requires target 'y'.")

        # Map monotonic parameter to OptBinning's monotonic_trend
        if self.monotonic_setting is None or self.monotonic_setting is False:
            monotonic_trend = "auto"
        elif self.monotonic_setting is True or self.monotonic_setting == "auto":
            monotonic_trend = "auto_asc_desc"
        elif self.monotonic_setting == "ascending":
            monotonic_trend = "ascending"
        elif self.monotonic_setting == "descending":
            monotonic_trend = "descending"
        else:
            monotonic_trend = "auto"

        opt = OptimalBinning(
            name="feature",
            dtype="numerical",
            max_n_bins=self.n_bins,
            monotonic_trend=monotonic_trend,
            **self.kwargs,
        )

        opt.fit(X.values, y.values)

        # Get splits
        return sorted(opt.splits.tolist())
