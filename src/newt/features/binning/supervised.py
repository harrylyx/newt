from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

try:
    from optbinning import OptimalBinning
except ImportError:
    OptimalBinning = None

from .base import BaseBinner


class DecisionTreeBinner(BaseBinner):
    """
    Bins continuous data using a Decision Tree to find optimal splits.
    """

    def __init__(
        self,
        n_bins: int = 5,
        monotonic: Union[bool, str, None] = None,
        min_samples_leaf: float = 0.05,
        **kwargs,
    ):
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
    """
    Bins continuous data using Fast ChiMerge algorithm.
    """

    def __init__(
        self,
        n_bins: int = 5,
        monotonic: Union[bool, str, None] = None,
        alpha: float = 0.05,
        min_samples: float = 0.05,
        **kwargs,
    ):
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
        X_arr = df["X"].values
        y_arr = df["y"].values

        if len(X_arr) == 0:
            return []

        # 2. Sort
        sort_idx = np.argsort(X_arr)
        X_sorted = X_arr[sort_idx]
        y_sorted = y_arr[sort_idx]

        # 3. Initial binning using unique values
        unique_vals, counts = np.unique(X_sorted, return_counts=True)

        # Calculate event counts for each unique value
        event_counts = []
        start = 0
        for count in counts:
            end = start + count
            event_counts.append(np.sum(y_sorted[start:end]))
            start = end

        bins = list(zip(unique_vals, counts, event_counts))

        # 4. Merge Iterations
        chi_squares = self._compute_chi_squares(bins)
        max_bins = self.n_bins

        # Threshold from user code logic:
        # while len(bins) > max_bins AND min_chi < threshold
        threshold = stats.chi2.ppf(1 - self.alpha, 1)

        while len(bins) > max_bins:
            if len(chi_squares) == 0:
                break

            min_chi2 = np.min(chi_squares)

            # User Code: len(bins) > max and min_chi < threshold
            # This allows returning more bins than max_bins if they are all significant.

            if min_chi2 >= threshold:
                # If all adjacent bins are significantly different
                if len(bins) <= max_bins:
                    break
                # If still too many bins, force merge the most similar one?
                # Standard ChiMerge implementations often stop here.
                # I'll stick to user logic: stop if significant.
                if len(chi_squares) > 0 and min_chi2 >= threshold:
                    # Force merge if strictly required? User code stops.
                    break

            min_idx = np.argmin(chi_squares)
            bins = self._merge_bins(bins, min_idx)
            chi_squares = self._compute_chi_squares(bins)

        # 5. Extract splits
        # bins[i][0] is the value.
        # If we merged, the value is the representative (first one).
        # We need the cut points.
        # For unique values v1, v2... cut point is usually (v1+v2)/2 or just v1.
        # User code: `self.cut_points_ = np.array([b[0] for b in bins[:-1]])`
        # This uses the value itself as cut point.
        # np.digitize(x, cuts) means: if x < cut[0] -> bin 0.
        # So cuts should be upper bounds? or lower bounds?
        # digitize: bins[i-1] <= x < bins[i] (if right=False default).
        # user `np.digitize(X, cuts)`
        # If cuts = [10, 20], x=5 -> 0. x=10 -> 1.
        # So bins define the lower bound of the next bin.
        # This implies splits are "Start of Bin 1, Start of Bin 2..."
        # BaseBinner expects splits to be Upper Bounds of bins (for pd.cut).
        # pd.cut(x, [-inf, s1, s2, inf]).
        # If user code returns [b[0] for b in bins[:-1]], these are values in the bins.
        # E.g. bins val: 10, 20, 30.
        # splits: 10, 20.
        # digitize puts <10 in 0. 10..19 in 1. >=20 in 2.
        # So splits are Lower bounds of bin 1, bin 2...

        # We need Upper bounds for bin 0, bin 1...
        # Upper bound of bin 0 is Lower bound of bin 1.
        # So we can use the same values.

        # Taking [b[0] for b in bins[1:]] gives the start of next bin.
        # This serves as Upper Bound for current bin (conceptually).

        # User code used `bins[:-1]` which is strange for digitize usually.
        # If bins are [10, 20, 30]. digitize with [10, 20].
        # x=5 (<10) -> 0.
        # x=15 (>=10, <20) -> 1.
        # x=25 (>=20) -> 2.
        # So bin 0 is x < 10. Split 10 is upper bound of bin 0.
        # Bin 0 representative was 10? No, 10 is rep of bin 1?
        # User initialization: `bins = list(zip(unique_vals...))`
        # Bin 0 has val unique_vals[0].
        # If we return bins[:-1], we include unique_vals[0] as a split.
        # If min val is 10. Split is 10.
        # x=5 -> 0.
        # x=10 -> 1.
        # So <10 is bin 0. but min val is 10. So bin 0 is empty?
        # Correct logic for BaseBinner:
        # We want splits s1, s2... such that (-inf, s1], (s1, s2]...
        # If unique vals are 10, 20, 30.
        # Ideal splits: 15, 25.
        # User logic seems to pick exact values.
        # I will use (val_i + val_{i+1}) / 2 for cut points if possible,
        # Or just use the start of the next bin as the split.

        if len(bins) < 2:
            return []

        final_splits = []
        for i in range(len(bins) - 1):
            # Split should be betweeen bin i and bin i+1
            # bin i val: bins[i][0]
            # bin i+1 val: bins[i+1][0]
            # simple avg
            split = (bins[i][0] + bins[i + 1][0]) / 2
            final_splits.append(split)

        return final_splits

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
    """
    Bins using the OptBinning library.
    
    Supports monotonic constraints via the monotonic parameter:
    - None/False: no constraint (monotonic_trend="auto")
    - True/"auto": auto-detect direction (monotonic_trend="auto_asc_desc")
    - "ascending": force increasing bad rate (monotonic_trend="ascending")
    - "descending": force decreasing bad rate (monotonic_trend="descending")
    """

    def __init__(
        self,
        n_bins: int = 5,
        monotonic: Union[bool, str, None] = None,
        **kwargs,
    ):
        # OptBinning handles monotonicity internally, so we don't pass to base
        super().__init__(n_bins=n_bins, monotonic=None)
        self.monotonic_setting = monotonic
        self.kwargs = kwargs

    def _fit_splits(self, X: pd.Series, y: Optional[pd.Series] = None) -> List[float]:
        if OptimalBinning is None:
            raise ImportError(
                "optbinning is not installed. "
                "Please install it via `pip install optbinning`."
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
