import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.tree import DecisionTreeClassifier
from .base import BaseBinner


class DecisionTreeBinner(BaseBinner):
    """
    Bins continuous data using a Decision Tree to find optimal splits.
    """

    def __init__(
        self,
        n_bins: int = 5,
        force_monotonic: bool = False,
        min_samples_leaf: float = 0.05,
    ):
        super().__init__(n_bins=n_bins, force_monotonic=force_monotonic)
        self.min_samples_leaf = min_samples_leaf

    def _fit_splits(
        self, X: pd.Series, y: Optional[pd.Series] = None
    ) -> List[float]:
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
    Bins continuous data using ChiMerge algorithm (bottom-up merging).
    """

    def __init__(
        self,
        n_bins: int = 5,
        force_monotonic: bool = False,
        init_bins: int = 50,
        confidence_level: float = 0.95,
    ):
        super().__init__(n_bins=n_bins, force_monotonic=force_monotonic)
        self.init_bins = init_bins
        # Chi-square critical values (approximate or look up)
        # For simplicity, we implement merging until n_bins is reached OR threshold met.
        # User requested support for n_bins mainly?
        # Usually ChiMerge stops when Chi2 > Threshold.
        # Here we prioritize n_bins, but respect statistical difference if possible.
        self.confidence_level = confidence_level

    def _fit_splits(
        self, X: pd.Series, y: Optional[pd.Series] = None
    ) -> List[float]:
        if y is None:
            raise ValueError("ChiMergeBinner requires target 'y'.")

        # 1. Initial fine binning (Equal Frequency)
        # We use strict quantiles to handle large data
        try:
            _, init_splits = pd.qcut(
                X, q=self.init_bins, duplicates="drop", retbins=True
            )
            splits = list(init_splits[1:-1])
        except Exception:
            _, init_splits = pd.cut(X, bins=self.init_bins, retbins=True)
            splits = list(init_splits[1:-1])

        if not splits:
            return []

        # 2. Iterative merging
        while len(splits) + 1 > self.n_bins:
            # Create current bins
            bins = [-np.inf] + splits + [np.inf]
            binned_X = pd.cut(X, bins=bins, include_lowest=True)

            # Calculate counts
            df = pd.DataFrame({"bin": binned_X, "target": y})
            # Group by bin code/order to ensure adjacency
            # Cat codes are 0..N-1
            df["bin_code"] = df["bin"].cat.codes

            grouped = df.groupby("bin_code", observed=False)["target"].agg(
                ["count", "sum"]
            )
            grouped["bad"] = grouped["sum"]
            grouped["good"] = grouped["count"] - grouped["bad"]

            # grouped index corresponds to intervals defined by splits
            # Interval i corresponds to index i.
            # Adjacent pairs: (0,1), (1,2)...

            chi2_values = []
            indices = grouped.index

            if len(indices) < 2:
                break

            for i in range(len(indices) - 1):
                # Chi square for pair (i, i+1)
                idx1, idx2 = indices[i], indices[i + 1]

                bad1, good1 = (
                    grouped.loc[idx1, "bad"],
                    grouped.loc[idx1, "good"],
                )
                bad2, good2 = (
                    grouped.loc[idx2, "bad"],
                    grouped.loc[idx2, "good"],
                )

                total1 = bad1 + good1
                total2 = bad2 + good2

                if total1 == 0 or total2 == 0:
                    chi2 = 0  # Merge empty bins immediately
                else:
                    # Expected
                    total_bad = bad1 + bad2
                    total_good = good1 + good2
                    total_count = total1 + total2

                    exp_bad1 = total_bad * (total1 / total_count)
                    exp_good1 = total_good * (total1 / total_count)
                    exp_bad2 = total_bad * (total2 / total_count)
                    exp_good2 = total_good * (total2 / total_count)

                    # Avoid div by zero in chi calculation
                    def safe_chi(obs, exp):
                        if exp < 1e-6:
                            return 0.0
                        return (obs - exp) ** 2 / exp

                    chi2 = (
                        safe_chi(bad1, exp_bad1)
                        + safe_chi(good1, exp_good1)
                        + safe_chi(bad2, exp_bad2)
                        + safe_chi(good2, exp_good2)
                    )

                chi2_values.append(chi2)

            # Find min chi2 (most similar adjacent intervals)
            min_chi2_idx = np.argmin(chi2_values)

            # Merge interval min_chi2_idx and min_chi2_idx+1
            # Interval i ends at splits[i]. Interval i+1 ends at splits[i+1].
            # Removing split at splits[i] merges them.
            # indices correspond to bins:
            # i=0 -> bin 0 (ends at splits[0])
            # i=1 -> bin 1 (ends at splits[1])
            # ...
            # pair (i, i+1) uses boundary splits[i] as the separator.
            # So removing splits[min_chi2_idx] is correct.

            splits.pop(min_chi2_idx)

        return splits
