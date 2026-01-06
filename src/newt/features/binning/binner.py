from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base import BaseBinner
from .supervised import ChiMergeBinner, DecisionTreeBinner, OptBinningBinner
from .unsupervised import EqualFrequencyBinner, EqualWidthBinner, KMeansBinner


class Binner:
    """
    A unified interface for binning multiple features using various algorithms.
    Supported methods: 'chi', 'dt', 'kmean', 'quantile', 'step', 'opt'.
    """

    def __init__(self):
        self.rules_: Dict[str, List[float]] = {}
        self.method_map = {
            "chi": ChiMergeBinner,
            "dt": DecisionTreeBinner,
            "kmean": KMeansBinner,
            "quantile": EqualFrequencyBinner,
            "step": EqualWidthBinner,
            "opt": OptBinningBinner,
        }
        self.binners_: Dict[str, BaseBinner] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, str]] = None,
        method: str = "chi",
        n_bins: int = 5,
        min_samples: Union[int, float, None] = None,
        empty_separate: bool = False,
        cols: Optional[List[str]] = None,
        **kwargs,
    ) -> "Binner":
        """
        Fit the binning model.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be binned.
        y : str or pd.Series, optional
            Target data. Required for supervised methods ('chi', 'dt').
        method : str
            Binning method. 'chi', 'dt', 'kmean', 'quantile', 'step', 'opt'.
        n_bins : int
            Number of bins.
        min_samples : int, float
            Minimum samples per leaf (for decision tree).
        empty_separate : bool
            Whether to separate empty values - Not implemented yet
            (handled by pd.cut usually).
        cols : List[str]
            List of columns to bin. If None, selects all numeric columns.
        """
        if isinstance(y, str):
            y = X[y]
            if y.name in X.columns:
                X = X.drop(columns=[y.name])

        # Determine columns to bin
        if cols:
            # Use provided columns (filtered by existing columns)
            numeric_cols = [c for c in cols if c in X.columns]
            # Check for missing requested cols? Optional warn.
        else:
            # Select numeric columns automatically
            numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            binner_cls = self.method_map.get(method)
            if not binner_cls:
                raise ValueError(f"Unknown method ensure: {method}")

            # Instantiate binner
            # Adjust params based on method
            kwargs_binner = {"n_bins": n_bins}

            # Specific params
            if method == "dt" and min_samples is not None:
                kwargs_binner["min_samples_leaf"] = min_samples

            # Merge with user kwargs
            kwargs_binner.update(kwargs)

            binner = binner_cls(**kwargs_binner)

            # Fit
            try:
                binner.fit(X[col], y)
                self.binners_[col] = binner
                self.rules_[col] = binner.splits_
            except Exception as e:
                print(f"Failed to bin column {col}: {e}")
                raise e

        return self

    def transform(self, X: pd.DataFrame, labels: bool = False) -> pd.DataFrame:
        """
        Apply binning rules to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.
        labels : bool
            If True, return interval string (e.g., '(0, 10]').
            If False, return integer code (0, 1, 2...).

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        X_new = X.copy()
        for col, binner in self.binners_.items():
            if col in X_new.columns:
                # BaseBinner.transform returns Categorical (Intervals)
                binned = binner.transform(X[col])

                if labels:
                    # Convert to string or keep as interval
                    X_new[col] = binned.astype(str)
                else:
                    # Return codes
                    X_new[col] = binned.cat.codes

        return X_new

    def export(self) -> Dict[str, List[float]]:
        """Export binning rules."""
        return self.rules_

    def load(self, rules: Dict[str, List[float]]):
        """Load binning rules manually."""
        self.rules_ = rules
        self.binners_ = {}

        # We need to reconstruct binners to use them for transform
        # We can use BaseBinner with set_splits
        # We'll use EqualWidthBinner as a generic container since it inherits BaseBinner
        # and doesn't enforce strict logic on transform other than splits.
        for col, splits in rules.items():
            binner = EqualWidthBinner()
            binner.set_splits(splits)
            self.binners_[col] = binner

        return self
