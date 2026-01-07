"""
Unified binning interface.

Provides a single entry point for binning features using various algorithms.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from newt.config import BINNING

from .base import BaseBinner
from .binner_mixins import BinnerIOMixin, BinnerStatsMixin, BinnerWOEMixin
from .supervised import ChiMergeBinner, DecisionTreeBinner, OptBinningBinner
from .unsupervised import EqualFrequencyBinner, EqualWidthBinner, KMeansBinner
from .woe_storage import WOEStorage


class Binner(BinnerStatsMixin, BinnerIOMixin, BinnerWOEMixin):
    """
    A unified interface for binning multiple features using various algorithms.

    Supported methods: 'chi', 'dt', 'kmean', 'quantile', 'step', 'opt'.

    Features:
    - Access binning info with binner['feature_name'] after fitting
    - Missing values are automatically binned separately
    - WOE encoders are stored for scorecard generation

    Examples
    --------
    >>> binner = Binner()
    >>> binner.fit(X, y, method='opt', n_bins=5)
    >>> print(binner['age'])  # View binning stats
    >>> binner.set_splits('age', [25, 35, 45, 55])  # Adjust splits
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
        self.woe_storage = WOEStorage()
        self.stats_: Dict[str, pd.DataFrame] = {}
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
        self._features: List[str] = []
        self._missing_label = "Missing"

    @property
    def woe_encoders_(self) -> Dict[str, Any]:
        """Get WOE encoders dictionary (for backward compatibility)."""
        return self.woe_storage.encoders_

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, str]] = None,
        method: str = "chi",
        n_bins: int = BINNING.DEFAULT_N_BINS,
        min_samples: Union[int, float, None] = None,
        cols: Optional[List[str]] = None,
        **kwargs,
    ) -> "Binner":
        """
        Fit the binning model.

        Missing values are automatically handled as a separate bin.

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
        cols : List[str]
            List of columns to bin. If None, selects all numeric columns.

        Returns
        -------
        Binner
            Fitted binner instance.
        """
        y_series = y
        if isinstance(y, str):
            y_series = X[y]
            if y in X.columns:
                X = X.drop(columns=[y])

        # Store references for later use
        self._X = X.copy()
        self._y = y_series.copy() if y_series is not None else None

        # Determine columns to bin
        if cols:
            numeric_cols = [c for c in cols if c in X.columns]
        else:
            numeric_cols = list(X.select_dtypes(include=[np.number]).columns)

        self._features = numeric_cols

        for col in numeric_cols:
            binner_cls = self.method_map.get(method)
            if not binner_cls:
                raise ValueError(f"Unknown method: {method}")

            kwargs_binner = {"n_bins": n_bins}

            if method == "dt" and min_samples is not None:
                kwargs_binner["min_samples_leaf"] = min_samples

            kwargs_binner.update(kwargs)

            binner = binner_cls(**kwargs_binner)

            # For fitting, drop missing values
            col_data = X[col]
            valid_mask = col_data.notna()

            if valid_mask.sum() == 0:
                continue

            try:
                binner.fit(col_data[valid_mask], y_series[valid_mask])
                self.binners_[col] = binner
                self.rules_[col] = binner.splits_
            except Exception as e:
                print(f"Failed to bin column {col}: {e}")
                raise e

        # Calculate and store statistics
        self._update_all_stats()

        return self

    def transform(
        self,
        X: pd.DataFrame,
        labels: bool = False,
    ) -> pd.DataFrame:
        """
        Apply binning rules to the data.

        Missing values are placed in 'Missing' bin.

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
            if col not in X_new.columns:
                continue

            col_data = X[col]
            valid_mask = col_data.notna()

            # Transform valid values
            binned = pd.Series(index=col_data.index, dtype=object)

            if valid_mask.any():
                valid_binned = binner.transform(col_data[valid_mask])

                if labels:
                    binned[valid_mask] = valid_binned.astype(str)
                else:
                    binned[valid_mask] = valid_binned.cat.codes

            # Handle missing values - separate bin
            binned[~valid_mask] = self._missing_label if labels else -1

            X_new[col] = binned

        return X_new

    def __getitem__(self, feature: str) -> pd.DataFrame:
        """
        Get binning statistics for a feature.

        Parameters
        ----------
        feature : str
            Feature name.

        Returns
        -------
        pd.DataFrame
            Binning statistics for the feature.

        Examples
        --------
        >>> binner.fit(X, y, method='opt')
        >>> print(binner['age'])
        """
        if feature not in self.binners_:
            raise KeyError(f"Feature '{feature}' not found in binner.")

        if feature not in self.stats_:
            self._calculate_and_store_stats(feature)

        return self.stats_.get(feature, pd.DataFrame())

    def __contains__(self, feature: str) -> bool:
        """Check if feature is in binner."""
        return feature in self.binners_

    def __iter__(self):
        """Iterate over feature names."""
        return iter(self._features)

    def __len__(self) -> int:
        """Number of binned features."""
        return len(self.binners_)

    def features(self) -> List[str]:
        """Get list of binned feature names."""
        return list(self.binners_.keys())

