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


class BinningResult:
    """
    Proxy object for accessing binning results of a single feature.

    Attributes
    ----------
    stats : pd.DataFrame
        Binning statistics.
    """

    def __init__(self, binner: "Binner", feature: str):
        self._binner = binner
        self._feature = feature

        # Ensure stats are calculated
        if feature not in binner.stats_:
            binner._calculate_and_store_stats(feature)

        self.stats = binner.stats_[feature]

    def plot(
        self,
        x: str = "bin",
        y: Union[str, List[str]] = ["bad_prop"],
        secondary_y: Optional[Union[str, List[str]]] = "bad_rate",
        **kwargs,
    ):
        """
        Plot binning results for this feature.

        Parameters
        ----------
        x : str
            Column for x-axis. Default 'bin'.
        y : str or list
            Column(s) for primary y-axis (bar). Default ['bad_prop'].
        secondary_y : str or list, optional
            Column(s) for secondary y-axis (line). Default 'bad_rate'.
        **kwargs :
            Arguments passed to plot_binning_result.
        """
        from newt.visualization.binning_viz import plot_binning_result

        if self._binner._X is None or self._binner._y is None:
            print("Plotting requires X and y data to be present in Binner.")
            return

        return plot_binning_result(
            binner=self._binner,
            X=self._binner._X,
            y=self._binner._y,
            feature=self._feature,
            woe_encoder=self._binner.woe_storage.get(self._feature),
            x_col=x,
            y_col=y,
            secondary_y_col=secondary_y,
            **kwargs,
        )

    def woe_map(self) -> Dict[Any, float]:
        """Get WOE mapping for this feature."""
        return self._binner.get_woe_map(self._feature)

    def __repr__(self):
        return self.stats.__repr__()

    def _repr_html_(self):
        return self.stats._repr_html_()


class Binner(BinnerStatsMixin, BinnerIOMixin, BinnerWOEMixin):
    """
    A unified interface for binning multiple features using various algorithms.

    Supported methods: 'chi', 'dt', 'kmean', 'quantile', 'step', 'opt'.

    Features:
    - Access binning info with binner['feature_name'] -> returns BinningResult
    - Missing values are automatically binned separately
    - WOE encoders are stored for scorecard generation

    Examples
    --------
    >>> binner = Binner()
    >>> binner.fit(X, y, method='opt', n_bins=5)
    >>> binner['age'].stats         # View stats DataFrame
    >>> binner['age'].plot()        # Plot results
    >>> binner['age'].woe_map()     # View WOE map
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
        monotonic: Union[bool, str, None] = None,
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
        monotonic : bool, str, or None
            Enforce monotonic bad rate across bins.
            - None/False: no constraint
            - True/"auto": auto-detect direction and enforce monotonicity
            - "ascending": force bad rate to increase with bin value
            - "descending": force bad rate to decrease with bin value

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

            kwargs_binner = {"n_bins": n_bins, "monotonic": monotonic}

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

    def woe_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply WOE transformation to the data.

        Uses the WOE values computed during fit(). This is a convenient
        method that combines binning and WOE transformation in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            WOE-transformed data.

        Examples
        --------
        >>> binner = Binner()
        >>> binner.fit(X_train, y_train, method='opt', n_bins=5)
        >>> X_woe = binner.woe_transform(X_train)
        """
        X_new = X.copy()

        for col in self.binners_.keys():
            if col not in X_new.columns:
                continue

            # Get WOE encoder for this feature
            woe_encoder = self.woe_storage.get(col)
            if woe_encoder is None:
                continue

            # First bin the data
            col_data = X[col]
            valid_mask = col_data.notna()
            binned = pd.Series(index=col_data.index, dtype=object)

            if valid_mask.any():
                valid_binned = self.binners_[col].transform(col_data[valid_mask])
                binned[valid_mask] = valid_binned.astype(str)

            binned[~valid_mask] = self._missing_label

            # Apply WOE transformation
            X_new[col] = woe_encoder.transform(binned)

        return X_new

    def __getitem__(self, feature: str) -> Union[BinningResult, pd.DataFrame]:
        """
        Get binning result proxy for a feature.

        Parameters
        ----------
        feature : str
            Feature name.

        Returns
        -------
        BinningResult
            Proxy object with stats and plot methods.
        """
        if feature not in self.binners_:
            raise KeyError(f"Feature '{feature}' not found in binner.")

        return BinningResult(self, feature)

    def stats(self) -> Dict[str, pd.DataFrame]:
        """Get dictionary of statistics for all features."""
        try:
            from IPython.display import display

            HAS_IPYTHON = True
        except ImportError:
            HAS_IPYTHON = False

        result = {}
        for feat in self._features:
            if feat in self.binners_:
                result[feat] = self[feat].stats
                print(f"--- Binning Result: {feat} ---")

                # Render stats table
                if HAS_IPYTHON:
                    display(self[feat].stats)
                else:
                    print(self[feat].stats)

        return result

    def stats_plot(self):
        """Display stats and plot for all features."""
        try:
            from IPython.display import HTML, display

            HAS_IPYTHON = True
        except ImportError:
            HAS_IPYTHON = False

        for feat in self._features:
            if feat in self.binners_:
                print(f"--- Binning Result: {feat} ---")

                # Render stats table
                if HAS_IPYTHON:
                    display(self[feat].stats)
                else:
                    print(self[feat].stats)

                # Plot
                fig = self[feat].plot()
                if HAS_IPYTHON:
                    display(fig)
                else:
                    try:
                        import matplotlib.pyplot as plt

                        plt.show()
                    except ImportError:
                        pass

    def woe_map(self) -> Dict[str, Dict[Any, float]]:
        """Get WOE maps for all features."""
        return {
            feat: self.get_woe_map(feat)
            for feat in self._features
            if feat in self.binners_
        }

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
