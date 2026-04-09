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
from .supervised import (
    ChiMergeBinner,
    DecisionTreeBinner,
    OptBinningBinner,
    _load_rust_engine,
)
from .unsupervised import EqualFrequencyBinner, EqualWidthBinner, KMeansBinner
from .woe_storage import WOEStorage


class BinningResult:
    """Proxy object for accessing binning results of a single feature.

    This class provides a convenient way to access statistics, plots, and WOE
    mappings for a specific feature within a fitted Binner instance.

    Attributes:
        stats (pd.DataFrame): Binning statistics including bin ranges, counts,
            bad rates, WOE, and IV contribution.
    """

    def __init__(self, binner: "Binner", feature: str):
        """Initialize BinningResult.

        Args:
            binner: The parent Binner instance.
            feature: The name of the feature to proxy.
        """
        self._binner = binner
        self._feature = feature

        # Ensure stats are calculated
        if feature not in binner.stats_:
            binner._calculate_and_store_stats(feature)

        self.stats = binner.stats_[feature]

    def plot(
        self,
        x: str = "bin",
        y: Optional[Union[str, List[str]]] = None,
        secondary_y: Optional[Union[str, List[str]]] = "bad_rate",
        **kwargs,
    ):
        """Plot binning results for this feature.

        Generates a bar-and-line chart showing the distribution of samples and the
        bad rate (or other metrics) across bins.

        Args:
            x: Column for x-axis. Default 'bin'.
            y: Column(s) for primary y-axis (bar). Default ['bad_prop'].
            secondary_y: Column(s) for secondary y-axis (line). Default 'bad_rate'.
            **kwargs: Arguments passed to plot_binning_result.

        Returns:
            matplotlib.figure.Figure: The generated figure.

        Examples:
            >>> binner['age'].plot(title="Age Binning Distribution")
        """
        if y is None:
            y = ["bad_prop"]

        from newt.results import BinningPlotData
        from newt.visualization.binning_viz import plot_binning_result

        return plot_binning_result(
            binner=BinningPlotData.from_binner(self._binner, self._feature),
            woe_encoder=self._binner.woe_storage.get(self._feature),
            x_col=x,
            y_col=y,
            secondary_y_col=secondary_y,
            **kwargs,
        )

    def woe_map(self) -> Dict[Any, float]:
        """Get the WOE mapping for this feature.

        Returns:
            Dict[Any, float]: A dictionary mapping bin labels to WOE values.
        """
        return self._binner.get_woe_map(self._feature)


class Binner(BinnerStatsMixin, BinnerIOMixin, BinnerWOEMixin):
    """Unified interface for multi-feature binning using various algorithms.

    The Binner class manages the discretization of multiple features, handles
    missing values automatically, and stores WOE encoders for downstream modeling.
    It supports both supervised (ChiMerge, Decision Tree, Optimal) and
    unsupervised (K-Means, Equal Width, Equal Frequency) algorithms.

    Supported methods:
        - 'chi': ChiMerge (Default)
        - 'dt': Decision Tree
        - 'opt': Optimal Binning
        - 'kmean': K-Means
        - 'quantile': Equal Frequency
        - 'step': Equal Width

    Examples:
        >>> from newt.features.binning import Binner
        >>> binner = Binner()
        >>> binner.fit(X_train, y_train, method='chi', n_bins=5, monotonic=True)
        >>> # Access results via item access
        >>> print(binner['age'].stats)
        >>> binner['age'].plot()
        >>> # Transform new data
        >>> X_binned = binner.transform(X_test)
    """

    def __init__(self):
        """Initialize the Binner."""
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
        """Get WOE encoders dictionary (for backward compatibility).

        Returns:
            Dict[str, Any]: Mapping of feature names to WOEEncoder objects.
        """
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
        show_progress: bool = True,
        **kwargs,
    ) -> "Binner":
        """Fit the binning model to multiple features.

        Initializes and fits specific binning algorithms for each selected feature,
        calculates binning statistics, and stores WOE mappings.

        Args:
            X: Data to be binned.
            y: Target data or target column name. Required for supervised methods.
            method: Binning algorithm name ('chi', 'dt', 'opt', 'kmean', etc.).
            n_bins: Target number of bins.
            min_samples: Minimum samples per leaf (relevant for 'dt').
            cols: List of columns to bin. If None, all numeric columns are selected.
            monotonic: Enforce monotonic bad rate trend.
                - True/'auto': Enforce auto-detected trend.
                - 'ascending'/'descending': Enforce specific trend.
            show_progress: Whether to show a progress bar.
            **kwargs: Additional parameters passed to the underlying binner.

        Returns:
            Binner: The fitted Binner instance.

        Examples:
            >>> binner.fit(df, target='default', method='chi', monotonic=True)
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

        # tqdm for progress tracking
        try:
            from tqdm.auto import tqdm

            has_tqdm = True
        except ImportError:
            has_tqdm = False

        # Determine if we can use parallel Rust
        rust_module = _load_rust_engine()
        use_parallel = (
            method == "chi"
            and rust_module
            and hasattr(rust_module, "calculate_chi_merge_numpy")
        )

        if use_parallel:
            from concurrent.futures import ThreadPoolExecutor

            from scipy import stats

            threshold = float(stats.chi2.ppf(1 - (kwargs.get("alpha", 0.05)), 1))
            n_threads = min(len(numeric_cols), 8)  # Limit threads to 8 or num cols

            def fit_single_col(col):
                binner_cls = self.method_map.get(method)
                kwargs_binner = {"n_bins": n_bins, "monotonic": monotonic}
                kwargs_binner.update(kwargs)
                binner = binner_cls(**kwargs_binner)

                col_data = X[col]
                valid_mask = col_data.notna()
                if valid_mask.sum() == 0:
                    return col, None, None

                # Directly call Rust if it's ChiMerge
                try:
                    splits = rust_module.calculate_chi_merge_numpy(
                        col_data[valid_mask].astype(np.float64).to_numpy(),
                        y_series[valid_mask].astype(np.int64).to_numpy(),
                        n_bins,
                        threshold,
                    )
                    splits = sorted(list(set(splits)))

                    # Handle monotonicity if requested
                    if binner.monotonic and y_series is not None:
                        splits = binner._adjust_monotonicity(
                            col_data[valid_mask], y_series[valid_mask], splits
                        )

                    binner.splits_ = splits
                    binner.is_fitted_ = True
                except Exception:
                    # Fallback to standard fit
                    binner.fit(col_data[valid_mask], y_series[valid_mask])

                return col, binner, binner.splits_

            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                if has_tqdm and show_progress:
                    results = list(
                        tqdm(
                            executor.map(fit_single_col, numeric_cols),
                            total=len(numeric_cols),
                            desc="Binning features (Parallel)",
                        )
                    )
                else:
                    results = list(executor.map(fit_single_col, numeric_cols))

            for col, binner, splits in results:
                if binner is not None:
                    self.binners_[col] = binner
                    self.rules_[col] = splits

        else:
            # Sequential fallback
            pbar = (
                tqdm(numeric_cols, desc="Binning features", disable=not show_progress)
                if has_tqdm
                else numeric_cols
            )
            for col in pbar:
                binner_cls = self.method_map.get(method)
                if binner_cls is None:
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

                binner.fit(col_data[valid_mask], y_series[valid_mask])
                self.binners_[col] = binner
                self.rules_[col] = binner.splits_

        # Calculate and store statistics
        self._update_all_stats()

        return self

    def transform(
        self,
        X: pd.DataFrame,
        labels: bool = False,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Discretizes values based on splits discovered during fitting. Missing
        values are automatically assigned to a 'Missing' bin.

        Args:
            X: Data to transform.
            labels: If True, return bin intervals (str).
                If False, return bin indices (int).
            show_progress: Whether to show a progress bar.

        Returns:
            pd.DataFrame: Binned data with original columns replaced by
                bin codes/labels.
        """
        X_new = X.copy()

        # tqdm for progress tracking
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(
                self.binners_.items(),
                desc="Transforming features",
                disable=not show_progress,
            )
        except ImportError:
            pbar = self.binners_.items()

        for col, binner in pbar:
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
        """Convenience method to bin and WOE-transform data in one pass.

        Args:
            X: Raw feature DataFrame.

        Returns:
            pd.DataFrame: WOE-encoded DataFrame.

        Examples:
            >>> X_woe = binner.woe_transform(X_raw)
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
            raise KeyError(f"Feature '{feature}' is missing from binner.")

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
            from IPython.display import display

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
