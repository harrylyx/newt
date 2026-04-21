"""
Unified binning interface.

Provides a single entry point for binning features using various algorithms.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from newt.config import BINNING

from .base import BaseBinner
from .binner_mixins import BinnerIOMixin, BinnerStatsMixin
from .supervised import (
    ChiMergeBinner,
    DecisionTreeBinner,
    OptBinningBinner,
    _load_rust_engine,
    _resolve_chi_min_samples_count,
    _resolve_monotonic_mode,
    _validate_chi_target,
)
from .unsupervised import EqualFrequencyBinner, EqualWidthBinner, KMeansBinner


def _resolve_min_samples_count(
    min_samples: Union[int, float],
    sample_count: int,
    context: str,
) -> int:
    """Resolve min_samples into an absolute threshold for a feature."""
    if sample_count <= 0:
        return 1

    if isinstance(min_samples, bool):
        raise TypeError(f"{context}: min_samples must be int or float, got bool.")

    if isinstance(min_samples, (int, np.integer)):
        value = int(min_samples)
        if value <= 0:
            raise ValueError(f"{context}: min_samples must be > 0, got {value}.")
        if value > sample_count:
            raise ValueError(
                f"{context}: min_samples={value} exceeds non-missing sample count "
                f"{sample_count}."
            )
        return value

    if isinstance(min_samples, (float, np.floating)):
        value = float(min_samples)
        if not np.isfinite(value):
            raise ValueError(f"{context}: min_samples must be finite, got {value}.")
        if value <= 0.0:
            raise ValueError(f"{context}: min_samples must be > 0, got {value}.")
        if value > 1.0:
            raise ValueError(
                f"{context}: float min_samples must be in (0, 1], got {value}."
            )
        return max(1, int(np.ceil(value * sample_count)))

    raise TypeError(
        f"{context}: min_samples must be int or float, "
        f"got {type(min_samples).__name__}."
    )


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

        from newt.features.analysis.woe_calculator import WOEEncoder
        from newt.results import BinningPlotData
        from newt.visualization.binning_viz import plot_binning_result

        # Create a temporary encoder for plot if it exists in native storage
        woe_encoder = None
        if self._feature in self._binner.woe_maps_:
            woe_encoder = WOEEncoder()
            woe_encoder.woe_map_ = self._binner.woe_maps_[self._feature]
            woe_encoder.iv_ = self._binner.ivs_.get(self._feature, 0.0)
            woe_encoder.is_fitted_ = True

        return plot_binning_result(
            binner=BinningPlotData.from_binner(self._binner, self._feature),
            woe_encoder=woe_encoder,
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


class Binner(BinnerStatsMixin, BinnerIOMixin):
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
        self.woe_maps_: Dict[str, Dict[Any, float]] = {}
        self.ivs_: Dict[str, float] = {}
        self.stats_: Dict[str, pd.DataFrame] = {}
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
        self._features: List[str] = []
        self._missing_label = "Missing"

    @staticmethod
    def _count_bins_from_splits(values: pd.Series, splits: List[float]) -> List[int]:
        """Count samples per bin under ``pd.cut(..., right=True)`` semantics."""
        if values.empty:
            return [0]
        if not splits:
            return [int(values.shape[0])]

        split_array = np.asarray(splits, dtype=np.float64)
        value_array = values.to_numpy(dtype=np.float64, copy=False)
        bin_index = np.searchsorted(split_array, value_array, side="right")
        counts = np.bincount(bin_index, minlength=len(split_array) + 1)
        return counts.astype(int).tolist()

    @staticmethod
    def _select_split_to_merge(counts: List[int], small_bin_index: int) -> int:
        """Select one split index to remove for a small bin."""
        if len(counts) <= 1:
            raise ValueError("At least two bins are required to merge.")

        last_bin_index = len(counts) - 1
        if small_bin_index <= 0:
            return 0
        if small_bin_index >= last_bin_index:
            return last_bin_index - 1

        left_count = counts[small_bin_index - 1]
        right_count = counts[small_bin_index + 1]
        if left_count <= right_count:
            return small_bin_index - 1
        return small_bin_index

    def _converge_feature_splits(
        self,
        binner: BaseBinner,
        col_data: pd.Series,
        y_series: Optional[pd.Series],
        min_sample_count: Optional[int],
    ) -> List[float]:
        """Converge feature splits under min-sample and monotonic constraints."""
        current_splits = sorted(list(set(getattr(binner, "splits_", []))))
        valid_mask = col_data.notna()
        X_valid = col_data[valid_mask]
        if X_valid.empty:
            return current_splits

        y_valid = y_series[valid_mask] if y_series is not None else None

        while True:
            if binner.monotonic and y_valid is not None and current_splits:
                current_splits = sorted(
                    list(
                        set(
                            binner._adjust_monotonicity(
                                X_valid, y_valid, current_splits
                            )
                        )
                    )
                )

            if min_sample_count is None:
                break

            counts = self._count_bins_from_splits(X_valid, current_splits)
            small_bin_index = next(
                (idx for idx, count in enumerate(counts) if count < min_sample_count),
                None,
            )
            if small_bin_index is None or not current_splits:
                break

            split_index = self._select_split_to_merge(counts, small_bin_index)
            current_splits.pop(split_index)

        return current_splits

    def _store_feature_binner(
        self,
        feature: str,
        binner: BaseBinner,
        col_data: pd.Series,
        y_series: Optional[pd.Series],
        min_sample_count: Optional[int],
    ) -> None:
        """Finalize and store one fitted feature binner."""
        final_splits = self._converge_feature_splits(
            binner=binner,
            col_data=col_data,
            y_series=y_series,
            min_sample_count=min_sample_count,
        )
        binner.splits_ = final_splits
        binner.is_fitted_ = True
        self.binners_[feature] = binner
        self.rules_[feature] = final_splits

    @property
    def woe_encoders_(self) -> Dict[str, Any]:
        """Get WOE encoders dictionary (for backward compatibility).

        Returns:
            Dict[str, Any]: Mapping of feature names to WOEEncoder objects.
        """
        from newt.features.analysis.woe_calculator import WOEEncoder

        encoders = {}
        for feature, woe_map in self.woe_maps_.items():
            encoder = WOEEncoder()
            encoder.woe_map_ = woe_map
            encoder.iv_ = self.ivs_.get(feature, 0.0)
            encoder.is_fitted_ = True
            encoders[feature] = encoder
        return encoders

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
            min_samples: Minimum samples threshold.
                - For 'dt': minimum samples per leaf.
                - For 'chi': float in (0, 1] means minimum bin proportion,
                  int means minimum absolute samples per bin.
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

        if method == "chi":
            if y_series is None:
                raise ValueError("ChiMerge requires target 'y'.")
            if not isinstance(y_series, pd.Series):
                y_series = pd.Series(y_series, index=X.index)
            y_series = _validate_chi_target(
                y_series,
                context="Binner.fit(method='chi')",
            )

        # Reset state for a fresh fit.
        self.rules_ = {}
        self.binners_ = {}
        self.woe_maps_ = {}
        self.ivs_ = {}
        self.stats_ = {}

        # Store references for later use
        self._X = X.copy()
        self._y = y_series.copy() if y_series is not None else None

        # Determine columns to bin
        if cols:
            numeric_cols = [c for c in cols if c in X.columns]
        else:
            numeric_cols = list(X.select_dtypes(include=[np.number]).columns)

        feature_min_samples: Dict[str, Optional[int]] = {
            col: None for col in numeric_cols
        }
        if min_samples is not None:
            for col in numeric_cols:
                valid_count = int(X[col].notna().sum())
                if valid_count <= 0:
                    continue
                feature_min_samples[col] = _resolve_min_samples_count(
                    min_samples=min_samples,
                    sample_count=valid_count,
                    context=f"Binner.fit(feature='{col}')",
                )

        self._features = numeric_cols

        # tqdm for progress tracking
        try:
            from tqdm.auto import tqdm

            has_tqdm = True
        except ImportError:
            has_tqdm = False

        # Determine if we can use batch Rust ChiMerge
        rust_module = _load_rust_engine()
        use_batch_rust = (
            method == "chi"
            and y_series is not None
            and rust_module
            and hasattr(rust_module, "calculate_batch_chi_merge_numpy")
        )

        if use_batch_rust:
            from scipy import stats

            threshold = float(stats.chi2.ppf(1 - (kwargs.get("alpha", 0.05)), 1))
            binner_cls = self.method_map.get(method)
            if binner_cls is None:
                raise ValueError(f"Unknown method: {method}")

            kwargs_binner = {"n_bins": n_bins, "monotonic": monotonic}
            if min_samples is not None:
                kwargs_binner["min_samples"] = min_samples
            kwargs_binner.update(kwargs)

            pbar = (
                tqdm(
                    total=len(numeric_cols),
                    desc="Binning features (Rust Batch)",
                    disable=not show_progress,
                )
                if has_tqdm
                else None
            )

            # Group columns by missing-mask so each batch call can share the same y.
            grouped_cols: Dict[bytes, Dict[str, Any]] = {}
            feature_meta: Dict[str, Dict[str, Any]] = {}
            for col in numeric_cols:
                binner = binner_cls(**kwargs_binner)
                col_data = X[col]
                valid_mask = col_data.notna()
                if valid_mask.sum() == 0:
                    if pbar is not None:
                        pbar.update(1)
                    continue

                feature_meta[col] = {
                    "binner": binner,
                    "col_data": col_data,
                    "valid_mask": valid_mask,
                }

                mask_key = valid_mask.to_numpy(dtype=np.bool_, copy=False).tobytes()
                if mask_key not in grouped_cols:
                    grouped_cols[mask_key] = {"mask": valid_mask, "cols": []}
                grouped_cols[mask_key]["cols"].append(col)

            def _fit_single_column_fallback(col: str):
                meta = feature_meta[col]
                binner = meta["binner"]
                col_data = meta["col_data"]
                valid_mask = meta["valid_mask"]
                binner.fit(col_data[valid_mask], y_series[valid_mask])
                self._store_feature_binner(
                    feature=col,
                    binner=binner,
                    col_data=col_data,
                    y_series=y_series,
                    min_sample_count=feature_min_samples.get(col),
                )
                if pbar is not None:
                    pbar.update(1)

            for group in grouped_cols.values():
                valid_mask = group["mask"]
                cols_in_group = group["cols"]
                y_arr = y_series[valid_mask].astype(np.int64).to_numpy()
                feature_arrays = [
                    X[col][valid_mask].astype(np.float64).to_numpy()
                    for col in cols_in_group
                ]
                min_sample_count = _resolve_chi_min_samples_count(
                    kwargs_binner.get("min_samples", 0.05),
                    len(y_arr),
                    context="Binner.fit(method='chi')",
                )

                try:
                    batch_splits = rust_module.calculate_batch_chi_merge_numpy(
                        feature_arrays,
                        y_arr,
                        n_bins,
                        threshold,
                        min_sample_count,
                    )
                except Exception:
                    for col in cols_in_group:
                        _fit_single_column_fallback(col)
                    continue

                if len(batch_splits) != len(cols_in_group):
                    for col in cols_in_group:
                        _fit_single_column_fallback(col)
                    continue

                split_lists = [sorted(list(set(splits))) for splits in batch_splits]
                adjusted_split_lists = split_lists
                monotonic_success = [False] * len(cols_in_group)

                if monotonic:
                    if hasattr(rust_module, "adjust_batch_chi_merge_monotonic_numpy"):
                        try:
                            native_result = (
                                rust_module.adjust_batch_chi_merge_monotonic_numpy(
                                    feature_arrays,
                                    y_arr,
                                    split_lists,
                                    _resolve_monotonic_mode(monotonic),
                                )
                            )
                            if (
                                isinstance(native_result, tuple)
                                and len(native_result) == 2
                            ):
                                candidate_splits, success_flags = native_result
                                if len(candidate_splits) == len(cols_in_group) and len(
                                    success_flags
                                ) == len(cols_in_group):
                                    adjusted_split_lists = [
                                        sorted(list(set(splits)))
                                        for splits in candidate_splits
                                    ]
                                    monotonic_success = [
                                        bool(success) for success in success_flags
                                    ]
                            elif len(native_result) == len(cols_in_group):
                                adjusted_split_lists = [
                                    sorted(list(set(splits)))
                                    for splits in native_result
                                ]
                                monotonic_success = [True] * len(cols_in_group)
                        except Exception:
                            monotonic_success = [False] * len(cols_in_group)
                else:
                    monotonic_success = [True] * len(cols_in_group)

                for split_idx, col in enumerate(cols_in_group):
                    meta = feature_meta[col]
                    binner = meta["binner"]
                    col_data = meta["col_data"]
                    valid_mask = meta["valid_mask"]

                    try:
                        split_list = split_lists[split_idx]
                        if binner.monotonic:
                            if monotonic_success[split_idx]:
                                split_list = adjusted_split_lists[split_idx]
                            else:
                                split_list = BaseBinner._adjust_monotonicity(
                                    binner,
                                    col_data[valid_mask],
                                    y_series[valid_mask],
                                    split_list,
                                )

                        binner.splits_ = split_list
                        self._store_feature_binner(
                            feature=col,
                            binner=binner,
                            col_data=col_data,
                            y_series=y_series,
                            min_sample_count=feature_min_samples.get(col),
                        )
                    except Exception:
                        _fit_single_column_fallback(col)
                        continue

                    if pbar is not None:
                        pbar.update(1)

            if pbar is not None:
                pbar.close()

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
                if method == "chi" and min_samples is not None:
                    kwargs_binner["min_samples"] = min_samples

                kwargs_binner.update(kwargs)

                binner = binner_cls(**kwargs_binner)

                # For fitting, drop missing values
                col_data = X[col]
                valid_mask = col_data.notna()

                if valid_mask.sum() == 0:
                    continue

                binner.fit(col_data[valid_mask], y_series[valid_mask])
                self._store_feature_binner(
                    feature=col,
                    binner=binner,
                    col_data=col_data,
                    y_series=y_series,
                    min_sample_count=feature_min_samples.get(col),
                )

        # Calculate and store statistics
        self.fit_woe(X, y_series, show_progress=show_progress)

        return self

    def fit_woe(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, str],
        show_progress: bool = True,
    ) -> "Binner":
        """Calculate and update WOE mappings for all features.

        Applicable when rules are loaded or manually set. This method updates
        WOE and IV statistics without changing existing split points.

        Args:
            X: Input DataFrame.
            y: Target data or target column name.
            show_progress: Whether to show a progress bar.

        Returns:
            Binner: Self.
        """
        y_series = y
        if isinstance(y, str):
            y_series = X[y]

        self._X = X.copy()
        self._y = y_series.copy() if y_series is not None else None

        if self._y is None:
            return self

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
        from newt.features.analysis.woe_calculator import WOEEncoder

        target_features = [col for col in self.binners_.keys() if col in X_new.columns]
        missing_woe = [col for col in target_features if col not in self.woe_maps_]
        if missing_woe:
            missing = ", ".join(missing_woe)
            raise ValueError(
                f"WOE mappings are missing for feature(s): {missing}. "
                "Call fit_woe() before woe_transform()."
            )

        for col in self.binners_.keys():
            if col not in X_new.columns:
                continue

            woe_map = self.woe_maps_[col]
            iv = self.ivs_.get(col, 0.0)

            # Create temporary encoder for transformation
            encoder = WOEEncoder()
            encoder.woe_map_ = woe_map
            encoder.iv_ = iv
            encoder.is_fitted_ = True

            # First bin the data
            col_data = X[col]
            valid_mask = col_data.notna()
            binned = pd.Series(index=col_data.index, dtype=object)

            if valid_mask.any():
                valid_binned = self.binners_[col].transform(col_data[valid_mask])
                binned[valid_mask] = valid_binned.astype(str)

            binned[~valid_mask] = self._missing_label

            # Apply WOE transformation
            X_new[col] = encoder.transform(binned)

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
