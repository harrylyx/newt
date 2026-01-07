"""
Mixin classes for Binner functionality.

Separates Binner responsibilities into focused mixins for better maintainability.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .binning_stats import calculate_bin_stats


class BinnerStatsMixin:
    """
    Mixin providing statistics calculation functionality.

    Requires the following attributes from the main class:
    - _X: pd.DataFrame
    - _y: pd.Series
    - _features: List[str]
    - binners_: Dict[str, BaseBinner]
    - woe_storage: WOEStorage
    - stats_: Dict[str, pd.DataFrame]
    - _missing_label: str
    """

    def _update_all_stats(self):
        """Update statistics for all binned features."""
        if self._X is None or self._y is None:
            return

        for feature in self._features:
            if feature in self.binners_:
                self._calculate_and_store_stats(feature)

    def _calculate_and_store_stats(self, feature: str):
        """Calculate and store statistics for a feature."""
        from newt.features.analysis.woe_calculator import WOEEncoder

        if self._X is None or self._y is None:
            return

        binner = self.binners_[feature]
        col_data = self._X[feature]
        y_data = self._y

        # Transform with missing handling
        valid_mask = col_data.notna()
        binned = pd.Series(index=col_data.index, dtype=object)

        if valid_mask.any():
            valid_binned = binner.transform(col_data[valid_mask])
            binned[valid_mask] = valid_binned.astype(str)

        binned[~valid_mask] = self._missing_label

        # Calculate WOE using encoder
        woe_encoder = WOEEncoder()
        woe_encoder.fit(binned, y_data)

        # Store WOE encoder
        self.woe_storage.store(feature, woe_encoder)

        # Calculate full stats
        self.stats_[feature] = calculate_bin_stats(
            X=col_data,
            y=y_data,
            binned=binned,
            splits=binner.splits_,
            woe_encoder=woe_encoder,
        )


class BinnerIOMixin:
    """
    Mixin providing import/export functionality.

    Requires the following attributes from the main class:
    - rules_: Dict[str, List[float]]
    - binners_: Dict[str, BaseBinner]
    - _features: List[str]
    - _X: Optional[pd.DataFrame]
    - _y: Optional[pd.Series]
    """

    def get_splits(self, feature: str) -> List[float]:
        """
        Get split points for a feature.

        Parameters
        ----------
        feature : str
            Feature name.

        Returns
        -------
        List[float]
            Split points.
        """
        if feature not in self.rules_:
            raise KeyError(f"Feature '{feature}' not found.")
        return self.rules_[feature].copy()

    def set_splits(self, feature: str, splits: List[float]):
        """
        Set split points for a feature.

        Replaces existing splits with new values.

        Parameters
        ----------
        feature : str
            Feature name.
        splits : List[float]
            New split points.

        Examples
        --------
        >>> binner.set_splits('age', [25, 35, 45, 55])
        """
        from .unsupervised import EqualWidthBinner

        splits = sorted(splits)
        self.rules_[feature] = splits

        if feature in self.binners_:
            self.binners_[feature].set_splits(splits)
        else:
            binner = EqualWidthBinner()
            binner.set_splits(splits)
            self.binners_[feature] = binner
            if feature not in self._features:
                self._features.append(feature)

        # Update statistics
        if self._X is not None and self._y is not None:
            self._calculate_and_store_stats(feature)

    def print_splits(self, feature: Optional[str] = None):
        """
        Print split points for feature(s).

        Parameters
        ----------
        feature : str, optional
            Feature name. If None, prints all features.
        """
        if feature:
            if feature not in self.rules_:
                print(f"Feature '{feature}' not found.")
                return
            print(f"{feature}: {self.rules_[feature]}")
        else:
            for f, splits in self.rules_.items():
                print(f"{f}: {splits}")

    def export(self) -> Dict[str, List[float]]:
        """Export binning rules."""
        return self.rules_.copy()

    def load(self, rules: Dict[str, List[float]]) -> "BinnerIOMixin":
        """
        Load binning rules manually.

        Parameters
        ----------
        rules : Dict[str, List[float]]
            Binning rules mapping feature to split points.

        Returns
        -------
        Binner
            Self.
        """
        from .unsupervised import EqualWidthBinner

        self.rules_ = rules.copy()
        self.binners_ = {}

        for col, splits in rules.items():
            binner = EqualWidthBinner()
            binner.set_splits(splits)
            self.binners_[col] = binner

        return self


class BinnerWOEMixin:
    """
    Mixin providing WOE/IV access functionality.

    Requires the following attributes from the main class:
    - woe_storage: WOEStorage
    """

    def get_iv(self, feature: Optional[str] = None) -> Union[float, pd.DataFrame]:
        """
        Get IV value(s).

        Parameters
        ----------
        feature : str, optional
            Feature name. If None, returns all IVs.

        Returns
        -------
        float or pd.DataFrame
            IV value or DataFrame with all IVs.
        """
        if feature:
            return self.woe_storage.get_iv(feature)
        return self.woe_storage.get_all_iv()

    def get_woe_map(self, feature: str) -> Dict[Any, float]:
        """
        Get WOE mapping for a feature.

        Parameters
        ----------
        feature : str
            Feature name.

        Returns
        -------
        Dict
            Mapping from bin label to WOE value.
        """
        return self.woe_storage.get_woe_map(feature)
