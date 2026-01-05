import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from .base import BaseBinner
from .supervised import ChiMergeBinner, DecisionTreeBinner
from .unsupervised import EqualWidthBinner, EqualFrequencyBinner, KMeansBinner


class Combiner:
    """
    A unified interface for binning multiple features using various algorithms.
    Supported methods: 'chi', 'dt', 'kmean', 'quantile', 'step'.
    """

    def __init__(self):
        self.rules_: Dict[str, List[float]] = {}
        self.method_map = {
            "chi": ChiMergeBinner,
            "dt": DecisionTreeBinner,
            "kmean": KMeansBinner,
            "quantile": EqualFrequencyBinner,
            "step": EqualWidthBinner,
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
        exclude: Optional[List[str]] = None,
    ) -> "Combiner":
        """
        Fit the binning model.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be binned.
        y : str or pd.Series, optional
            Target data. Required for supervised methods ('chi', 'dt').
        method : str
            Binning method. 'chi', 'dt', 'kmean', 'quantile', 'step'.
        n_bins : int
            Number of bins.
        min_samples : int, float
            Minimum samples per leaf (for decision tree).
        empty_separate : bool
            Whether to separate empty values - Not implemented yet (handled by pd.cut usually).
        exclude : List[str]
            Columns to exclude.
        """
        if isinstance(y, str):
            y = X[y]
            if y.name in X.columns:
                 X = X.drop(columns=[y.name])

        if exclude:
            X = X.drop(columns=exclude, errors="ignore")

        # Select numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            binner_cls = self.method_map.get(method)
            if not binner_cls:
                raise ValueError(f"Unknown method ensure: {method}")

            # Instantiate binner
            # Adjust params based on method
            kwargs = {"n_bins": n_bins}
            
            # Specific params
            if method == "dt" and min_samples is not None:
                kwargs["min_samples_leaf"] = min_samples

            binner = binner_cls(**kwargs)
            
            # Fit
            try:
                binner.fit(X[col], y)
                self.binners_[col] = binner
                self.rules_[col] = binner.splits_
            except Exception as e:
                print(f"Failed to bin column {col}: {e}")
                # Keep fitting other columns

        return self

    def transform(
        self, X: pd.DataFrame, labels: bool = False
    ) -> pd.DataFrame:
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
        # But we don't know the method used. It doesn't matter for transform,
        # as long as we have splits.
        # We can default to any Binner, or simple BaseBinner wrapper.
        
        # We'll use EqualWidthBinner as a generic container since it inherits BaseBinner
        # and doesn't enforce strict logic on transform other than splits.
        for col, splits in rules.items():
            binner = EqualWidthBinner()
            binner.set_splits(splits)
            self.binners_[col] = binner

        return self
