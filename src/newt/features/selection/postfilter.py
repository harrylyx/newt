"""
PostFilter module for variable fine-screening.

Provides filtering based on PSI and VIF after WOE transformation.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from newt.metrics.psi import calculate_psi
from newt.metrics.vif import calculate_vif


class PostFilter:
    """
    Variable post-filter based on PSI and VIF.

    This filter performs variable screening after WOE transformation, using:
    - Population Stability Index (PSI) threshold
    - Variance Inflation Factor (VIF) threshold

    Examples
    --------
    >>> postfilter = PostFilter(psi_threshold=0.25, vif_threshold=10)
    >>> postfilter.fit(X_train_woe, X_test_woe)
    >>> X_filtered = postfilter.transform(X_train_woe)
    """

    def __init__(
        self,
        psi_threshold: float = 0.25,
        vif_threshold: float = 10.0,
        psi_buckets: int = 10,
        remove_high_vif_iteratively: bool = True,
    ):
        """
        Initialize PostFilter.

        Parameters
        ----------
        psi_threshold : float
            Maximum PSI value to keep a variable. Default 0.25.
            PSI < 0.1: no significant change
            PSI 0.1-0.25: moderate change
            PSI > 0.25: significant change
        vif_threshold : float
            Maximum VIF value to keep a variable. Default 10.0.
            VIF > 5: moderate multicollinearity
            VIF > 10: high multicollinearity
        psi_buckets : int
            Number of buckets for PSI calculation. Default 10.
        remove_high_vif_iteratively : bool
            If True, iteratively remove features with highest VIF.
            If False, remove all features with VIF > threshold at once.
        """
        self.psi_threshold = psi_threshold
        self.vif_threshold = vif_threshold
        self.psi_buckets = psi_buckets
        self.remove_high_vif_iteratively = remove_high_vif_iteratively

        # Results
        self.selected_features_: List[str] = []
        self.removed_features_: Dict[str, str] = {}
        self.psi_dict_: Dict[str, float] = {}
        self.vif_df_: pd.DataFrame = pd.DataFrame()
        self.is_fitted_: bool = False

    def _remove_high_vif_iteratively(
        self,
        X: pd.DataFrame,
        features: List[str],
        threshold: float,
        max_iterations: int = 100,
    ) -> Dict:
        """
        Iteratively remove features with highest VIF until all VIF < threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.
        features : List[str]
            Features to consider.
        threshold : float
            Maximum allowed VIF.
        max_iterations : int
            Maximum iterations to prevent infinite loops.

        Returns
        -------
        Dict
            Dictionary with selected_features, removed_features, final_vif.
        """
        current_features = features.copy()
        removed = []

        for _ in range(max_iterations):
            if len(current_features) < 2:
                break

            vif_df = calculate_vif(X[current_features])

            # Find max VIF
            max_vif_row = vif_df.loc[vif_df["vif"].idxmax()]
            max_vif = max_vif_row["vif"]
            max_feature = max_vif_row["feature"]

            if max_vif <= threshold or np.isnan(max_vif) or np.isinf(max_vif):
                break

            # Remove feature with highest VIF
            removed.append(max_feature)
            current_features.remove(max_feature)

        # Calculate final VIF
        final_vif = (
            calculate_vif(X[current_features]) if current_features else pd.DataFrame()
        )

        return {
            "selected_features": current_features,
            "removed_features": removed,
            "final_vif": final_vif,
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
    ) -> "PostFilter":
        """
        Fit the post-filter.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data (WOE transformed).
        X_test : pd.DataFrame, optional
            Test/validation data for PSI calculation.
            If None, PSI filtering is skipped.

        Returns
        -------
        PostFilter
            Fitted instance.
        """
        X_train = X_train.copy()
        if X_test is not None:
            X_test = X_test.copy()

        # Get numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        # Reset results
        self.psi_dict_ = {}
        self.removed_features_ = {}
        self.selected_features_ = []
        candidates = numeric_cols.copy()

        # Step 1: PSI filtering (if X_test provided)
        if X_test is not None:
            psi_candidates = []
            for col in candidates:
                if col not in X_test.columns:
                    continue

                psi = calculate_psi(
                    X_train[col].values,
                    X_test[col].values,
                    buckets=self.psi_buckets,
                )
                self.psi_dict_[col] = psi

                if np.isnan(psi):
                    psi_candidates.append(col)
                elif psi > self.psi_threshold:
                    self.removed_features_[col] = f"psi={psi:.4f}"
                else:
                    psi_candidates.append(col)

            candidates = psi_candidates

        # Step 2: VIF filtering
        if len(candidates) >= 2:
            if self.remove_high_vif_iteratively:
                vif_result = self._remove_high_vif_iteratively(
                    X_train,
                    candidates,
                    threshold=self.vif_threshold,
                )
                self.selected_features_ = vif_result["selected_features"]
                self.vif_df_ = vif_result["final_vif"]

                for removed_feat in vif_result["removed_features"]:
                    self.removed_features_[removed_feat] = "high_vif"
            else:
                vif_df = calculate_vif(X_train[candidates])
                self.vif_df_ = vif_df

                for _, row in vif_df.iterrows():
                    feat = row["feature"]
                    vif = row["vif"]

                    if vif > self.vif_threshold:
                        self.removed_features_[feat] = f"vif={vif:.2f}"
                    else:
                        self.selected_features_.append(feat)
        else:
            self.selected_features_ = candidates
            if candidates:
                self.vif_df_ = calculate_vif(X_train[candidates])

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filter columns based on fitted selection.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Filtered data with only selected features.
        """
        if not self.is_fitted_:
            raise ValueError("PostFilter is not fitted. Call fit() first.")

        cols_to_keep = [c for c in self.selected_features_ if c in X.columns]
        return X[cols_to_keep]

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X_train, X_test)
        return self.transform(X_train)

    def report(self) -> Dict[str, pd.DataFrame]:
        """
        Generate filtering reports.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with:
            - 'summary': Overall filtering summary
            - 'psi': PSI values for each feature
            - 'vif': VIF values for selected features
        """
        if not self.is_fitted_:
            raise ValueError("PostFilter is not fitted. Call fit() first.")

        # Summary report
        all_features = set(self.psi_dict_.keys())
        if len(self.vif_df_) > 0:
            all_features = all_features | set(self.vif_df_["feature"].tolist())
        all_features = (
            all_features
            | set(self.removed_features_.keys())
            | set(self.selected_features_)
        )

        summary_records = []
        for feat in all_features:
            psi = self.psi_dict_.get(feat, np.nan)
            vif_row = (
                self.vif_df_[self.vif_df_["feature"] == feat]
                if len(self.vif_df_) > 0
                else pd.DataFrame()
            )
            vif = vif_row["vif"].values[0] if len(vif_row) > 0 else np.nan

            if feat in self.selected_features_:
                status = "selected"
                reason = ""
            else:
                status = "removed"
                reason = self.removed_features_.get(feat, "")

            summary_records.append(
                {
                    "feature": feat,
                    "psi": psi,
                    "vif": vif,
                    "status": status,
                    "reason": reason,
                }
            )

        summary_df = pd.DataFrame(summary_records)
        summary_df = summary_df.sort_values(["status", "psi"]).reset_index(drop=True)

        # PSI report
        psi_df = pd.DataFrame(
            [{"feature": k, "psi": v} for k, v in self.psi_dict_.items()]
        )
        if len(psi_df) > 0:
            psi_df = psi_df.sort_values("psi", ascending=False).reset_index(drop=True)

        return {
            "summary": summary_df,
            "psi": psi_df,
            "vif": self.vif_df_,
        }
