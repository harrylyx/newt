"""
Stepwise regression feature selection.

Provides forward, backward, and bidirectional stepwise selection
based on statistical significance (p-values) or information criteria (AIC/BIC).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from newt.config import MODELING
from newt.utils.decorators import requires_fit


class StepwiseSelector:
    """
    Stepwise regression feature selector.

    Uses hypothesis testing to select optimal features for logistic regression.
    Supports forward selection, backward elimination, and bidirectional stepwise.

    This is typically used after WOE transformation and before final model building.

    Examples
    --------
    >>> selector = StepwiseSelector(direction='both', criterion='aic')
    >>> selector.fit(X_woe, y)
    >>> X_selected = selector.transform(X_woe)
    >>> print(selector.selected_features_)
    """

    def __init__(
        self,
        direction: str = "both",
        criterion: str = "aic",
        p_enter: float = MODELING.DEFAULT_P_ENTER,
        p_remove: float = MODELING.DEFAULT_P_REMOVE,
        max_iter: int = 100,
        fit_intercept: bool = True,
        exclude: Optional[List[str]] = None,
    ):
        """
        Initialize StepwiseSelector.

        Parameters
        ----------
        direction : str
            Selection direction:
            - 'forward': Start with no features, add one at a time
            - 'backward': Start with all features, remove one at a time
            - 'both': Bidirectional stepwise (forward + backward)
            Default 'both'.
        criterion : str
            Selection criterion:
            - 'pvalue': Use p-value for selection
            - 'aic': Use Akaike Information Criterion
            - 'bic': Use Bayesian Information Criterion
            Default 'aic'.
        p_enter : float
            P-value threshold for entering a feature. Default 0.05.
            Used when direction='forward' or 'both'.
        p_remove : float
            P-value threshold for removing a feature. Default 0.10.
            Used when direction='backward' or 'both'.
        max_iter : int
            Maximum iterations. Default 100.
        fit_intercept : bool
            Whether to include intercept. Default True.
        exclude : List[str], optional
            Features to always keep in the model (force include).
        """
        if direction not in ("forward", "backward", "both"):
            raise ValueError("direction must be 'forward', 'backward', or 'both'")
        if criterion not in ("pvalue", "aic", "bic"):
            raise ValueError("criterion must be 'pvalue', 'aic', or 'bic'")

        self.direction = direction
        self.criterion = criterion
        self.p_enter = p_enter
        self.p_remove = p_remove
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.exclude = exclude or []

        # Fitted attributes
        self.selected_features_: List[str] = []
        self.removed_features_: List[str] = []
        self.selection_history_: List[Dict] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StepwiseSelector":
        """
        Fit the stepwise selector.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data (typically WOE transformed).
        y : pd.Series
            Binary target variable (0/1).

        Returns
        -------
        StepwiseSelector
            Fitted instance.
        """
        try:
            import statsmodels.api as sm
        except ImportError:
            raise ImportError(
                "statsmodels is required for StepwiseSelector. "
                "Install it with: pip install statsmodels"
            )

        X = X.copy()
        y = y.copy()

        all_features = X.columns.tolist()

        # Ensure exclude features are valid
        exclude_set = set(self.exclude) & set(all_features)

        if self.direction == "forward":
            selected = self._forward_selection(X, y, all_features, exclude_set, sm)
        elif self.direction == "backward":
            selected = self._backward_elimination(X, y, all_features, exclude_set, sm)
        else:  # both
            selected = self._bidirectional_selection(
                X, y, all_features, exclude_set, sm
            )

        self.selected_features_ = selected
        self.removed_features_ = [f for f in all_features if f not in selected]
        self.is_fitted_ = True

        return self

    def _fit_model(self, X: pd.DataFrame, y: pd.Series, features: List[str], sm):
        """Fit logistic regression model and return result."""
        if not features:
            return None

        X_subset = X[features]
        if self.fit_intercept:
            X_subset = sm.add_constant(X_subset, has_constant="add")

        try:
            model = sm.Logit(y, X_subset)
            result = model.fit(disp=False, maxiter=self.max_iter)
            return result
        except Exception:
            return None

    def _get_criterion_value(self, result, criterion: str) -> float:
        """Get criterion value for model comparison."""
        if result is None:
            return np.inf

        if criterion == "aic":
            return result.aic
        elif criterion == "bic":
            return result.bic
        else:  # pvalue - return max p-value (for backward)
            pvalues = result.pvalues
            if self.fit_intercept and "const" in pvalues.index:
                pvalues = pvalues.drop("const")
            return pvalues.max() if len(pvalues) > 0 else 0.0

    def _forward_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        all_features: List[str],
        exclude_set: set,
        sm,
    ) -> List[str]:
        """Forward selection: start empty, add features one by one."""
        selected = list(exclude_set)
        remaining = [f for f in all_features if f not in selected]

        for iteration in range(self.max_iter):
            best_feature = None
            best_criterion = np.inf if self.criterion != "pvalue" else 1.0
            best_pvalue = 1.0

            for feature in remaining:
                candidate = selected + [feature]
                result = self._fit_model(X, y, candidate, sm)

                if result is None:
                    continue

                if self.criterion == "pvalue":
                    # Get p-value of the new feature
                    pvalue = result.pvalues.get(feature, 1.0)
                    if pvalue < best_pvalue and pvalue < self.p_enter:
                        best_pvalue = pvalue
                        best_feature = feature
                        best_criterion = pvalue
                else:
                    criterion_val = self._get_criterion_value(result, self.criterion)
                    current_result = self._fit_model(X, y, selected, sm)
                    current_criterion = self._get_criterion_value(
                        current_result, self.criterion
                    )

                    # Lower AIC/BIC is better
                    if criterion_val < current_criterion and criterion_val < best_criterion:
                        best_criterion = criterion_val
                        best_feature = feature

            if best_feature is None:
                break

            selected.append(best_feature)
            remaining.remove(best_feature)

            self.selection_history_.append(
                {
                    "iteration": iteration + 1,
                    "action": "add",
                    "feature": best_feature,
                    "criterion": self.criterion,
                    "value": best_criterion,
                }
            )

        return selected

    def _backward_elimination(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        all_features: List[str],
        exclude_set: set,
        sm,
    ) -> List[str]:
        """Backward elimination: start with all, remove features one by one."""
        selected = all_features.copy()

        for iteration in range(self.max_iter):
            result = self._fit_model(X, y, selected, sm)

            if result is None or len(selected) <= len(exclude_set):
                break

            # Find feature to remove (highest p-value or worst criterion impact)
            removable = [f for f in selected if f not in exclude_set]
            if not removable:
                break

            worst_feature = None
            worst_pvalue = 0.0

            if self.criterion == "pvalue":
                pvalues = result.pvalues
                for feature in removable:
                    pvalue = pvalues.get(feature, 0.0)
                    if pvalue > worst_pvalue:
                        worst_pvalue = pvalue
                        worst_feature = feature

                if worst_pvalue <= self.p_remove:
                    break
            else:
                # For AIC/BIC, try removing each feature and find best improvement
                current_criterion = self._get_criterion_value(result, self.criterion)
                best_improvement = 0

                for feature in removable:
                    candidate = [f for f in selected if f != feature]
                    test_result = self._fit_model(X, y, candidate, sm)
                    test_criterion = self._get_criterion_value(
                        test_result, self.criterion
                    )

                    improvement = current_criterion - test_criterion
                    if improvement > best_improvement:
                        best_improvement = improvement
                        worst_feature = feature
                        worst_pvalue = result.pvalues.get(feature, 0.0)

                if best_improvement <= 0:
                    break

            if worst_feature is None:
                break

            selected.remove(worst_feature)

            self.selection_history_.append(
                {
                    "iteration": iteration + 1,
                    "action": "remove",
                    "feature": worst_feature,
                    "criterion": self.criterion,
                    "value": worst_pvalue,
                }
            )

        return selected

    def _bidirectional_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        all_features: List[str],
        exclude_set: set,
        sm,
    ) -> List[str]:
        """Bidirectional stepwise: combine forward and backward."""
        selected = list(exclude_set)
        remaining = [f for f in all_features if f not in selected]

        for iteration in range(self.max_iter):
            changed = False

            # Forward step: try to add a feature
            best_feature = None
            best_criterion = np.inf if self.criterion != "pvalue" else 1.0

            for feature in remaining:
                candidate = selected + [feature]
                result = self._fit_model(X, y, candidate, sm)

                if result is None:
                    continue

                if self.criterion == "pvalue":
                    pvalue = result.pvalues.get(feature, 1.0)
                    if pvalue < best_criterion and pvalue < self.p_enter:
                        best_criterion = pvalue
                        best_feature = feature
                else:
                    criterion_val = self._get_criterion_value(result, self.criterion)
                    current_result = self._fit_model(X, y, selected, sm)
                    current_criterion = self._get_criterion_value(
                        current_result, self.criterion
                    )

                    if criterion_val < current_criterion and criterion_val < best_criterion:
                        best_criterion = criterion_val
                        best_feature = feature

            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
                changed = True

                self.selection_history_.append(
                    {
                        "iteration": iteration + 1,
                        "action": "add",
                        "feature": best_feature,
                        "criterion": self.criterion,
                        "value": best_criterion,
                    }
                )

            # Backward step: try to remove a feature
            if len(selected) > len(exclude_set):
                result = self._fit_model(X, y, selected, sm)

                if result is not None:
                    removable = [f for f in selected if f not in exclude_set]
                    worst_feature = None
                    worst_pvalue = 0.0

                    if self.criterion == "pvalue":
                        pvalues = result.pvalues
                        for feature in removable:
                            pvalue = pvalues.get(feature, 0.0)
                            if pvalue > worst_pvalue and pvalue > self.p_remove:
                                worst_pvalue = pvalue
                                worst_feature = feature
                    else:
                        current_criterion = self._get_criterion_value(
                            result, self.criterion
                        )
                        for feature in removable:
                            candidate = [f for f in selected if f != feature]
                            test_result = self._fit_model(X, y, candidate, sm)
                            test_criterion = self._get_criterion_value(
                                test_result, self.criterion
                            )

                            if test_criterion < current_criterion:
                                pvalue = result.pvalues.get(feature, 0.0)
                                if pvalue > worst_pvalue:
                                    worst_pvalue = pvalue
                                    worst_feature = feature

                    if worst_feature is not None:
                        selected.remove(worst_feature)
                        remaining.append(worst_feature)
                        changed = True

                        self.selection_history_.append(
                            {
                                "iteration": iteration + 1,
                                "action": "remove",
                                "feature": worst_feature,
                                "criterion": self.criterion,
                                "value": worst_pvalue,
                            }
                        )

            if not changed:
                break

        return selected

    @requires_fit()
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
        cols_to_keep = [c for c in self.selected_features_ if c in X.columns]
        return X[cols_to_keep]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    @requires_fit()
    def report(self) -> pd.DataFrame:
        """
        Generate selection report.

        Returns
        -------
        pd.DataFrame
            Selection history with iterations, actions, and criteria values.
        """
        if not self.selection_history_:
            return pd.DataFrame(
                columns=["iteration", "action", "feature", "criterion", "value"]
            )

        return pd.DataFrame(self.selection_history_)

    @requires_fit()
    def summary(self) -> str:
        """
        Get selection summary.

        Returns
        -------
        str
            Summary of stepwise selection results.
        """
        lines = [
            "=" * 50,
            "Stepwise Selection Summary",
            "=" * 50,
            f"Direction: {self.direction}",
            f"Criterion: {self.criterion}",
            f"P-enter: {self.p_enter}, P-remove: {self.p_remove}",
            "-" * 50,
            f"Selected features: {len(self.selected_features_)}",
            f"Removed features: {len(self.removed_features_)}",
            "-" * 50,
            "Selected:",
        ]

        for f in self.selected_features_:
            lines.append(f"  - {f}")

        if self.removed_features_:
            lines.append("-" * 50)
            lines.append("Removed:")
            for f in self.removed_features_:
                lines.append(f"  - {f}")

        lines.append("=" * 50)
        return "\n".join(lines)
