"""
Logistic Regression model wrapper using statsmodels.

Provides a scikit-learn-like interface for statsmodels Logit.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class LogisticModel:
    """
    Logistic Regression model wrapper using statsmodels.

    Provides a familiar fit/predict interface while leveraging statsmodels
    for detailed statistical output (p-values, confidence intervals, etc.).

    Examples
    --------
    >>> model = LogisticModel()
    >>> model.fit(X_woe, y)
    >>> print(model.summary())
    >>> predictions = model.predict_proba(X_woe)
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        method: str = "bfgs",
        maxiter: int = 100,
        regularization: Optional[str] = None,
        alpha: float = 0.0,
        **kwargs,
    ):
        """
        Initialize LogisticModel.

        Parameters
        ----------
        fit_intercept : bool
            Whether to fit an intercept term. Default True.
        method : str
            Optimization method for statsmodels. Default 'bfgs'.
            Options: 'newton', 'bfgs', 'lbfgs', 'powell', 'cg', 'ncg'.
        maxiter : int
            Maximum iterations for optimization. Default 100.
        regularization : str, optional
            Regularization type: 'l1' or 'l2'. Default None (no regularization).
        alpha : float
            Regularization strength. Default 0.0.
        **kwargs
            Additional arguments passed to statsmodels fit method.
        """
        self.fit_intercept = fit_intercept
        self.method = method
        self.maxiter = maxiter
        self.regularization = regularization
        self.alpha = alpha
        self.extra_kwargs = kwargs

        # Fitted attributes
        self.model_ = None
        self.result_ = None
        self.feature_names_: list = []
        self.coefficients_: pd.DataFrame = pd.DataFrame()
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LogisticModel":
        """
        Fit the logistic regression model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data (typically WOE transformed).
        y : pd.Series
            Binary target variable (0/1).
        sample_weight : np.ndarray, optional
            Sample weights. Not directly supported by statsmodels Logit,
            but can be approximated using frequency weights.

        Returns
        -------
        LogisticModel
            Fitted instance.
        """
        try:
            import statsmodels.api as sm
        except ImportError:
            raise ImportError(
                "statsmodels is required for LogisticModel. "
                "Install it with: pip install statsmodels"
            )

        X = X.copy()
        y = y.copy()

        # Store feature names
        self.feature_names_ = X.columns.tolist()

        # Add constant if fitting intercept
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant="add")

        # Build model
        if sample_weight is not None:
            # Use frequency weights (approximate)
            self.model_ = sm.Logit(y, X, freq_weights=sample_weight)
        else:
            self.model_ = sm.Logit(y, X)

        # Fit model
        fit_kwargs = {
            "method": self.method,
            "maxiter": self.maxiter,
            "disp": False,
            **self.extra_kwargs,
        }

        if self.regularization == "l1":
            self.result_ = self.model_.fit_regularized(
                method="l1",
                alpha=self.alpha,
                disp=False,
            )
        elif self.regularization == "l2":
            # L2 not directly supported, use ridge approximation
            fit_kwargs["cov_type"] = "HC0"  # Robust standard errors
            self.result_ = self.model_.fit(**fit_kwargs)
        else:
            self.result_ = self.model_.fit(**fit_kwargs)

        # Extract coefficients
        self._extract_coefficients()

        self.is_fitted_ = True
        return self

    def _extract_coefficients(self):
        """Extract coefficients into a DataFrame."""
        if self.result_ is None:
            return

        coef_df = pd.DataFrame(
            {
                "feature": self.result_.params.index,
                "coefficient": self.result_.params.values,
                "std_error": self.result_.bse.values,
                "z_value": self.result_.tvalues.values,
                "p_value": self.result_.pvalues.values,
            }
        )

        # Add confidence intervals
        conf_int = self.result_.conf_int()
        coef_df["ci_lower"] = conf_int[0].values
        coef_df["ci_upper"] = conf_int[1].values

        # Add odds ratio
        coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])

        self.coefficients_ = coef_df

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of positive class.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.

        Returns
        -------
        np.ndarray
            Predicted probabilities for positive class.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit() first.")

        try:
            import statsmodels.api as sm
        except ImportError:
            raise ImportError("statsmodels is required.")

        X = X.copy()

        # Ensure same columns as training
        X = X[self.feature_names_]

        if self.fit_intercept:
            X = sm.add_constant(X, has_constant="add")

        return self.result_.predict(X)

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.
        threshold : float
            Classification threshold. Default 0.5.

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def summary(self) -> str:
        """
        Get statsmodels summary.

        Returns
        -------
        str
            Model summary as string.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit() first.")

        return self.result_.summary().as_text()

    def get_coefficients(self) -> pd.DataFrame:
        """
        Get coefficients DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with coefficient details.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit() first.")

        return self.coefficients_.copy()

    def get_significant_features(
        self,
        p_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """
        Get features with p-value below threshold.

        Parameters
        ----------
        p_threshold : float
            P-value threshold. Default 0.05.

        Returns
        -------
        pd.DataFrame
            Significant coefficients.
        """
        coef = self.get_coefficients()
        return coef[coef["p_value"] < p_threshold]

    def to_dict(self) -> Dict[str, Any]:
        """
        Export model parameters as dictionary.

        Returns
        -------
        Dict
            Model parameters including coefficients.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit() first.")

        return {
            "intercept": (
                float(
                    self.coefficients_[self.coefficients_["feature"] == "const"][
                        "coefficient"
                    ].values[0]
                )
                if self.fit_intercept
                else 0.0
            ),
            "coefficients": {
                row["feature"]: float(row["coefficient"])
                for _, row in self.coefficients_.iterrows()
                if row["feature"] != "const"
            },
            "feature_names": self.feature_names_,
        }
