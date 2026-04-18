"""
Logistic Regression model wrapper using statsmodels.

Provides a scikit-learn-like interface for statsmodels Logit.
"""

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from newt.config import MODELING
from newt.utils.decorators import requires_fit


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

    SERIALIZATION_VERSION = 1

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
        self.feature_names_: List[str] = []
        self.coefficients_: pd.DataFrame = pd.DataFrame()
        self.summary_text_: str = ""
        self.model_statistics_: Dict[str, float] = {}
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
        self._cache_fit_diagnostics()

        self.is_fitted_ = True
        return self

    def _extract_coefficients(self) -> None:
        """Extract coefficients into a DataFrame."""
        if self.result_ is None:
            return

        params = self.result_.params
        if hasattr(params, "index"):
            feature_index = [str(name) for name in params.index]
            coefficient_values = [float(value) for value in params.values]
        else:
            feature_index = self.feature_names_.copy()
            if self.fit_intercept:
                feature_index = ["const"] + feature_index
            coefficient_values = [float(value) for value in np.asarray(params).ravel()]

        coef_df = pd.DataFrame(
            {
                "feature": feature_index,
                "coefficient": coefficient_values,
                "std_error": [float(value) for value in self.result_.bse.values],
                "z_value": [float(value) for value in self.result_.tvalues.values],
                "p_value": [float(value) for value in self.result_.pvalues.values],
            }
        )

        # Add confidence intervals
        conf_int = self.result_.conf_int()
        coef_df["ci_lower"] = conf_int[0].values
        coef_df["ci_upper"] = conf_int[1].values

        # Add odds ratio
        coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])

        self.coefficients_ = coef_df

    def _cache_fit_diagnostics(self) -> None:
        """Cache summary text and model-level diagnostics for lightweight restore."""
        self.model_statistics_ = self._extract_model_statistics()
        if self.result_ is None:
            self.summary_text_ = ""
            return
        try:
            self.summary_text_ = str(self.result_.summary().as_text())
        except Exception:
            self.summary_text_ = ""

    def _extract_model_statistics(self) -> Dict[str, float]:
        """Extract finite model-level summary statistics."""
        if self.result_ is None:
            return {}

        mapping = {
            "aic": "aic",
            "bic": "bic",
            "llf": "log_likelihood",
            "prsquared": "pseudo_r2",
            "nobs": "nobs",
        }
        output: Dict[str, float] = {}
        for attr_name, output_name in mapping.items():
            value = getattr(self.result_, attr_name, None)
            numeric = self._as_finite_float(value)
            if numeric is None:
                continue
            output[output_name] = numeric
        return output

    def _intercept(self) -> float:
        """Return intercept coefficient (const) if present."""
        if not self.fit_intercept or self.coefficients_.empty:
            return 0.0

        const_row = self.coefficients_[self.coefficients_["feature"] == "const"]
        if const_row.empty:
            return 0.0
        return float(const_row["coefficient"].iloc[0])

    def _coefficient_map(self) -> Dict[str, float]:
        """Return feature coefficient mapping excluding intercept."""
        if self.coefficients_.empty:
            return {}
        coef_frame = self.coefficients_[self.coefficients_["feature"] != "const"]
        return {
            str(row["feature"]): float(row["coefficient"])
            for _, row in coef_frame.iterrows()
        }

    def _feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Return finite feature-level statistics from the coefficient table."""
        if self.coefficients_.empty:
            return {}

        fields = [
            "coefficient",
            "std_error",
            "z_value",
            "p_value",
            "ci_lower",
            "ci_upper",
            "odds_ratio",
        ]
        output: Dict[str, Dict[str, float]] = {}
        coef_frame = self.coefficients_[self.coefficients_["feature"] != "const"]
        for _, row in coef_frame.iterrows():
            feature = str(row["feature"])
            stats: Dict[str, float] = {}
            for field in fields:
                if field not in row:
                    continue
                numeric = self._as_finite_float(row[field])
                if numeric is None:
                    continue
                stats[field] = numeric
            if stats:
                output[feature] = stats
        return output

    @staticmethod
    def _as_finite_float(value: Any) -> Optional[float]:
        """Convert value to finite float if possible."""
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    @staticmethod
    def _normalize_model_statistics(raw: Any) -> Dict[str, float]:
        """Normalize persisted model statistics."""
        if not isinstance(raw, dict):
            return {}
        normalized: Dict[str, float] = {}
        for key, value in raw.items():
            numeric = LogisticModel._as_finite_float(value)
            if numeric is None:
                continue
            normalized[str(key)] = numeric
        return normalized

    @staticmethod
    def _serialize_extra_kwargs(raw: Any) -> Dict[str, Any]:
        """Keep only scalar fit kwargs that are safe to serialize."""
        if not isinstance(raw, dict):
            return {}
        output: Dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, (bool, int, str)):
                output[str(key)] = value
                continue
            if isinstance(value, float) and np.isfinite(value):
                output[str(key)] = float(value)
        return output

    @staticmethod
    def _resolve_newt_version() -> str:
        """Resolve installed package version if available."""
        try:
            return version("newt")
        except PackageNotFoundError:
            return "unknown"

    @classmethod
    def _build_coefficients_frame(
        cls,
        intercept: float,
        coefficients: Dict[str, Any],
        feature_names: List[str],
        feature_statistics: Any,
        fit_intercept: bool,
    ) -> pd.DataFrame:
        """Build coefficient frame for lightweight restored model."""
        stats_by_feature = (
            feature_statistics if isinstance(feature_statistics, dict) else {}
        )
        ordered_features: List[str] = [str(feature) for feature in feature_names]
        for feature in coefficients:
            feature_name = str(feature)
            if feature_name not in ordered_features:
                ordered_features.append(feature_name)

        records: List[Dict[str, Any]] = []
        if fit_intercept:
            records.append(
                {
                    "feature": "const",
                    "coefficient": float(intercept),
                    "std_error": np.nan,
                    "z_value": np.nan,
                    "p_value": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "odds_ratio": float(np.exp(intercept)),
                }
            )

        for feature in ordered_features:
            coefficient = cls._as_finite_float(coefficients.get(feature))
            if coefficient is None:
                coefficient = 0.0
            stats = stats_by_feature.get(feature, {})
            if not isinstance(stats, dict):
                stats = {}
            record = {
                "feature": feature,
                "coefficient": float(coefficient),
                "std_error": cls._as_finite_float(stats.get("std_error")),
                "z_value": cls._as_finite_float(stats.get("z_value")),
                "p_value": cls._as_finite_float(stats.get("p_value")),
                "ci_lower": cls._as_finite_float(stats.get("ci_lower")),
                "ci_upper": cls._as_finite_float(stats.get("ci_upper")),
                "odds_ratio": cls._as_finite_float(stats.get("odds_ratio")),
            }
            if record["odds_ratio"] is None:
                record["odds_ratio"] = float(np.exp(coefficient))
            records.append(record)

        return pd.DataFrame.from_records(records)

    @requires_fit()
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
        X = X.copy()

        # Ensure same columns as training
        X = X[self.feature_names_]

        if self.result_ is not None:
            try:
                import statsmodels.api as sm
            except ImportError:
                raise ImportError("statsmodels is required.")

            if self.fit_intercept:
                X = sm.add_constant(X, has_constant="add")

            return np.asarray(self.result_.predict(X), dtype=float)

        coefficients = self._coefficient_map()
        coef_vector = np.asarray(
            [coefficients.get(feature, 0.0) for feature in self.feature_names_],
            dtype=float,
        )
        linear_part = X.to_numpy(dtype=float) @ coef_vector + self._intercept()
        stabilized = np.clip(linear_part, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-stabilized))

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = MODELING.DEFAULT_CLASSIFICATION_THRESHOLD,
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

    @requires_fit()
    def summary(self) -> str:
        """
        Get statsmodels summary.

        Returns
        -------
        str
            Model summary as string.
        """
        if self.result_ is not None:
            return self.result_.summary().as_text()
        if self.summary_text_:
            return self.summary_text_
        return (
            "Model restored from serialized coefficients; "
            "statsmodels summary output is unavailable."
        )

    @requires_fit()
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get coefficients DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with coefficient details.
        """
        return self.coefficients_.copy()

    def get_significant_features(
        self,
        p_threshold: float = MODELING.DEFAULT_P_ENTER,
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

    @requires_fit()
    def to_dict(self) -> Dict[str, Any]:
        """
        Export model parameters as dictionary.

        Returns
        -------
        Dict
            Model parameters including coefficients.
        """
        coefficients = self._coefficient_map()
        ordered_coefficients = {
            feature: float(coefficients.get(feature, 0.0))
            for feature in self.feature_names_
        }
        for feature, coefficient in coefficients.items():
            if feature in ordered_coefficients:
                continue
            ordered_coefficients[feature] = float(coefficient)

        model_statistics = (
            self._extract_model_statistics()
            if self.result_ is not None
            else dict(self.model_statistics_)
        )
        summary_text = self.summary_text_
        if self.result_ is not None and not summary_text:
            try:
                summary_text = str(self.result_.summary().as_text())
            except Exception:
                summary_text = ""

        return {
            "schema_version": self.SERIALIZATION_VERSION,
            "newt_version": self._resolve_newt_version(),
            "fit_intercept": bool(self.fit_intercept),
            "method": str(self.method),
            "maxiter": int(self.maxiter),
            "regularization": self.regularization,
            "alpha": float(self.alpha),
            "extra_kwargs": self._serialize_extra_kwargs(self.extra_kwargs),
            "intercept": float(self._intercept()),
            "coefficients": ordered_coefficients,
            "feature_names": list(self.feature_names_),
            "feature_statistics": self._feature_statistics(),
            "model_statistics": model_statistics,
            "summary_text": summary_text,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LogisticModel":
        """
        Restore a fitted LogisticModel from serialized payload.

        Parameters
        ----------
        payload : Dict[str, Any]
            Dictionary exported by ``to_dict``.

        Returns
        -------
        LogisticModel
            Restored fitted model instance.
        """
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary.")
        raw_coefficients = payload.get("coefficients", {})
        if not isinstance(raw_coefficients, dict):
            raise ValueError("payload['coefficients'] must be a dictionary.")

        fit_intercept = bool(payload.get("fit_intercept", "intercept" in payload))
        method = str(payload.get("method", "bfgs"))
        maxiter = int(payload.get("maxiter", 100))
        regularization = payload.get("regularization")
        alpha = float(payload.get("alpha", 0.0))
        extra_kwargs = cls._serialize_extra_kwargs(payload.get("extra_kwargs", {}))

        model = cls(
            fit_intercept=fit_intercept,
            method=method,
            maxiter=maxiter,
            regularization=regularization,
            alpha=alpha,
            **extra_kwargs,
        )

        features = payload.get("feature_names", list(raw_coefficients.keys()))
        if not isinstance(features, list):
            raise ValueError("payload['feature_names'] must be a list if provided.")
        feature_names = [str(feature) for feature in features]
        for feature in raw_coefficients:
            feature_name = str(feature)
            if feature_name not in feature_names:
                feature_names.append(feature_name)

        intercept = cls._as_finite_float(payload.get("intercept"))
        if intercept is None:
            intercept = 0.0

        coefficients: Dict[str, float] = {}
        for feature, value in raw_coefficients.items():
            numeric = cls._as_finite_float(value)
            coefficients[str(feature)] = float(numeric) if numeric is not None else 0.0

        model.feature_names_ = feature_names
        model.model_ = None
        model.result_ = None
        model.coefficients_ = cls._build_coefficients_frame(
            intercept=intercept,
            coefficients=coefficients,
            feature_names=feature_names,
            feature_statistics=payload.get("feature_statistics", {}),
            fit_intercept=fit_intercept,
        )
        model.model_statistics_ = cls._normalize_model_statistics(
            payload.get("model_statistics", {})
        )
        model.summary_text_ = str(payload.get("summary_text", "") or "")
        model.is_fitted_ = True
        return model

    @requires_fit()
    def dump(self, path: Union[str, Path]) -> None:
        """
        Dump the model payload to a JSON file.

        Parameters
        ----------
        path : Union[str, Path]
            Output JSON path.
        """
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LogisticModel":
        """
        Load a model payload from JSON file.

        Parameters
        ----------
        path : Union[str, Path]
            Input JSON path.

        Returns
        -------
        LogisticModel
            Restored fitted model instance.
        """
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return cls.from_dict(payload)
