"""Model adapters for report generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class ModelAdapter:
    """Unify feature importance and parameter extraction."""

    model: Any

    def __post_init__(self) -> None:
        self.model_family = self._detect_family()

    def get_feature_names(self) -> List[str]:
        """Return feature names from the wrapped model."""
        if self.model_family == "lightgbm":
            booster = self._get_lightgbm_booster()
            if hasattr(booster, "feature_name"):
                return list(booster.feature_name())
            if hasattr(self.model, "feature_name_"):
                return list(self.model.feature_name_)

        if self.model_family == "xgboost":
            booster = self._get_xgboost_booster()
            if hasattr(booster, "feature_names") and booster.feature_names is not None:
                return list(booster.feature_names)

        if hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)

        return []

    def get_importance_table(self) -> pd.DataFrame:
        """Return feature importance with gain and weight percentages."""
        if self.model_family == "lightgbm":
            booster = self._get_lightgbm_booster()
            features = self.get_feature_names()
            gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
            weight = np.asarray(
                booster.feature_importance(importance_type="split"),
                dtype=float,
            )
        elif self.model_family == "xgboost":
            booster = self._get_xgboost_booster()
            gain_map = booster.get_score(importance_type="gain")
            weight_map = booster.get_score(importance_type="weight")
            features = sorted(set(gain_map) | set(weight_map) | set(self.get_feature_names()))
            gain = np.asarray([float(gain_map.get(feature, 0.0)) for feature in features])
            weight = np.asarray([float(weight_map.get(feature, 0.0)) for feature in features])
        else:
            features = self.get_feature_names()
            raw = np.asarray(
                getattr(self.model, "feature_importances_", np.zeros(len(features), dtype=float)),
                dtype=float,
            )
            if not features:
                features = [f"feature_{index}" for index in range(len(raw))]
            gain = raw
            weight = raw

        gain_total = float(gain.sum())
        weight_total = float(weight.sum())
        table = pd.DataFrame(
            {
                "feature": features,
                "gain": gain,
                "gain_per": gain / gain_total if gain_total else 0.0,
                "weight": weight,
                "weight_per": weight / weight_total if weight_total else 0.0,
            }
        )
        return table.sort_values(["gain", "weight"], ascending=False).reset_index(drop=True)

    def get_param_table(self) -> pd.DataFrame:
        """Return a parameter table for the report."""
        params = self._collect_params()
        rows = [
            {
                "参数名称": key,
                "数值": self._resolve_param_value(params, aliases),
                "参数解释": description,
            }
            for key, aliases, description in PARAMETER_SPECS
        ]
        return pd.DataFrame(rows)

    def _detect_family(self) -> str:
        module_name = getattr(self.model.__class__, "__module__", "").lower()
        class_name = getattr(self.model.__class__, "__name__", "").lower()
        if "lightgbm" in module_name or "lgbm" in class_name:
            return "lightgbm"
        if "xgboost" in module_name or "xgb" in class_name:
            return "xgboost"
        if hasattr(self.model, "booster_") and hasattr(self.model.booster_, "feature_importance"):
            return "lightgbm"
        if hasattr(self.model, "get_booster"):
            return "xgboost"
        return "generic"

    def _get_lightgbm_booster(self) -> Any:
        return getattr(self.model, "booster_", self.model)

    def _get_xgboost_booster(self) -> Any:
        return self.model.get_booster()

    def _collect_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if hasattr(self.model, "get_params"):
            params.update(dict(self.model.get_params(deep=True)))
        for source in self._parameter_sources():
            params.update(source)
        return params

    def _parameter_sources(self) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        raw_model_params = getattr(self.model, "params", {})
        if isinstance(raw_model_params, dict):
            sources.append(raw_model_params)

        if hasattr(self.model, "booster_"):
            booster_params = getattr(self.model.booster_, "params", {})
            if isinstance(booster_params, dict):
                sources.append(booster_params)

        if self.model_family == "xgboost":
            booster = self._get_xgboost_booster()
            booster_attributes = getattr(booster, "attributes", None)
            if callable(booster_attributes):
                attribute_values = booster_attributes()
                if isinstance(attribute_values, dict):
                    sources.append(attribute_values)
        return sources

    def _resolve_param_value(
        self,
        params: Dict[str, Any],
        aliases: Sequence[str],
    ) -> Any:
        for alias in aliases:
            if alias in params:
                return params[alias]
        return ""


PARAMETER_SPECS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("n_estimators", ("n_estimators", "num_iterations"), "训练轮次"),
    ("learning_rate", ("learning_rate",), "学习率"),
    ("objective", ("objective",), "函数类型"),
    ("subsample", ("subsample", "bagging_fraction"), "训练样本采样比例"),
    ("colsample_bytree", ("colsample_bytree", "feature_fraction"), "特征采样率"),
    ("reg_alpha", ("reg_alpha", "lambda_l1"), "L1正则化系数"),
    ("reg_lambda", ("reg_lambda", "lambda_l2"), "L2正则化系数"),
    ("num_leaves", ("num_leaves",), "叶子节点数"),
    ("max_depth", ("max_depth",), "最大深度"),
)
