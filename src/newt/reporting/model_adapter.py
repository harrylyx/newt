"""Model adapters for report generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        if self.model_family == "scorecard":
            if hasattr(self.model, "feature_names_"):
                return [str(feature) for feature in self.model.feature_names_]
            spec = self.model.spec_ if hasattr(self.model, "spec_") else None
            if spec is not None and hasattr(spec, "feature_names"):
                return [str(feature) for feature in spec.feature_names]
            return []

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
        if self.model_family == "scorecard":
            return self._build_scorecard_importance_table()
        if self.model_family == "lightgbm":
            booster = self._get_lightgbm_booster()
            features = self.get_feature_names()
            gain = np.asarray(
                booster.feature_importance(importance_type="gain"), dtype=float
            )
            weight = np.asarray(
                booster.feature_importance(importance_type="split"),
                dtype=float,
            )
        elif self.model_family == "xgboost":
            booster = self._get_xgboost_booster()
            gain_map = booster.get_score(importance_type="gain")
            weight_map = booster.get_score(importance_type="weight")
            features = sorted(
                set(gain_map) | set(weight_map) | set(self.get_feature_names())
            )
            gain = np.asarray(
                [float(gain_map.get(feature, 0.0)) for feature in features]
            )
            weight = np.asarray(
                [float(weight_map.get(feature, 0.0)) for feature in features]
            )
        else:
            features = self.get_feature_names()
            raw = np.asarray(
                getattr(
                    self.model,
                    "feature_importances_",
                    np.zeros(len(features), dtype=float),
                ),
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
        return table.sort_values(["gain", "weight"], ascending=False).reset_index(
            drop=True
        )

    def get_param_table(self) -> pd.DataFrame:
        """Return a parameter table for the report."""
        if self.model_family == "scorecard":
            return self._build_scorecard_param_table()
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

    def get_lr_feature_summary_table(self) -> pd.DataFrame:
        """Return feature-level logistic summary statistics for scorecard models."""
        columns = [
            "feature",
            "coefficient",
            "std_error",
            "z_value",
            "p_value",
            "ci_lower",
            "ci_upper",
            "odds_ratio",
        ]
        if self.model_family != "scorecard":
            return pd.DataFrame(columns=columns)

        feature_names = self.get_feature_names()
        if not feature_names:
            return pd.DataFrame(columns=columns)

        stats_frame = self._get_scorecard_feature_statistics_frame()
        if stats_frame.empty:
            return pd.DataFrame({"feature": feature_names}).reindex(columns=columns)

        if "feature" not in stats_frame.columns:
            return pd.DataFrame({"feature": feature_names}).reindex(columns=columns)

        normalized = stats_frame.copy()
        normalized["feature"] = normalized["feature"].astype(str)
        normalized = normalized.drop_duplicates("feature", keep="first")
        output = pd.DataFrame({"feature": feature_names}).merge(
            normalized,
            on="feature",
            how="left",
        )
        return output.reindex(columns=columns)

    def get_lr_model_summary_table(self) -> pd.DataFrame:
        """Return model-level logistic summary statistics for scorecard models."""
        if self.model_family != "scorecard":
            return pd.DataFrame(columns=["统计项", "数值"])

        summary = self._get_scorecard_model_statistics()
        order = [
            ("aic", "AIC"),
            ("bic", "BIC"),
            ("log_likelihood", "Log Likelihood"),
            ("pseudo_r2", "Pseudo R²"),
            ("nobs", "Nobs"),
        ]
        rows = []
        for metric_key, display_name in order:
            rows.append({"统计项": display_name, "数值": summary.get(metric_key, np.nan)})
        return pd.DataFrame(rows)

    def get_scorecard_base_table(self) -> pd.DataFrame:
        """Return scorecard scaling parameters for scorecard models."""
        if self.model_family != "scorecard":
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "base_score": self.model.base_score
                    if hasattr(self.model, "base_score")
                    else np.nan,
                    "pdo": self.model.pdo if hasattr(self.model, "pdo") else np.nan,
                    "base_odds": self.model.base_odds
                    if hasattr(self.model, "base_odds")
                    else np.nan,
                    "factor": self.model.factor
                    if hasattr(self.model, "factor")
                    else np.nan,
                    "offset": self.model.offset
                    if hasattr(self.model, "offset")
                    else np.nan,
                    "intercept_points": self.model.intercept_points_
                    if hasattr(self.model, "intercept_points_")
                    else np.nan,
                }
            ]
        )

    def get_scorecard_points_table(self) -> pd.DataFrame:
        """Return scorecard point decomposition table for scorecard models."""
        if self.model_family != "scorecard":
            return pd.DataFrame()
        if hasattr(self.model, "export"):
            return self.model.export()
        return pd.DataFrame()

    def _detect_family(self) -> str:
        module_name = getattr(self.model.__class__, "__module__", "").lower()
        class_name = getattr(self.model.__class__, "__name__", "").lower()
        if (
            "newt.modeling.scorecard" in module_name
            or class_name == "scorecard"
            or (
                hasattr(self.model, "feature_names_")
                and hasattr(self.model, "scorecard_")
                and hasattr(self.model, "intercept_points_")
            )
        ):
            return "scorecard"
        if "lightgbm" in module_name or "lgbm" in class_name:
            return "lightgbm"
        if "xgboost" in module_name or "xgb" in class_name:
            return "xgboost"
        if hasattr(self.model, "booster_") and hasattr(
            self.model.booster_, "feature_importance"
        ):
            return "lightgbm"
        if hasattr(self.model, "get_booster"):
            return "xgboost"
        return "generic"

    def _get_lightgbm_booster(self) -> Any:
        return getattr(self.model, "booster_", self.model)

    def _get_xgboost_booster(self) -> Any:
        # xgboost models can be sklearn wrappers (with get_booster)
        # or native Booster objects (with get_score).
        if hasattr(self.model, "get_booster"):
            return self.model.get_booster()
        if hasattr(self.model, "get_score"):
            return self.model
        raise AttributeError("Unsupported xgboost model type: missing booster handle")

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

    def _build_scorecard_importance_table(self) -> pd.DataFrame:
        """Build a pseudo-importance table from scorecard feature statistics."""
        features = self.get_feature_names()
        if not features:
            return pd.DataFrame(
                columns=["feature", "gain", "gain_per", "weight", "weight_per"]
            )

        stats = self.get_lr_feature_summary_table().set_index("feature")
        gain_values = stats.get("coefficient")
        if gain_values is not None:
            gain_list: List[float] = []
            for feature in features:
                numeric = self._to_finite_float(gain_values.get(feature, np.nan))
                gain_list.append(abs(numeric) if numeric is not None else np.nan)
            gain = np.asarray(gain_list, dtype=float)
        else:
            gain = np.zeros(len(features), dtype=float)

        if not np.isfinite(gain).any() or np.allclose(
            np.nan_to_num(gain, nan=0.0), 0.0
        ):
            gain = self._estimate_importance_from_points(features)

        gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
        weight = gain.copy()
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
        return table.sort_values(["gain", "weight"], ascending=False).reset_index(
            drop=True
        )

    def _estimate_importance_from_points(self, features: Sequence[str]) -> np.ndarray:
        points_by_feature = getattr(self.model, "scorecard_", {})
        values: List[float] = []
        for feature in features:
            table = points_by_feature.get(feature)
            if isinstance(table, pd.DataFrame) and "points" in table.columns:
                points = pd.to_numeric(table["points"], errors="coerce")
                numeric = self._to_finite_float(points.abs().mean(skipna=True))
                values.append(0.0 if numeric is None else numeric)
            else:
                values.append(0.0)
        return np.asarray(values, dtype=float)

    def _build_scorecard_param_table(self) -> pd.DataFrame:
        rows = [
            {
                "参数名称": "base_score",
                "数值": getattr(self.model, "base_score", ""),
                "参数解释": "基准分",
            },
            {
                "参数名称": "pdo",
                "数值": getattr(self.model, "pdo", ""),
                "参数解释": "翻倍赔率分值（PDO）",
            },
            {
                "参数名称": "base_odds",
                "数值": getattr(self.model, "base_odds", ""),
                "参数解释": "基准赔率",
            },
            {
                "参数名称": "factor",
                "数值": getattr(self.model, "factor", ""),
                "参数解释": "缩放因子",
            },
            {
                "参数名称": "offset",
                "数值": getattr(self.model, "offset", ""),
                "参数解释": "缩放偏移",
            },
            {
                "参数名称": "intercept_points",
                "数值": getattr(self.model, "intercept_points_", ""),
                "参数解释": "截距分值",
            },
        ]
        return pd.DataFrame(rows)

    def _get_scorecard_feature_statistics_frame(self) -> pd.DataFrame:
        stats = (
            self.model.feature_statistics_
            if hasattr(self.model, "feature_statistics_")
            else pd.DataFrame()
        )
        if isinstance(stats, pd.DataFrame) and not stats.empty:
            return stats.copy()
        spec = self.model.spec_ if hasattr(self.model, "spec_") else None
        spec_stats = (
            spec.feature_statistics
            if spec is not None and hasattr(spec, "feature_statistics")
            else {}
        )
        if not spec_stats:
            return pd.DataFrame()
        return (
            pd.DataFrame.from_dict(spec_stats, orient="index")
            .reset_index()
            .rename(columns={"index": "feature"})
        )

    def _get_scorecard_model_statistics(self) -> Dict[str, float]:
        stats = (
            self.model.model_statistics_
            if hasattr(self.model, "model_statistics_")
            else {}
        )
        if isinstance(stats, dict) and stats:
            output: Dict[str, float] = {}
            for metric, value in stats.items():
                numeric = self._to_finite_float(value)
                if numeric is None:
                    continue
                output[str(metric)] = numeric
            return output
        spec = self.model.spec_ if hasattr(self.model, "spec_") else None
        spec_stats = (
            spec.model_statistics
            if spec is not None and hasattr(spec, "model_statistics")
            else {}
        )
        if isinstance(spec_stats, dict):
            output: Dict[str, float] = {}
            for metric, value in spec_stats.items():
                numeric = self._to_finite_float(value)
                if numeric is None:
                    continue
                output[str(metric)] = numeric
            return output
        return {}

    def _to_finite_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric


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
