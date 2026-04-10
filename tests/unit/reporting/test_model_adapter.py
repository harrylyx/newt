import pandas as pd

from newt.reporting.model_adapter import ModelAdapter


def test_model_adapter_extracts_lightgbm_importance(fake_lightgbm_model):
    adapter = ModelAdapter(fake_lightgbm_model)

    importance = adapter.get_importance_table()

    assert adapter.model_family == "lightgbm"
    assert list(importance["feature"]) == ["feature_a", "feature_b"]
    assert importance["gain"].sum() == 200.0
    assert importance["weight"].sum() == 30.0


def test_model_adapter_extracts_xgboost_importance(fake_xgboost_model):
    adapter = ModelAdapter(fake_xgboost_model)

    importance = adapter.get_importance_table()

    assert adapter.model_family == "xgboost"
    assert list(importance["feature"]) == ["feature_a", "feature_b"]
    assert importance["gain_per"].sum() == 1.0
    assert importance["weight_per"].sum() == 1.0


class DirectXGBoostBooster:
    def __init__(self):
        self.feature_names = ["feature_a", "feature_b"]

    def get_score(self, importance_type="gain"):
        if importance_type == "gain":
            return {"feature_a": 20.0, "feature_b": 10.0}
        if importance_type == "weight":
            return {"feature_a": 4.0, "feature_b": 2.0}
        raise ValueError(f"Unsupported importance type: {importance_type}")


def test_model_adapter_supports_direct_xgboost_booster():
    adapter = ModelAdapter(DirectXGBoostBooster())

    importance = adapter.get_importance_table()

    assert adapter.model_family == "xgboost"
    assert adapter.get_feature_names() == ["feature_a", "feature_b"]
    assert list(importance["feature"]) == ["feature_a", "feature_b"]
    assert importance["gain"].tolist() == [20.0, 10.0]
    assert importance["weight"].tolist() == [4.0, 2.0]


class AliasOnlyBooster:
    def __init__(self):
        self.params = {
            "num_iterations": 80,
            "learning_rate": 0.1,
            "objective": "binary",
            "bagging_fraction": 0.8,
            "feature_fraction": 0.75,
            "lambda_l1": 3,
            "lambda_l2": 4,
            "max_depth": -1,
        }

    def feature_name(self):
        return ["feature_a"]

    def feature_importance(self, importance_type="split"):
        if importance_type == "gain":
            return [1.0]
        return [1.0]


class AliasOnlyLightGBMModel:
    def __init__(self):
        self.booster_ = AliasOnlyBooster()

    def get_params(self, deep=True):
        return {}


def test_model_adapter_returns_fixed_parameter_rows_with_alias_fallback():
    adapter = ModelAdapter(AliasOnlyLightGBMModel())

    table = adapter.get_param_table()

    assert list(table["参数名称"]) == [
        "n_estimators",
        "learning_rate",
        "objective",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "num_leaves",
        "max_depth",
    ]
    assert list(table["参数解释"]) == [
        "训练轮次",
        "学习率",
        "函数类型",
        "训练样本采样比例",
        "特征采样率",
        "L1正则化系数",
        "L2正则化系数",
        "叶子节点数",
        "最大深度",
    ]

    values = table.set_index("参数名称")["数值"].to_dict()
    assert values["n_estimators"] == 80
    assert values["subsample"] == 0.8
    assert values["colsample_bytree"] == 0.75
    assert values["reg_alpha"] == 3
    assert values["reg_lambda"] == 4
    assert values["num_leaves"] == ""


def test_model_adapter_extracts_scorecard_metadata(fake_scorecard_model):
    adapter = ModelAdapter(fake_scorecard_model)

    assert adapter.model_family == "scorecard"
    assert adapter.get_feature_names() == ["feature_a", "feature_b"]

    importance = adapter.get_importance_table()
    assert list(importance["feature"]) == ["feature_a", "feature_b"]
    assert importance["gain"].gt(0).all()

    params = adapter.get_param_table()
    assert set(params["参数名称"]) == {
        "base_score",
        "pdo",
        "base_odds",
        "factor",
        "offset",
        "intercept_points",
    }

    feature_summary = adapter.get_lr_feature_summary_table()
    assert {
        "feature",
        "coefficient",
        "std_error",
        "z_value",
        "p_value",
        "ci_lower",
        "ci_upper",
        "odds_ratio",
    }.issubset(feature_summary.columns)
    assert feature_summary.set_index("feature").loc["feature_a", "p_value"] > 0

    model_summary = adapter.get_lr_model_summary_table()
    assert list(model_summary["统计项"]) == [
        "AIC",
        "BIC",
        "Log Likelihood",
        "Pseudo R²",
        "Nobs",
    ]
    assert model_summary.loc[model_summary["统计项"] == "AIC", "数值"].notna().all()

    assert not adapter.get_scorecard_base_table().empty
    assert not adapter.get_scorecard_points_table().empty


def test_model_adapter_scorecard_missing_stats_is_safe(fake_scorecard_model):
    from newt.modeling.scorecard import Scorecard

    payload = fake_scorecard_model.to_dict()
    payload.pop("feature_statistics", None)
    payload.pop("model_statistics", None)
    restored = Scorecard().from_dict(payload)
    adapter = ModelAdapter(restored)

    feature_summary = adapter.get_lr_feature_summary_table()
    assert list(feature_summary["feature"]) == ["feature_a", "feature_b"]
    assert feature_summary["p_value"].isna().all()

    model_summary = adapter.get_lr_model_summary_table()
    assert isinstance(model_summary, pd.DataFrame)
    assert model_summary["数值"].isna().all()
