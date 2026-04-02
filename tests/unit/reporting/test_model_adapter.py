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
