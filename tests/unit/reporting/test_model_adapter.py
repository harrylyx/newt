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
