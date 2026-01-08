import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from newt.modeling.logistic import LogisticModel

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    y = (X['feature1'] + X['feature2'] > 0).astype(int)
    return X, y

def test_logistic_init():
    model = LogisticModel(fit_intercept=False, method='newton')
    assert model.fit_intercept is False
    assert model.method == 'newton'

def test_logistic_fit(sample_data):
    X, y = sample_data
    model = LogisticModel()
    model.fit(X, y)
    
    assert model.is_fitted_
    assert model.model_ is not None
    assert model.result_ is not None
    assert len(model.coefficients_) == 3  # 2 features + intercept
    assert 'const' in model.feature_names_ or 'const' in model.coefficients_['feature'].values

def test_logistic_predict(sample_data):
    X, y = sample_data
    model = LogisticModel()
    model.fit(X, y)
    
    probs = model.predict_proba(X)
    assert len(probs) == len(X)
    assert np.all((probs >= 0) & (probs <= 1))
    
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})

def test_logistic_coefficients(sample_data):
    X, y = sample_data
    model = LogisticModel()
    model.fit(X, y)
    
    coefs = model.get_coefficients()
    assert isinstance(coefs, pd.DataFrame)
    required_cols = {'feature', 'coefficient', 'p_value', 'odds_ratio'}
    assert required_cols.issubset(coefs.columns)

def test_logistic_to_dict(sample_data):
    X, y = sample_data
    model = LogisticModel()
    model.fit(X, y)
    
    model_dict = model.to_dict()
    assert 'intercept' in model_dict
    assert 'coefficients' in model_dict
    assert isinstance(model_dict['coefficients'], dict)

def test_logistic_not_fitted_error():
    model = LogisticModel()
    with pytest.raises(ValueError, match="LogisticModel is not fitted"):
        model.predict(pd.DataFrame({'a': [1]}))

def test_logistic_regularization_l1(sample_data):
    X, y = sample_data
    model = LogisticModel(regularization='l1', alpha=0.1)
    model.fit(X, y)
    assert model.is_fitted_

def test_logistic_regularization_l2(sample_data):
    X, y = sample_data
    model = LogisticModel(regularization='l2', alpha=0.1)
    model.fit(X, y)
    assert model.is_fitted_
