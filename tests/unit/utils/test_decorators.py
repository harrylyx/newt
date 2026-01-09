import pytest

from newt.utils.decorators import requires_fit


class MockClass:
    def __init__(self):
        self.is_fitted_ = False
        self.custom_fitted = False

    def fit(self):
        self.is_fitted_ = True

    def custom_fit(self):
        self.custom_fitted = True

    @requires_fit()
    def predict(self):
        return "predicted"

    @requires_fit(attr_name="custom_fitted")
    def custom_predict(self):
        return "custom predicted"


def test_requires_fit_success():
    obj = MockClass()
    obj.fit()
    assert obj.predict() == "predicted"


def test_requires_fit_failure():
    obj = MockClass()
    with pytest.raises(ValueError, match="MockClass is not fitted"):
        obj.predict()


def test_custom_attr_success():
    obj = MockClass()
    obj.custom_fit()
    assert obj.custom_predict() == "custom predicted"


def test_custom_attr_failure():
    obj = MockClass()
    # Don't call custom_fit
    with pytest.raises(ValueError, match="MockClass is not fitted"):
        obj.custom_predict()
