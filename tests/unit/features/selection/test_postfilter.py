import numpy as np
import pandas as pd
import pytest

from newt.features.selection import PostFilter


@pytest.fixture
def collinear_frames():
    train = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5, np.nan],
            "x2": [0.2, 0.4, 0.6, 0.8, 1.0, np.nan],
            "x3": [1.0, 0.7, 0.4, 0.2, -0.1, 0.0],
        }
    )
    test = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.35, 0.45, np.nan, 0.55],
            "x2": [0.2, 0.4, 0.7, 0.9, np.nan, 1.1],
            "x3": [0.9, 0.6, 0.5, 0.1, 0.0, -0.2],
        }
    )
    return train, test


def test_postfilter_rejects_invalid_nan_strategy():
    with pytest.raises(ValueError, match="psi_nan_strategy"):
        PostFilter(psi_nan_strategy="keep")


def test_postfilter_iteratively_removes_high_vif_feature(collinear_frames):
    X_train, _ = collinear_frames
    postfilter = PostFilter(vif_threshold=5.0)

    transformed = postfilter.fit_transform(X_train)

    assert postfilter.is_fitted_
    assert transformed.columns.tolist() == postfilter.selected_features_
    assert {"x1", "x2"} & set(postfilter.removed_features_)
    assert not {"x1", "x2"}.issubset(set(postfilter.selected_features_))


def test_postfilter_report_includes_status_reason_and_metrics(collinear_frames):
    X_train, X_test = collinear_frames
    postfilter = PostFilter(vif_threshold=5.0, psi_threshold=0.01)
    postfilter.fit(X_train, X_test)

    report = postfilter.report()

    assert set(report) == {"summary", "psi", "vif"}
    assert {"feature", "status", "reason"}.issubset(report["summary"].columns)
    assert isinstance(report["psi"], pd.DataFrame)
    assert isinstance(report["vif"], pd.DataFrame)
