
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from src.newt.metrics.auc import calculate_auc
from src.newt.metrics.ks import calculate_ks
from src.newt.metrics.gini import calculate_gini


def test_weighted_auc():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    weights = np.array([1, 1, 1, 1])

    # 1. Base case: equal weights should match unweighted
    auc = calculate_auc(y_true, y_prob, sample_weight=None)
    auc_w = calculate_auc(y_true, y_prob, sample_weight=weights)
    assert auc == auc_w

    # 2. Unequal weights
    # Make the mistake (0 as high prob, 1 as low prob) heavier
    # 0.4 (Class 0) is > 0.35 (Class 1). This is a ranking error.
    # If we weight the error pair heavily, AUC should drop.
    weights_heavy = np.array([1, 10, 10, 1])
    auc_heavy = calculate_auc(y_true, y_prob, sample_weight=weights_heavy)

    # Sklearn comparison
    sk_auc = roc_auc_score(y_true, y_prob, sample_weight=weights_heavy)
    assert abs(auc_heavy - sk_auc) < 1e-9


def test_weighted_ks():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.4, 0.8])
    # 0.3 (Class 0) vs 0.4 (Class 1). Correctly ordered. KS should be 1.0 unweighted.

    ks_unweighted = calculate_ks(y_true, y_prob)
    assert ks_unweighted == 1.0

    # Weighted check
    # Effectively duplication
    weights = np.array([1, 2, 3, 1])
    # Equivalence: 0(0.1), 0(0.3)x2, 1(0.4)x3, 1(0.8)

    ks_weighted = calculate_ks(y_true, y_prob, sample_weight=weights)

    fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=weights)
    expected_ks = np.max(np.abs(tpr - fpr))

    assert abs(ks_weighted - expected_ks) < 1e-9
    assert ks_weighted == 1.0  # Still perfect separation


def test_weighted_ks_equivalence():
    # Verify that weight 2 is same as 2 samples
    y_true = np.array([0, 1])
    y_prob = np.array([0.4, 0.6])
    w = np.array([2, 1])

    # Expanded
    y_true_exp = np.array([0, 0, 1])
    y_prob_exp = np.array([0.4, 0.4, 0.6])

    ks_w = calculate_ks(y_true, y_prob, sample_weight=w)
    ks_exp = calculate_ks(y_true_exp, y_prob_exp)

    assert abs(ks_w - ks_exp) < 1e-9


def test_weighted_gini():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    weights = np.array([1, 10, 10, 1])

    gini = calculate_gini(y_true, y_prob, sample_weight=weights)
    auc = calculate_auc(y_true, y_prob, sample_weight=weights)

    assert abs(gini - (2 * auc - 1)) < 1e-9


def test_ks_empty_weights():
    y_true = np.array([0, 1])
    y_prob = np.array([0.4, 0.6])
    ks = calculate_ks(y_true, y_prob, sample_weight=None)
    assert ks == 1.0
