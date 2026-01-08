from typing import Union

import numpy as np
from sklearn.metrics import roc_curve


def calculate_ks(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    sample_weight: Union[np.ndarray, list, None] = None,
) -> float:
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic.
    Uses roc_curve to handle weighted calculation efficiently.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities.
        sample_weight: Optional sample weights.

    Returns:
        float: KS statistic.
    """
    try:
        # roc_curve returns thresholds in descending order
        fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=sample_weight)

        # KS is max difference between TPR and FPR
        return float(np.max(np.abs(tpr - fpr)))

    except Exception as e:
        import warnings

        warnings.warn(f"Error calculating KS: {str(e)}", stacklevel=2)
        return np.nan
