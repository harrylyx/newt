from typing import Union

import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_auc(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    sample_weight: Union[np.ndarray, list, None] = None,
) -> float:
    """
    Calculate Area Under the ROC Curve (AUC).

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        sample_weight: Optional sample weights.

    Returns:
        float: AUC score.
    """
    try:
        return float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight))
    except Exception as e:
        import warnings

        warnings.warn(f"Error calculating AUC: {str(e)}", stacklevel=2)
        return np.nan
