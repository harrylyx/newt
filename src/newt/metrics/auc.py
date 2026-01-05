import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Union


def calculate_auc(
    y_true: Union[np.ndarray, list], y_prob: Union[np.ndarray, list]
) -> float:
    """
    Calculate Area Under the ROC Curve (AUC).

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.

    Returns:
        float: AUC score.
    """
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception as e:
        import warnings

        warnings.warn(f"Error calculating AUC: {str(e)}")
        return np.nan
