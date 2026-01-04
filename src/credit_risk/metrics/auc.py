import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Union

def calculate_auc(y_true: Union[np.ndarray, list], y_prob: Union[np.ndarray, list]) -> float:
    """
    Calculate Area Under the ROC Curve (AUC).
    
    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        
    Returns:
        float: AUC score.
    """
    return float(roc_auc_score(y_true, y_prob))
