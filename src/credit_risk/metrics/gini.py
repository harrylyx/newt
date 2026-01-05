import numpy as np
from typing import Union
from .auc import calculate_auc

def calculate_gini(y_true: Union[np.ndarray, list], y_prob: Union[np.ndarray, list]) -> float:
    """
    Calculate Gini coefficient.
    Gini = 2 * AUC - 1
    
    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        
    Returns:
        float: Gini coefficient.
    """
    try:
        auc = calculate_auc(y_true, y_prob)
        return 2 * auc - 1
    except Exception as e:
        import warnings
        warnings.warn(f"Error calculating Gini: {str(e)}")
        return np.nan
