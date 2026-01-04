import numpy as np
from scipy.stats import ks_2samp
from typing import Union, Tuple

def calculate_ks(y_true: Union[np.ndarray, list], y_prob: Union[np.ndarray, list]) -> float:
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic.
    
    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities.
        
    Returns:
        float: KS statistic.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Calculate KS using scipy which computes the max difference between CDFs
    # of the two distributions (positives and negatives)
    return ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0]).statistic

def calculate_ks_fast(y_true: Union[np.ndarray, list], y_prob: Union[np.ndarray, list]) -> float:
    """
    Alternative fast implementation using sorting.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Sort by probability descending
    idx = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[idx]
    
    # Cumulative sum of positives and negatives
    cumsum_pos = np.cumsum(y_true_sorted)
    cumsum_neg = np.cumsum(1 - y_true_sorted)
    
    # Calculate TPR and FPR
    tpr = cumsum_pos / cumsum_pos[-1]
    fpr = cumsum_neg / cumsum_neg[-1]
    
    # KS is max difference
    return np.max(tpr - fpr)
