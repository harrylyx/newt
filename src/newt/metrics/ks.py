import numpy as np
from typing import Union


def calculate_ks(
    y_true: Union[np.ndarray, list], y_prob: Union[np.ndarray, list]
) -> float:
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic using efficient sorting.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities.

    Returns:
        float: KS statistic.
    """
    try:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Sort by probability descending
        idx = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[idx]

        # Cumulative sum of positives and negatives
        cumsum_pos = np.cumsum(y_true_sorted)
        cumsum_neg = np.cumsum(1 - y_true_sorted)

        # Calculate TPR and FPR
        # Handle division by zero if no positives or no negatives
        total_pos = cumsum_pos[-1]
        total_neg = cumsum_neg[-1]

        if total_pos == 0 or total_neg == 0:
            return 0.0  # Or nan? Standard KS is 0 if one class missing or undefined.

        tpr = cumsum_pos / total_pos
        fpr = cumsum_neg / total_neg

        # KS is max difference
        return np.max(tpr - fpr)

    except Exception as e:
        import warnings

        warnings.warn(f"Error calculating KS: {str(e)}")
        return np.nan
