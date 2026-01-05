import numpy as np
import pandas as pd
from typing import Union, List, Dict


def calculate_lift(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    bins: int = 10,
) -> pd.DataFrame:
    """
    Calculate Lift table.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        bins: Number of bins (deciles).

    Returns:
        pd.DataFrame: Lift table with columns for bin, min_prob, max_prob, count,
        events, event_rate, lift.
    """
    try:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        data = pd.DataFrame({"true": y_true, "prob": y_prob})

        # Create bins using qcut
        data["bin"] = pd.qcut(
            data["prob"], q=bins, duplicates="drop", labels=False
        )

        # If bins are collapsed due to duplicates, we might have fewer bins.
        # Reverse bin labels so 0 is highest probability (top decile) if preferred,
        # but qcut assigns 0 to lowest values.
        # Usually in credit risk, we want bin 1 (or 0) to be high risk.
        # Let's standardize: bin 0 = lowest prob, bin (N-1) = highest prob.
        # Often 'Lift' checks the highest decile.

        agg = data.groupby("bin").agg(
            {"prob": ["min", "max"], "true": ["count", "sum"]}
        )

        agg.columns = ["min_prob", "max_prob", "count", "events"]
        agg = agg.sort_index(
            ascending=False
        ).reset_index()  # Highest prob first

        global_event_rate = y_true.sum() / len(y_true)
        agg["event_rate"] = agg["events"] / agg["count"]
        agg["lift"] = agg["event_rate"] / global_event_rate

        return agg
    except Exception as e:
        import warnings

        warnings.warn(f"Error calculating Lift: {str(e)}")
        return pd.DataFrame()


def calculate_lift_at_k(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    k: float = 0.1,
) -> float:
    """
    Calculate Lift at top k portion (e.g. top 10%).

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        k: Fraction of top predictions to consider (e.g., 0.1 for 10%).

    Returns:
        float: Lift value at k.
    """
    try:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Validate k
        if not 0 < k <= 1:
            raise ValueError("k must be between 0 and 1")

        n = len(y_true)
        if n == 0:
            return np.nan

        # Global event rate
        global_event_rate = np.mean(y_true)
        if global_event_rate == 0:
            return 0.0  # No events, lift is 0? Or undefined? usually 0 or 1?
            # If no events, standard lift is technically undefined (div by 0).
            # Return 0 or nan.

        # Target count for top k
        n_top = int(np.ceil(n * k))
        n_top = max(1, n_top)  # Ensure at least 1

        # Sort by prob desc
        idx = np.argsort(y_prob)[::-1]
        top_idx = idx[:n_top]

        top_y_true = y_true[top_idx]
        top_event_rate = np.mean(top_y_true)

        return top_event_rate / global_event_rate

    except Exception as e:
        import warnings

        warnings.warn(f"Error calculating Lift@{k}: {str(e)}")
        return np.nan
