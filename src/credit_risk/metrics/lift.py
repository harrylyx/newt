import numpy as np
import pandas as pd
from typing import Union, List, Dict

def calculate_lift(y_true: Union[np.ndarray, list], y_prob: Union[np.ndarray, list], bins: int = 10) -> pd.DataFrame:
    """
    Calculate Lift table.
    
    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        bins: Number of bins (deciles).
        
    Returns:
        pd.DataFrame: Lift table with columns for bin, min_prob, max_prob, count, events, event_rate, lift.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    data = pd.DataFrame({'true': y_true, 'prob': y_prob})
    
    # Create bins using qcut
    data['bin'] = pd.qcut(data['prob'], q=bins, duplicates='drop', labels=False)
    
    # If bins are collapsed due to duplicates, we might have fewer bins. 
    # Reverse bin labels so 0 is highest probability (top decile) if preferred, 
    # but qcut assigns 0 to lowest values. 
    # Usually in credit risk, we want bin 1 (or 0) to be high risk or high probability.
    # Let's standardize: bin 0 = lowest prob, bin (N-1) = highest prob.
    # Often 'Lift' checks the highest decile.
    
    agg = data.groupby('bin').agg({
        'prob': ['min', 'max'],
        'true': ['count', 'sum']
    })
    
    agg.columns = ['min_prob', 'max_prob', 'count', 'events']
    agg = agg.sort_index(ascending=False).reset_index() # Highest prob first
    
    global_event_rate = y_true.sum() / len(y_true)
    agg['event_rate'] = agg['events'] / agg['count']
    agg['lift'] = agg['event_rate'] / global_event_rate
    
    return agg
