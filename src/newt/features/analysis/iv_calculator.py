import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional
from .woe_calculator import WOEEncoder

def calculate_iv(
    df: pd.DataFrame, 
    target: str, 
    feature: str, 
    buckets: int = 10, 
    epsilon: float = 1e-8
) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Calculate Information Value (IV) for a feature.
    High performance implementation using vectorized operations via WOEEncoder.
    
    Args:
        df: Input DataFrame.
        target: Target column name (binary 0/1).
        feature: Feature column name.
        buckets: Number of buckets for numerical features (if not already binned).
        epsilon: Small constant to avoid division by zero or log(0).
        
    Returns:
        Dict containing 'iv' (float) and 'woe_table' (pd.DataFrame).
    """
    encoder = WOEEncoder(buckets=buckets, epsilon=epsilon)
    encoder.fit(df[feature], df[target])
    
    return {
        'iv': float(encoder.iv_),
        'woe_table': encoder.summary_
    }
