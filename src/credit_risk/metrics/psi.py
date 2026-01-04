import numpy as np
from typing import Union

def calculate_psi(expected: Union[np.ndarray, list], actual: Union[np.ndarray, list], buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    Args:
        expected: Expected distribution (e.g. training set scores).
        actual: Actual distribution (e.g. production/validation set scores).
        buckets: Number of buckets/bins.
        
    Returns:
        float: PSI value.
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    
    # Define breakpoints based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    
    # Handle unique boundaries if many duplicates
    breakpoints = np.unique(breakpoints)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    epsilon = 1e-8
    expected_percents = np.maximum(expected_percents, epsilon)
    actual_percents = np.maximum(actual_percents, epsilon)
    
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return np.sum(psi_values)
