import numpy as np
from typing import Union


def calculate_psi(expected: Union[np.ndarray, list], actual: Union[np.ndarray, list], buckets: int = 10, include_nan: bool = True) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    Args:
        expected: Expected distribution (e.g. training set scores).
        actual: Actual distribution (e.g. production/validation set scores).
        buckets: Number of buckets/bins for non-NaN values.
        include_nan: If True, NaN values are treated as a separate bucket. 
                     If False, NaN values are dropped before calculation.
        
    Returns:
        float: PSI value.
    """
    try:
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        
        # Separate NaN and non-NaN
        expected_nan_mask = np.isnan(expected)
        actual_nan_mask = np.isnan(actual)
        
        expected_not_nan = expected[~expected_nan_mask]
        actual_not_nan = actual[~actual_nan_mask]
        
        # Calculate counts for non-NaN
        if len(expected_not_nan) > 0:
            breakpoints = np.percentile(expected_not_nan, np.linspace(0, 100, buckets + 1))
            # Handle unique boundaries
            breakpoints = np.unique(breakpoints)
            
            # Ensure at least 2 edges
            if len(breakpoints) < 2:
                breakpoints = np.array([-np.inf, np.inf])
            else:
                breakpoints[0] = -np.inf
                breakpoints[-1] = np.inf
            
            expected_counts = np.histogram(expected_not_nan, breakpoints)[0]
            actual_counts = np.histogram(actual_not_nan, breakpoints)[0]
        else:
            # All NaNs in expected? Handle edge case
            expected_counts = np.zeros(buckets)
            actual_counts = np.zeros(buckets) 
        
        # Handle NaN bucket if requested
        if include_nan:
            expected_nan_count = np.sum(expected_nan_mask)
            actual_nan_count = np.sum(actual_nan_mask)
            
            expected_counts = np.append(expected_counts, expected_nan_count)
            actual_counts = np.append(actual_counts, actual_nan_count)
            # If not including nan, we just use the counts from non-nan parts
        
        # Calculate proportions
        expected_total = np.sum(expected_counts)
        actual_total = np.sum(actual_counts)
        
        if expected_total == 0 or actual_total == 0:
            return np.nan
            
        expected_percents = expected_counts / expected_total
        actual_percents = actual_counts / actual_total
        
        # Avoid division by zero
        epsilon = 1e-8
        expected_percents = np.maximum(expected_percents, epsilon)
        actual_percents = np.maximum(actual_percents, epsilon)
        
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        return np.sum(psi_values)

    except Exception as e:
        import warnings
        warnings.warn(f"Error calculating PSI: {str(e)}")
        return np.nan
