from typing import Optional, Union

import numpy as np
import pandas as pd

from newt.config import BINNING

NAN_STRATEGIES = frozenset(["separate", "exclude"])


def calculate_psi(
    expected: Union[np.ndarray, list],
    actual: Union[np.ndarray, list],
    buckets: int = BINNING.DEFAULT_BUCKETS,
    include_nan: bool = True,
    nan_strategy: Optional[str] = None,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    Args:
        expected: Expected distribution (e.g. training set scores).
        actual: Actual distribution (e.g. production/validation set scores).
        buckets: Number of buckets/bins for non-NaN values.
        include_nan: Backward-compatible flag.
            - True: treat missing values as a separate bucket.
            - False: drop missing values before calculation.
        nan_strategy: Missing-value handling strategy.
            - 'separate': treat missing values as a separate bucket.
            - 'exclude': drop missing values before calculation.
            If None, strategy is inferred from include_nan.

    Returns:
        float: PSI value.
    """
    try:
        if buckets < 1:
            raise ValueError("buckets must be >= 1")

        strategy = nan_strategy
        if strategy is None:
            strategy = "separate" if include_nan else "exclude"
        elif strategy not in NAN_STRATEGIES:
            raise ValueError(
                f"nan_strategy must be one of {sorted(NAN_STRATEGIES)}, got: {strategy}"
            )

        expected = pd.to_numeric(
            pd.Series(np.asarray(expected).ravel()),
            errors="coerce",
        ).to_numpy(dtype=float)
        actual = pd.to_numeric(
            pd.Series(np.asarray(actual).ravel()),
            errors="coerce",
        ).to_numpy(dtype=float)

        # Separate missing and non-missing values
        expected_nan_mask = np.isnan(expected)
        actual_nan_mask = np.isnan(actual)

        expected_not_nan = expected[~expected_nan_mask]
        actual_not_nan = actual[~actual_nan_mask]

        # Calculate counts for non-NaN
        if len(expected_not_nan) > 0:
            breakpoints = np.percentile(
                expected_not_nan, np.linspace(0, 100, buckets + 1)
            )
            # Handle unique boundaries
            breakpoints = np.unique(breakpoints)

            # Ensure at least 2 edges
            if len(breakpoints) < 2:
                breakpoints = np.array([-np.inf, np.inf])
            else:
                breakpoints[0] = -np.inf
                breakpoints[-1] = np.inf
        else:
            # No valid values in expected: keep one non-missing bucket
            # so that actual non-missing values are still counted.
            breakpoints = np.array([-np.inf, np.inf], dtype=float)

        expected_counts = np.histogram(expected_not_nan, breakpoints)[0]
        actual_counts = np.histogram(actual_not_nan, breakpoints)[0]

        # Handle NaN bucket if requested
        if strategy == "separate":
            expected_nan_count = np.sum(expected_nan_mask)
            actual_nan_count = np.sum(actual_nan_mask)

            expected_counts = np.append(expected_counts, expected_nan_count)
            actual_counts = np.append(actual_counts, actual_nan_count)

        # Calculate proportions
        expected_total = np.sum(expected_counts)
        actual_total = np.sum(actual_counts)

        if expected_total == 0 or actual_total == 0:
            return np.nan

        expected_percents = expected_counts / expected_total
        actual_percents = actual_counts / actual_total

        # Avoid division by zero
        epsilon = BINNING.DEFAULT_EPSILON
        expected_percents = np.maximum(expected_percents, epsilon)
        actual_percents = np.maximum(actual_percents, epsilon)

        psi_values = (actual_percents - expected_percents) * np.log(
            actual_percents / expected_percents
        )
        return np.sum(psi_values)

    except Exception as e:
        import warnings

        warnings.warn(f"Error calculating PSI: {str(e)}", stacklevel=2)
        return np.nan
