"""
VIF (Variance Inflation Factor) calculation.

Used to detect multicollinearity in regression models.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


def calculate_vif(
    X: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for each feature.

    VIF measures multicollinearity. A VIF > 5-10 suggests high multicollinearity.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data (should not include target variable).
    features : List[str], optional
        List of features to calculate VIF for. If None, uses all numeric columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, vif

    Examples
    --------
    >>> vif_df = calculate_vif(X)
    >>> high_vif = vif_df[vif_df['vif'] > 10]
    """
    X = X.copy()

    # Select features
    if features is None:
        features = X.select_dtypes(include=[np.number]).columns.tolist()
    else:
        features = [f for f in features if f in X.columns]

    if len(features) < 2:
        # VIF is not meaningful with < 2 features
        return pd.DataFrame({"feature": features, "vif": [1.0] * len(features)})

    # Drop rows with NaN
    X_clean = X[features].dropna()

    if len(X_clean) == 0:
        return pd.DataFrame({"feature": features, "vif": [np.nan] * len(features)})

    vif_data = []

    for feature in features:
        # Get the feature and other features
        y = X_clean[feature].values
        X_others = X_clean.drop(columns=[feature]).values

        # Add constant column for intercept
        X_with_const = np.column_stack([np.ones(len(y)), X_others])

        try:
            # Calculate R-squared using OLS
            # R² = 1 - SSres/SStot
            # VIF = 1 / (1 - R²)

            # Use numpy lstsq for efficiency
            coeffs, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)

            # Calculate predicted values
            y_pred = X_with_const @ coeffs

            # Calculate R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot == 0:
                r_squared = 0.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
                r_squared = max(0, min(1, r_squared))  # Clip to [0, 1]

            # Calculate VIF
            if r_squared >= 1.0:
                vif = np.inf
            else:
                vif = 1 / (1 - r_squared)

            vif_data.append({"feature": feature, "vif": vif})

        except Exception:
            vif_data.append({"feature": feature, "vif": np.nan})

    return pd.DataFrame(vif_data)
