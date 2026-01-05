import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class WOEEncoder:
    """
    Weight of Evidence (WoE) Encoder.

    Encodes features using WoE values based on a binary target.
    Can handle both numerical (via binning) and categorical features.
    """

    def __init__(self, buckets: int = 10, epsilon: float = 1e-8):
        self.buckets = buckets
        self.epsilon = epsilon
        self.woe_map_ = {}
        self.iv_ = 0.0
        self.summary_ = pd.DataFrame()
        self.bins_ = None
        self.is_numeric_ = False

    def fit(self, X: pd.Series, y: pd.Series) -> "WOEEncoder":
        """
        Fit the WoE encoder to the data.

        Args:
            X: Feature data (pandas Series).
            y: Target data (pandas Series, binary 0/1).
        """
        X = X.copy()
        y = y.copy()

        self.is_numeric_ = pd.api.types.is_numeric_dtype(X)

        # Binning logic
        if self.is_numeric_ and X.nunique() > 20:
            # Use qcut for equal frequency, fallback to cut
            try:
                # Store the retbins=True to get edges for transform
                _, self.bins_ = pd.qcut(
                    X, q=self.buckets, duplicates="drop", retbins=True
                )
                X_binned = pd.cut(X, bins=self.bins_, include_lowest=True)
            except Exception:
                # Fallback
                _, self.bins_ = pd.cut(X, bins=self.buckets, retbins=True)
                X_binned = pd.cut(X, bins=self.bins_, include_lowest=True)
        else:
            X_binned = X.astype(str)
            self.bins_ = None

        # Create temporary dataframe for aggregation
        df = pd.DataFrame({"bin": X_binned, "target": y})

        # Calculate Good/Bad stats
        grouped = df.groupby("bin", observed=True)["target"].agg(
            ["count", "sum"]
        )
        grouped = grouped.rename(columns={"count": "total", "sum": "bad"})
        grouped["good"] = grouped["total"] - grouped["bad"]

        total_bad = grouped["bad"].sum()
        total_good = grouped["good"].sum()

        if total_bad == 0 or total_good == 0:
            # Degenerate case, set everything to 0
            self.iv_ = 0.0
            self.woe_map_ = {k: 0.0 for k in grouped.index}
            return self

        # Distributions with smoothing
        dist_bad = (grouped["bad"] / total_bad).clip(lower=self.epsilon)
        dist_good = (grouped["good"] / total_good).clip(lower=self.epsilon)

        # WoE and IV
        woe = np.log(dist_good / dist_bad)
        iv_contrib = (dist_good - dist_bad) * woe

        # Store results
        self.woe_map_ = woe.to_dict()
        self.iv_ = iv_contrib.sum()

        # Create summary table
        self.summary_ = grouped.copy()
        self.summary_["dist_good"] = dist_good
        self.summary_["dist_bad"] = dist_bad
        self.summary_["woe"] = woe
        self.summary_["iv_contribution"] = iv_contrib

        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """
        Transform X using the learned WoE mapping.

        Args:
            X: Feature data to transform.

        Returns:
            Transformed data (WoE values).
        """
        if not self.woe_map_:
            raise ValueError("Encoder is not fitted. Call fit() first.")

        X = X.copy()

        if self.is_numeric_ and self.bins_ is not None:
            # Use stored bins
            X_binned = pd.cut(X, bins=self.bins_, include_lowest=True)
        else:
            # Categorical - direct map
            if (
                self.is_numeric_
            ):  # Was numeric but low cardinality treated as cat
                X_binned = X.astype(
                    str
                )  # wait, fit converted low card num to str? Yes line 45.
                # If X here is numeric, we must convert to str to match keys
                # Actually line 45: X_binned = X.astype(str)
                # But X passed to fit was numeric low cardinality.
                # So we should convert input to str if bins is None but numeric is True?
                # Actually if it was treated as categorical, keys are likely strings.
                pass
            # If original was not numeric, keys are strings.
            # If original was numeric low card, keys are strings from line 45.
            # So we ensure X is converted to str if we didn't bin it?
            # Or better, just map.
            if self.bins_ is None:
                X_binned = X.astype(str)
            else:
                # Should have been handled by numerical block
                pass

        # Map values
        # Note: If X_binned contains categories not in woe_map_, map returns NaN.
        # We fill with 0 (neutral WoE) as standard fallback.
        mapped = X_binned.map(self.woe_map_)

        # Ensure float
        try:
            mapped = mapped.astype(float)
        except ValueError:
            pass

        return mapped.fillna(0.0)

    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


# Backward compatibility functions (wrappers)
def calculate_woe_mapping(
    df: pd.DataFrame,
    target: str,
    feature: str,
    bins: int = 10,
    epsilon: float = 1e-8,
) -> Dict[Any, float]:
    """Wrapper using WOEEncoder for backward compatibility."""
    encoder = WOEEncoder(
        buckets=bins if isinstance(bins, int) else 10, epsilon=epsilon
    )
    encoder.fit(df[feature], df[target])
    return encoder.woe_map_


def apply_woe_transform(
    df: pd.DataFrame,
    feature: str,
    woe_map: Dict[Any, float],
    new_col_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Wrapper for applying WoE transform.
    NOTE: This legacy function assumes the input matches the map keys
    (e.g. already binned if necessary), OR it does a direct map.
    It does NOT use the WOEEncoder's binning logic because it receives a raw dict.
    Use WOEEncoder class for robust transformation of raw numeric data.
    """
    if new_col_name is None:
        new_col_name = f"{feature}_woe"

    df_out = df.copy()
    mapped = df_out[feature].map(woe_map)
    try:
        mapped = mapped.astype(float)
    except ValueError:
        pass
    df_out[new_col_name] = mapped.fillna(0.0)
    return df_out
