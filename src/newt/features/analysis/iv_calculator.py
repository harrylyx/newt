from typing import Dict, Union

import pandas as pd

from newt._native import require_native_module
from newt.config import BINNING

from .iv_math import build_iv_summary, prepare_feature_for_iv


def calculate_iv(
    df: pd.DataFrame,
    target: str,
    feature: str,
    buckets: int = BINNING.DEFAULT_BUCKETS,
    epsilon: float = BINNING.DEFAULT_EPSILON,
    engine: str = "rust",
) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Calculate Information Value (IV) for a feature.

    Args:
        df: Input DataFrame.
        target: Target column name (binary 0/1).
        feature: Feature column name.
        buckets: Number of buckets for numerical features (if not already binned).
        epsilon: Retained for backward compatibility.
        engine: IV engine, either "rust" (default) or "python".

    Returns:
        Dict containing 'iv' (float) and 'woe_table' (pd.DataFrame).
    """
    if engine not in {"rust", "python"}:
        raise ValueError("engine must be 'rust' or 'python'")

    feature_data = prepare_feature_for_iv(df[feature], buckets=buckets)
    woe_table, python_iv = build_iv_summary(feature_data, df[target])

    # Keep epsilon in the signature for API compatibility.
    _ = epsilon

    if engine == "python":
        iv_value = float(python_iv)
    else:
        rust_module = _load_rust_extension()
        target_numeric = pd.to_numeric(df[target], errors="coerce")
        valid_mask = target_numeric.isin([0, 1])
        prepared_feature = (
            feature_data.loc[valid_mask]
            .astype("object")
            .where(lambda s: s.notna(), None)
            .tolist()
        )
        target_values = target_numeric.loc[valid_mask].astype(int).tolist()
        iv_value = float(
            rust_module.calculate_categorical_iv(
                prepared_feature,
                target_values,
            )
        )

    return {"iv": iv_value, "woe_table": woe_table}


def _load_rust_extension():
    """Import the compiled Rust extension from the package."""
    return require_native_module()
