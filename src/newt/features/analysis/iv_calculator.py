from typing import Dict, Union

import pandas as pd

from newt._engine import ensure_native_functions, validate_engine
from newt._native import load_native_module, require_native_module
from newt.config import BINNING

from .iv_math import build_iv_summary, prepare_feature_for_iv


def calculate_iv(
    df: pd.DataFrame,
    target: str,
    feature: str,
    buckets: int = BINNING.DEFAULT_BUCKETS,
    epsilon: float = BINNING.DEFAULT_EPSILON,
    engine: str = "auto",
) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Calculate Information Value (IV) for a feature.

    Args:
        df: Input DataFrame.
        target: Target column name (binary 0/1).
        feature: Feature column name.
        buckets: Number of buckets for numerical features (if not already binned).
        epsilon: Retained for backward compatibility.
        engine: IV engine, one of "auto" (default), "rust", or "python".

    Returns:
        Dict containing 'iv' (float) and 'woe_table' (pd.DataFrame).
    """
    try:
        validate_engine(engine)
    except ValueError as exc:
        raise ValueError("engine must be 'auto', 'rust' or 'python'") from exc

    feature_data = prepare_feature_for_iv(df[feature], buckets=buckets)
    woe_table, python_iv = build_iv_summary(feature_data, df[target])

    # Keep epsilon in the signature for API compatibility.
    _ = epsilon

    if engine == "python":
        iv_value = float(python_iv)
    elif engine == "rust":
        rust_module = _load_rust_extension()
        ensure_native_functions(
            rust_module,
            ["calculate_categorical_iv"],
            component="Rust IV engine",
        )
        iv_value = float(
            _calculate_rust_iv(
                rust_module=rust_module,
                feature_data=feature_data,
                target_series=df[target],
            )
        )
    else:
        rust_module = load_native_module()
        if rust_module is None:
            iv_value = float(python_iv)
        else:
            try:
                ensure_native_functions(
                    rust_module,
                    ["calculate_categorical_iv"],
                    component="Rust IV engine",
                )
            except RuntimeError:
                iv_value = float(python_iv)
            else:
                iv_value = float(
                    _calculate_rust_iv(
                        rust_module=rust_module,
                        feature_data=feature_data,
                        target_series=df[target],
                    )
                )

    return {"iv": iv_value, "woe_table": woe_table}


def _load_rust_extension():
    """Import the compiled Rust extension from the package."""
    return require_native_module()


def _calculate_rust_iv(
    rust_module,
    feature_data: pd.Series,
    target_series: pd.Series,
) -> float:
    """Calculate IV with Rust categorical backend on prepared feature labels."""
    target_numeric = pd.to_numeric(target_series, errors="coerce")
    valid_mask = target_numeric.isin([0, 1])
    prepared_feature = (
        feature_data.loc[valid_mask]
        .astype("object")
        .where(lambda s: s.notna(), None)
        .tolist()
    )
    target_values = target_numeric.loc[valid_mask].astype(int).tolist()
    return float(
        rust_module.calculate_categorical_iv(
            prepared_feature,
            target_values,
        )
    )
