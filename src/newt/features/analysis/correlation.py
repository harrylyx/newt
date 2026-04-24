from typing import Dict, List, Union

import numpy as np
import pandas as pd

from newt._engine import ensure_native_functions, validate_engine
from newt._native import load_native_module, require_native_module

VALID_CORRELATION_METHODS = frozenset(["pearson", "kendall", "spearman"])


def calculate_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    engine: str = "auto",
) -> pd.DataFrame:
    """Calculate a feature correlation matrix."""
    if method not in VALID_CORRELATION_METHODS:
        raise ValueError(
            f"method must be one of {sorted(VALID_CORRELATION_METHODS)}, got: {method}"
        )
    validate_engine(engine)

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()

    if engine == "python":
        return numeric_df.corr(method=method)

    if engine == "rust":
        return _calculate_correlation_matrix_rust(
            numeric_df, method=method, strict=True
        )

    try:
        return _calculate_correlation_matrix_rust(
            numeric_df, method=method, strict=False
        )
    except Exception:
        return numeric_df.corr(method=method)


def get_high_correlation_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.8,
    engine: str = "auto",
) -> List[Dict[str, Union[str, float]]]:
    """Identify pairs of variables with correlation above a threshold."""
    validate_engine(engine)
    if corr_matrix.empty:
        return []

    if engine == "python":
        return _get_high_correlation_pairs_python(corr_matrix, threshold)

    if engine == "rust":
        return _get_high_correlation_pairs_rust(corr_matrix, threshold, strict=True)

    try:
        return _get_high_correlation_pairs_rust(corr_matrix, threshold, strict=False)
    except Exception:
        return _get_high_correlation_pairs_python(corr_matrix, threshold)


def _calculate_correlation_matrix_rust(
    numeric_df: pd.DataFrame,
    method: str,
    strict: bool,
) -> pd.DataFrame:
    module = require_native_module() if strict else load_native_module()
    if module is None:
        raise ImportError("Rust native extension is unavailable.")

    ensure_native_functions(
        module,
        ["calculate_correlation_matrix_numpy"],
        component="Rust correlation engine",
    )
    rust_fn = module.calculate_correlation_matrix_numpy

    columns = [
        np.ascontiguousarray(
            pd.to_numeric(numeric_df[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
        )
        for column in numeric_df.columns
    ]
    matrix = rust_fn(columns, str(method))
    return pd.DataFrame(matrix, index=numeric_df.columns, columns=numeric_df.columns)


def _get_high_correlation_pairs_rust(
    corr_matrix: pd.DataFrame,
    threshold: float,
    strict: bool,
) -> List[Dict[str, Union[str, float]]]:
    module = require_native_module() if strict else load_native_module()
    if module is None:
        raise ImportError("Rust native extension is unavailable.")

    ensure_native_functions(
        module,
        ["extract_high_correlation_pairs_numpy"],
        component="Rust correlation engine",
    )
    rust_fn = module.extract_high_correlation_pairs_numpy

    matrix = np.ascontiguousarray(corr_matrix.to_numpy(dtype=np.float64))
    rows = [np.ascontiguousarray(matrix[idx, :]) for idx in range(matrix.shape[0])]
    raw_pairs = rust_fn(rows, float(threshold))
    columns = corr_matrix.columns.to_list()
    return [
        {
            "var1": str(columns[left]),
            "var2": str(columns[right]),
            "correlation": float(corr),
        }
        for left, right, corr in raw_pairs
    ]


def _get_high_correlation_pairs_python(
    corr_matrix: pd.DataFrame,
    threshold: float,
) -> List[Dict[str, Union[str, float]]]:
    high_corr_pairs = []

    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper = corr_matrix.where(mask)
    pairs = upper.stack()
    high_corr = pairs[pairs.abs() >= abs(threshold)]

    for (var1, var2), val in high_corr.items():
        high_corr_pairs.append({"var1": var1, "var2": var2, "correlation": float(val)})

    high_corr_pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return high_corr_pairs
