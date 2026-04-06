"""Population Stability Index (PSI) metrics."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from newt.config import BINNING

NAN_STRATEGIES = frozenset(["separate", "exclude"])
REFERENCE_MODES = frozenset(["latest", "value"])


@dataclass(frozen=True)
class _PsiReference:
    """Internal precomputed reference representation for PSI batch compute."""

    edges: np.ndarray
    expected_counts: np.ndarray
    include_missing_bucket: bool
    epsilon: float = BINNING.DEFAULT_EPSILON


def calculate_psi(
    expected: Union[np.ndarray, list],
    actual: Union[np.ndarray, list],
    buckets: int = BINNING.DEFAULT_BUCKETS,
    include_nan: bool = True,
    nan_strategy: Optional[str] = None,
) -> float:
    """Calculate Population Stability Index (PSI)."""
    try:
        values = calculate_psi_batch(
            expected=expected,
            actual_groups=[actual],
            buckets=buckets,
            include_nan=include_nan,
            nan_strategy=nan_strategy,
            engine="python",
        )
        return float(values[0]) if values else float("nan")
    except Exception as error:
        import warnings

        warnings.warn(f"Error calculating PSI: {str(error)}", stacklevel=2)
        return float("nan")


def calculate_psi_batch(
    expected: Union[np.ndarray, list, pd.Series],
    actual_groups: Sequence[Union[np.ndarray, list, pd.Series]],
    buckets: int = BINNING.DEFAULT_BUCKETS,
    include_nan: bool = True,
    nan_strategy: Optional[str] = None,
    engine: str = "rust",
) -> List[float]:
    """Calculate PSI values for multiple groups against one reference."""
    if buckets < 1:
        raise ValueError("buckets must be >= 1")

    strategy = _resolve_nan_strategy(include_nan=include_nan, nan_strategy=nan_strategy)
    expected_values = _to_numeric_array(expected)
    reference = _build_psi_reference(
        expected=expected_values,
        buckets=buckets,
        strategy=strategy,
    )

    groups = list(actual_groups)
    if not groups:
        return []

    if engine == "rust":
        rust_values = _calculate_psi_batch_with_rust(
            reference=reference,
            actual_groups=groups,
        )
        if rust_values is not None:
            return rust_values

    return _calculate_psi_batch_with_python(
        reference=reference,
        actual_groups=groups,
    )


def calculate_grouped_psi(
    data: pd.DataFrame,
    group_cols: Sequence[str],
    score_col: str,
    reference_mode: str = "latest",
    reference_col: Optional[str] = None,
    reference_value: Optional[object] = None,
    partition_cols: Optional[Sequence[str]] = None,
    buckets: int = BINNING.DEFAULT_BUCKETS,
    include_nan: bool = True,
    nan_strategy: Optional[str] = None,
    engine: str = "rust",
    include_stats: bool = False,
) -> pd.DataFrame:
    """Calculate PSI for grouped DataFrame slices."""
    if score_col not in data.columns:
        raise ValueError(f"Column not found: {score_col}")

    groups = list(dict.fromkeys(group_cols))
    if not groups:
        raise ValueError("group_cols must include at least one column")

    missing_groups = [column for column in groups if column not in data.columns]
    if missing_groups:
        raise ValueError(f"group_cols not found: {missing_groups}")

    partitions = list(dict.fromkeys(partition_cols or []))
    missing_partitions = [column for column in partitions if column not in data.columns]
    if missing_partitions:
        raise ValueError(f"partition_cols not found: {missing_partitions}")

    if reference_mode not in REFERENCE_MODES:
        raise ValueError(
            "reference_mode must be one of "
            f"{sorted(REFERENCE_MODES)}, got: {reference_mode}"
        )

    resolved_reference_col = reference_col or groups[-1]
    if resolved_reference_col not in data.columns:
        raise ValueError(f"reference_col not found: {resolved_reference_col}")

    if reference_mode == "value" and reference_value is None:
        raise ValueError("reference_value must be provided when reference_mode='value'")

    frame_columns = list(dict.fromkeys([*partitions, *groups, score_col]))
    frame = data.loc[:, frame_columns]

    partitioned = (
        list(frame.groupby(partitions, dropna=False, sort=False))
        if partitions
        else [(tuple(), frame)]
    )
    rows: List[dict] = []

    for partition_values, partition_frame in partitioned:
        partition_tuple = _normalize_group_values(partition_values)
        partition_map = {
            column: value for column, value in zip(partitions, partition_tuple)
        }
        if partition_frame.empty:
            continue

        resolved_reference_value = _resolve_reference_value(
            frame=partition_frame,
            reference_mode=reference_mode,
            reference_col=resolved_reference_col,
            reference_value=reference_value,
        )

        reference_mask = _series_equals(
            partition_frame[resolved_reference_col],
            resolved_reference_value,
        )
        reference_series = partition_frame.loc[reference_mask, score_col]
        reference_sample_count = int(len(reference_series))
        reference_missing_count = int(
            pd.to_numeric(reference_series, errors="coerce").isna().sum()
        )

        grouped = list(partition_frame.groupby(groups, dropna=False, sort=False))
        group_series = [group_frame[score_col] for _, group_frame in grouped]
        psi_values = calculate_psi_batch(
            expected=reference_series,
            actual_groups=group_series,
            buckets=buckets,
            include_nan=include_nan,
            nan_strategy=nan_strategy,
            engine=engine,
        )

        for (group_values, group_frame), psi_value in zip(grouped, psi_values):
            group_tuple = _normalize_group_values(group_values)
            group_map = {column: value for column, value in zip(groups, group_tuple)}
            row = {
                **partition_map,
                **group_map,
                "psi": float(psi_value),
                "is_reference": bool(
                    resolved_reference_col in group_map
                    and _value_equals(
                        group_map[resolved_reference_col],
                        resolved_reference_value,
                    )
                ),
            }
            if include_stats:
                numeric_score = pd.to_numeric(group_frame[score_col], errors="coerce")
                row["sample_count"] = int(len(group_frame))
                row["missing_count"] = int(numeric_score.isna().sum())
                row["reference_sample_count"] = reference_sample_count
                row["reference_missing_count"] = reference_missing_count
            rows.append(row)

    columns = [*partitions, *groups, "psi", "is_reference"]
    if include_stats:
        columns.extend(
            [
                "sample_count",
                "missing_count",
                "reference_sample_count",
                "reference_missing_count",
            ]
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)
    sort_cols = [
        column for column in [*partitions, *groups] if column in result.columns
    ]
    if sort_cols:
        result = result.sort_values(sort_cols, kind="mergesort")
    return result.reset_index(drop=True)


def calculate_feature_psi_against_base(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    base_col: str,
    base_value: object,
    compare_col: Optional[str] = None,
    compare_values: Optional[Sequence[object]] = None,
    buckets: int = BINNING.DEFAULT_BUCKETS,
    include_nan: bool = True,
    nan_strategy: Optional[str] = None,
    engine: str = "rust",
    include_stats: bool = True,
) -> pd.DataFrame:
    """Batch-calculate feature PSI values against a chosen base slice."""
    if base_col not in data.columns:
        raise ValueError(f"Column not found: {base_col}")

    features = list(dict.fromkeys(feature_cols))
    if not features:
        raise ValueError("feature_cols must include at least one column")

    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"feature_cols not found: {missing_features}")

    resolved_compare_col = compare_col or base_col
    if resolved_compare_col not in data.columns:
        raise ValueError(f"Column not found: {resolved_compare_col}")

    compared_values = (
        list(compare_values)
        if compare_values is not None
        else _ordered_unique(data[resolved_compare_col])
    )
    if resolved_compare_col == base_col and not any(
        _value_equals(value, base_value) for value in compared_values
    ):
        compared_values.append(base_value)

    base_mask = _series_equals(data[base_col], base_value)
    base_frame = data.loc[base_mask]

    rows: List[dict] = []
    for feature in features:
        reference_series = base_frame[feature]
        actual_groups = [
            data.loc[_series_equals(data[resolved_compare_col], value), feature]
            for value in compared_values
        ]
        psi_values = calculate_psi_batch(
            expected=reference_series,
            actual_groups=actual_groups,
            buckets=buckets,
            include_nan=include_nan,
            nan_strategy=nan_strategy,
            engine=engine,
        )
        reference_numeric = pd.to_numeric(reference_series, errors="coerce")
        reference_sample_count = int(len(reference_series))
        reference_missing_count = int(reference_numeric.isna().sum())

        for value, actual_series, psi_value in zip(
            compared_values,
            actual_groups,
            psi_values,
        ):
            row = {
                "feature": feature,
                "base_col": base_col,
                "base_value": base_value,
                "compare_col": resolved_compare_col,
                "compare_value": value,
                "psi": float(psi_value),
                "is_reference": bool(
                    resolved_compare_col == base_col
                    and _value_equals(value, base_value)
                ),
            }
            if include_stats:
                actual_numeric = pd.to_numeric(actual_series, errors="coerce")
                row["sample_count"] = int(len(actual_series))
                row["missing_count"] = int(actual_numeric.isna().sum())
                row["reference_sample_count"] = reference_sample_count
                row["reference_missing_count"] = reference_missing_count
            rows.append(row)

    columns = [
        "feature",
        "base_col",
        "base_value",
        "compare_col",
        "compare_value",
        "psi",
        "is_reference",
    ]
    if include_stats:
        columns.extend(
            [
                "sample_count",
                "missing_count",
                "reference_sample_count",
                "reference_missing_count",
            ]
        )
    return pd.DataFrame(rows, columns=columns)


def _resolve_nan_strategy(include_nan: bool, nan_strategy: Optional[str]) -> str:
    strategy = nan_strategy
    if strategy is None:
        strategy = "separate" if include_nan else "exclude"
    if strategy not in NAN_STRATEGIES:
        raise ValueError(
            f"nan_strategy must be one of {sorted(NAN_STRATEGIES)}, got: {strategy}"
        )
    return strategy


def _to_numeric_array(values: Union[np.ndarray, list, pd.Series]) -> np.ndarray:
    return pd.to_numeric(
        pd.Series(np.asarray(values).ravel()),
        errors="coerce",
    ).to_numpy(dtype=float)


def _build_psi_reference(
    expected: np.ndarray,
    buckets: int,
    strategy: str,
) -> _PsiReference:
    non_missing = expected[~np.isnan(expected)]
    if non_missing.size > 0:
        breakpoints = np.percentile(non_missing, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints.astype(float))
        if breakpoints.size < 2:
            edges = np.array([-np.inf, np.inf], dtype=float)
        else:
            breakpoints[0] = -np.inf
            breakpoints[-1] = np.inf
            edges = breakpoints
    else:
        edges = np.array([-np.inf, np.inf], dtype=float)

    include_missing_bucket = strategy == "separate"
    expected_counts = _count_values_by_edges(
        values=expected,
        edges=edges,
        include_missing_bucket=include_missing_bucket,
    )
    return _PsiReference(
        edges=np.asarray(edges, dtype=float),
        expected_counts=np.asarray(expected_counts, dtype=float),
        include_missing_bucket=include_missing_bucket,
    )


def _count_values_by_edges(
    values: np.ndarray,
    edges: np.ndarray,
    include_missing_bucket: bool,
) -> np.ndarray:
    nan_mask = np.isnan(values)
    non_missing = values[~nan_mask]
    counts, _ = np.histogram(non_missing, bins=np.asarray(edges, dtype=float))
    counts = counts.astype(float)
    if include_missing_bucket:
        counts = np.append(counts, float(nan_mask.sum()))
    return counts


def _psi_from_counts(
    expected_counts: np.ndarray,
    actual_counts: np.ndarray,
    epsilon: float = BINNING.DEFAULT_EPSILON,
) -> float:
    expected_total = float(np.sum(expected_counts))
    actual_total = float(np.sum(actual_counts))
    if expected_total == 0.0 or actual_total == 0.0:
        return float("nan")

    expected_percents = np.maximum(expected_counts / expected_total, epsilon)
    actual_percents = np.maximum(actual_counts / actual_total, epsilon)
    psi_values = (actual_percents - expected_percents) * np.log(
        actual_percents / expected_percents
    )
    return float(np.sum(psi_values))


def _calculate_psi_batch_with_python(
    reference: _PsiReference,
    actual_groups: Sequence[Union[np.ndarray, list, pd.Series]],
) -> List[float]:
    results: List[float] = []
    for group in actual_groups:
        actual_values = _to_numeric_array(group)
        actual_counts = _count_values_by_edges(
            values=actual_values,
            edges=reference.edges,
            include_missing_bucket=reference.include_missing_bucket,
        )
        results.append(
            _psi_from_counts(
                expected_counts=reference.expected_counts,
                actual_counts=actual_counts,
                epsilon=reference.epsilon,
            )
        )
    return results


def _calculate_psi_batch_with_rust(
    reference: _PsiReference,
    actual_groups: Sequence[Union[np.ndarray, list, pd.Series]],
) -> Optional[List[float]]:
    module = _load_rust_module()
    if module is None:
        return None

    # Prefer numpy path (zero-copy, avoids List[Optional[float]] overhead)
    numpy_fn = getattr(module, "calculate_psi_batch_from_edges_numpy", None)
    if callable(numpy_fn):
        try:
            group_arrays = [
                np.ascontiguousarray(_to_numeric_array(group))
                for group in actual_groups
            ]
            values = numpy_fn(
                np.ascontiguousarray(reference.edges, dtype=np.float64),
                np.ascontiguousarray(reference.expected_counts, dtype=np.float64),
                group_arrays,
                bool(reference.include_missing_bucket),
                float(reference.epsilon),
            )
            return [float(item) for item in values]
        except Exception:
            pass  # fall through to legacy path

    # Legacy path: List[Optional[float]]
    legacy_fn = getattr(module, "calculate_psi_batch_from_edges", None)
    if not callable(legacy_fn):
        return None
    try:
        group_values = [_to_optional_float_list(group) for group in actual_groups]
        args_common = (
            reference.edges.astype(float).tolist(),
            reference.expected_counts.astype(float).tolist(),
            group_values,
        )
        try:
            values = legacy_fn(
                *args_common,
                bool(reference.include_missing_bucket),
                float(reference.epsilon),
            )
        except TypeError:
            if not reference.include_missing_bucket:
                return None
            values = legacy_fn(
                *args_common,
                float(reference.epsilon),
            )
        return [float(item) for item in values]
    except Exception:
        return None


def _to_optional_float_list(
    values: Union[np.ndarray, list, pd.Series]
) -> List[Optional[float]]:
    numeric = _to_numeric_array(values)
    return [None if np.isnan(item) else float(item) for item in numeric.tolist()]


def _load_rust_module():
    for module_name in ("newt._newt_iv_rust", "_newt_iv_rust"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


def _resolve_reference_value(
    frame: pd.DataFrame,
    reference_mode: str,
    reference_col: str,
    reference_value: Optional[object],
) -> object:
    if reference_mode == "value":
        return reference_value

    candidates = _ordered_unique(frame[reference_col])
    if not candidates:
        return None
    candidates = sorted(candidates, key=_reference_sort_key)
    return candidates[-1]


def _ordered_unique(values: Iterable[object]) -> List[object]:
    return pd.Series(values).drop_duplicates().tolist()


def _normalize_group_values(group_values: object) -> Tuple[object, ...]:
    if isinstance(group_values, tuple):
        return group_values
    return (group_values,)


def _series_equals(values: pd.Series, target: object) -> pd.Series:
    if pd.isna(target):
        return values.isna()
    return values == target


def _value_equals(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right)


def _reference_sort_key(value: object) -> str:
    if pd.isna(value) or value == "":
        return "999999"
    text = str(value).strip()
    if text.isdigit() and len(text) == 6:
        return text
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y%m")
    return f"999998{text}"


__all__ = [
    "calculate_psi",
    "calculate_psi_batch",
    "calculate_grouped_psi",
    "calculate_feature_psi_against_base",
    "NAN_STRATEGIES",
]
