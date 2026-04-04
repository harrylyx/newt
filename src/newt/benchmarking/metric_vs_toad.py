"""Benchmark Newt metric implementations against toad."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TARGET_COL = "target"
SCORE_COLUMNS = ("score_p", "score_old_p")
SPLIT_FLAGS = {
    "train": "flag_train",
    "test": "flag_test",
    "oot": "flag_oot",
}
PSI_SPLITS = (("train", "test"), ("train", "oot"))
BENCHMARK_OUTPUT_JSON = "metric_vs_toad.json"
BENCHMARK_OUTPUT_MD = "metric_vs_toad.md"
METADATA_COLUMNS = frozenset(
    {
        "idx",
        TARGET_COL,
        "listinginfo",
        "mth",
        "tag",
        "flag_train",
        "flag_oot",
        "flag_oos",
        "flag_test",
        "score_p",
        "score",
        "score_old_p",
        "score_old",
    }
)


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


def default_data_path() -> Path:
    """Return the bundled benchmark dataset path."""
    return repo_root() / "examples" / "data" / "test_data" / "all_data.pq"


def default_output_dir() -> Path:
    """Return the default output directory."""
    return repo_root() / "out" / "benchmarks"


def load_benchmark_dataset(
    data_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load the bundled dataset and split it into benchmark subsets."""
    frame = pd.read_parquet(data_path)
    binary = frame.loc[frame[TARGET_COL].isin([0, 1])].copy()
    binary[TARGET_COL] = binary[TARGET_COL].astype(int)

    splits: Dict[str, pd.DataFrame] = {}
    for split_name, flag_col in SPLIT_FLAGS.items():
        mask = binary[flag_col].fillna(False).astype(bool)
        splits[split_name] = binary.loc[mask].copy()

    return binary, splits


def get_feature_columns(frame: pd.DataFrame) -> List[str]:
    """Return feature columns that should participate in IV/PSI checks."""
    return [column for column in frame.columns if column not in METADATA_COLUMNS]


def get_numeric_feature_columns(frame: pd.DataFrame) -> List[str]:
    """Return numeric feature columns for PSI and batch IV benchmarks."""
    feature_columns = get_feature_columns(frame)
    return [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]


def prepare_feature_for_iv(series: pd.Series, buckets: int) -> pd.Series:
    """Prepare a feature so Newt and toad use the same IV bins."""
    prepared: pd.Series
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > buckets:
        try:
            prepared = pd.qcut(series, q=buckets, duplicates="drop").astype("object")
        except ValueError:
            prepared = pd.cut(series, bins=buckets).astype("object")
    else:
        prepared = series.astype("object")

    return prepared.where(pd.notna(prepared), "Missing").astype(str)


def prepare_feature_frame_for_iv(
    frame: pd.DataFrame,
    features: Sequence[str],
    buckets: int,
) -> pd.DataFrame:
    """Prepare every feature column for aligned IV comparisons."""
    prepared = {
        feature: prepare_feature_for_iv(frame[feature], buckets=buckets)
        for feature in features
    }
    return pd.DataFrame(prepared, index=frame.index)


def build_newt_psi_breakpoints(values: Iterable[Any], buckets: int) -> np.ndarray:
    """Build the same reference breakpoints that Newt uses for PSI."""
    numeric = pd.to_numeric(
        pd.Series(np.asarray(list(values)).ravel()),
        errors="coerce",
    ).to_numpy(dtype=float)
    non_missing = numeric[~np.isnan(numeric)]

    if len(non_missing) > 0:
        breakpoints = np.percentile(non_missing, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            breakpoints = np.array([-np.inf, np.inf], dtype=float)
        else:
            breakpoints[0] = -np.inf
            breakpoints[-1] = np.inf
    else:
        breakpoints = np.array([-np.inf, np.inf], dtype=float)

    return breakpoints


def apply_reference_bins(values: Iterable[Any], breakpoints: np.ndarray) -> pd.Series:
    """Apply reference breakpoints and keep missing values as a dedicated bucket."""
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    binned = pd.cut(
        numeric,
        bins=breakpoints,
        include_lowest=True,
        duplicates="drop",
    ).astype("object")
    return binned.where(~numeric.isna(), "Missing").astype(str)


def make_serializable(value: Any) -> Any:
    """Convert NumPy and pandas values into JSON-friendly primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: make_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(item) for item in value]
    return value


def benchmark_callable(
    func,
    repeat: int,
    warmup: int,
) -> Dict[str, Any]:
    """Benchmark a callable and return summary timing stats."""
    for _ in range(warmup):
        func()

    timings: List[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        func()
        timings.append((time.perf_counter() - started) * 1000.0)

    return {
        "repeat": repeat,
        "warmup": warmup,
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "times_ms": timings,
    }


def _key_from_row(row: Dict[str, Any], keys: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(row[key] for key in keys)


def compare_value_rows(
    newt_rows: Sequence[Dict[str, Any]],
    toad_rows: Sequence[Dict[str, Any]],
    keys: Sequence[str],
) -> List[Dict[str, Any]]:
    """Compare value rows from Newt and toad."""
    toad_map = {_key_from_row(row, keys): row for row in toad_rows}
    merged: List[Dict[str, Any]] = []

    for newt_row in newt_rows:
        merged_row = {key: newt_row[key] for key in keys}
        toad_row = toad_map.get(_key_from_row(newt_row, keys))

        newt_value = make_serializable(newt_row["value"])
        toad_value = make_serializable(toad_row["value"]) if toad_row else None
        merged_row["newt"] = newt_value
        merged_row["toad"] = toad_value
        merged_row["abs_diff"] = (
            abs(newt_value - toad_value)
            if newt_value is not None and toad_value is not None
            else None
        )
        merged.append(merged_row)

    return merged


def compare_timing_rows(
    newt_rows: Sequence[Dict[str, Any]],
    toad_rows: Sequence[Dict[str, Any]],
    keys: Sequence[str],
) -> List[Dict[str, Any]]:
    """Compare timing rows from Newt and toad."""
    toad_map = {_key_from_row(row, keys): row for row in toad_rows}
    merged: List[Dict[str, Any]] = []

    for newt_row in newt_rows:
        merged_row = {key: newt_row[key] for key in keys}
        toad_row = toad_map.get(_key_from_row(newt_row, keys))
        newt_ms = make_serializable(newt_row["median_ms"])
        toad_ms = make_serializable(toad_row["median_ms"]) if toad_row else None
        merged_row["newt_ms"] = newt_ms
        merged_row["toad_ms"] = toad_ms
        merged_row["newt_speedup"] = (
            toad_ms / newt_ms
            if newt_ms not in (None, 0) and toad_ms is not None
            else None
        )
        merged.append(merged_row)

    return merged


def summarize_differences(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize absolute differences for a set of comparison rows."""
    values = [row["abs_diff"] for row in rows if row.get("abs_diff") is not None]
    if not values:
        return {
            "count": len(rows),
            "compared": 0,
            "mean_abs_diff": None,
            "max_abs_diff": None,
        }

    return {
        "count": len(rows),
        "compared": len(values),
        "mean_abs_diff": float(np.mean(values)),
        "max_abs_diff": float(np.max(values)),
    }


def sort_top_rows(
    rows: Sequence[Dict[str, Any]],
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Sort rows by absolute difference and return the largest entries."""
    ordered = sorted(
        rows,
        key=lambda item: item["abs_diff"] if item.get("abs_diff") is not None else -1.0,
        reverse=True,
    )
    return list(ordered[:limit])


def compute_newt_results(
    binary: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    feature_columns: Sequence[str],
    numeric_features: Sequence[str],
    repeat: int,
    warmup: int,
    iv_buckets: int,
    psi_buckets: int,
) -> Dict[str, Any]:
    """Compute benchmark results using Newt's implementations."""
    from newt.features.analysis.batch_iv import calculate_batch_iv
    from newt.features.analysis.iv_calculator import calculate_iv
    from newt.metrics.auc import calculate_auc
    from newt.metrics.ks import calculate_ks
    from newt.metrics.psi import calculate_psi

    prepared_iv = prepare_feature_frame_for_iv(
        binary, feature_columns, buckets=iv_buckets
    )
    iv_input = prepared_iv.copy()
    iv_input[TARGET_COL] = binary[TARGET_COL].values

    auc_rows: List[Dict[str, Any]] = []
    ks_rows: List[Dict[str, Any]] = []
    auc_timings: List[Dict[str, Any]] = []
    ks_timings: List[Dict[str, Any]] = []

    for split_name, split_frame in splits.items():
        y_true = split_frame[TARGET_COL].to_numpy()
        for score_col in SCORE_COLUMNS:
            score = pd.to_numeric(split_frame[score_col], errors="coerce").to_numpy()
            auc_rows.append(
                {
                    "split": split_name,
                    "score": score_col,
                    "value": calculate_auc(y_true, score),
                }
            )
            ks_rows.append(
                {
                    "split": split_name,
                    "score": score_col,
                    "value": calculate_ks(y_true, score),
                }
            )

            auc_timing = benchmark_callable(
                lambda labels=y_true, values=score: calculate_auc(labels, values),
                repeat=repeat,
                warmup=warmup,
            )
            ks_timing = benchmark_callable(
                lambda labels=y_true, values=score: calculate_ks(labels, values),
                repeat=repeat,
                warmup=warmup,
            )
            auc_timings.append(
                {"metric": "auc", "split": split_name, "score": score_col, **auc_timing}
            )
            ks_timings.append(
                {"metric": "ks", "split": split_name, "score": score_col, **ks_timing}
            )

    psi_score_rows: List[Dict[str, Any]] = []
    psi_feature_rows: List[Dict[str, Any]] = []
    psi_timings: List[Dict[str, Any]] = []

    for base_split, compare_split in PSI_SPLITS:
        base_frame = splits[base_split]
        compare_frame = splits[compare_split]

        for score_col in SCORE_COLUMNS:
            base_values = base_frame[score_col].to_numpy()
            compare_values = compare_frame[score_col].to_numpy()
            psi_score_rows.append(
                {
                    "base_split": base_split,
                    "compare_split": compare_split,
                    "feature": score_col,
                    "value": calculate_psi(
                        base_values,
                        compare_values,
                        buckets=psi_buckets,
                    ),
                }
            )
            timing = benchmark_callable(
                lambda expected=base_values, actual=compare_values: calculate_psi(
                    expected,
                    actual,
                    buckets=psi_buckets,
                ),
                repeat=repeat,
                warmup=warmup,
            )
            psi_timings.append(
                {
                    "metric": "psi",
                    "scope": "score",
                    "base_split": base_split,
                    "compare_split": compare_split,
                    "feature": score_col,
                    **timing,
                }
            )

        for feature in numeric_features:
            psi_feature_rows.append(
                {
                    "base_split": base_split,
                    "compare_split": compare_split,
                    "feature": feature,
                    "value": calculate_psi(
                        base_frame[feature].to_numpy(),
                        compare_frame[feature].to_numpy(),
                        buckets=psi_buckets,
                    ),
                }
            )

        timing = benchmark_callable(
            lambda expected=base_frame, actual=compare_frame: [
                calculate_psi(
                    expected[feature].to_numpy(),
                    actual[feature].to_numpy(),
                    buckets=psi_buckets,
                )
                for feature in numeric_features
            ],
            repeat=repeat,
            warmup=warmup,
        )
        psi_timings.append(
            {
                "metric": "psi",
                "scope": "all_numeric_features",
                "base_split": base_split,
                "compare_split": compare_split,
                "feature": "__all__",
                **timing,
            }
        )

    iv_rows: List[Dict[str, Any]] = []
    for feature in feature_columns:
        value = calculate_iv(
            iv_input[[feature, TARGET_COL]],
            target=TARGET_COL,
            feature=feature,
            buckets=iv_buckets,
        )["iv"]
        iv_rows.append(
            {
                "feature": feature,
                "kind": "numeric" if feature in numeric_features else "categorical",
                "value": value,
            }
        )

    iv_timings: List[Dict[str, Any]] = []
    single_timing = benchmark_callable(
        lambda: [
            calculate_iv(
                iv_input[[feature, TARGET_COL]],
                target=TARGET_COL,
                feature=feature,
                buckets=iv_buckets,
            )["iv"]
            for feature in feature_columns
        ],
        repeat=repeat,
        warmup=warmup,
    )
    iv_timings.append({"implementation": "newt_single_prepared", **single_timing})

    batch_python_timing = benchmark_callable(
        lambda: calculate_batch_iv(
            binary[list(numeric_features)],
            binary[TARGET_COL],
            features=numeric_features,
            bins=iv_buckets,
            engine="python",
        ),
        repeat=repeat,
        warmup=warmup,
    )
    iv_timings.append({"implementation": "newt_batch_python", **batch_python_timing})

    try:
        batch_rust_timing = benchmark_callable(
            lambda: calculate_batch_iv(
                binary[list(numeric_features)],
                binary[TARGET_COL],
                features=numeric_features,
                bins=iv_buckets,
                engine="rust",
            ),
            repeat=repeat,
            warmup=warmup,
        )
        iv_timings.append(
            {"implementation": "newt_batch_rust", **batch_rust_timing, "status": "ok"}
        )
    except Exception as exc:
        iv_timings.append(
            {
                "implementation": "newt_batch_rust",
                "status": "error",
                "error": str(exc),
            }
        )

    return {
        "metadata": {
            "runtime": {
                "python": sys.version.split()[0],
                "executable": sys.executable,
                "platform": platform.platform(),
            }
        },
        "correctness": {
            "auc": auc_rows,
            "ks": ks_rows,
            "psi_scores": psi_score_rows,
            "psi_features": psi_feature_rows,
            "iv": iv_rows,
        },
        "efficiency": {
            "auc": auc_timings,
            "ks": ks_timings,
            "psi": psi_timings,
            "iv": iv_timings,
        },
    }


def compute_toad_results(
    binary: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    feature_columns: Sequence[str],
    numeric_features: Sequence[str],
    repeat: int,
    warmup: int,
    iv_buckets: int,
    psi_buckets: int,
) -> Dict[str, Any]:
    """Compute benchmark results using toad."""
    import toad

    prepared_iv = prepare_feature_frame_for_iv(
        binary, feature_columns, buckets=iv_buckets
    )
    iv_input = prepared_iv.copy()
    iv_input[TARGET_COL] = binary[TARGET_COL].values

    def aligned_toad_psi(expected: Sequence[Any], actual: Sequence[Any]) -> float:
        breakpoints = build_newt_psi_breakpoints(expected, buckets=psi_buckets)
        expected_binned = apply_reference_bins(expected, breakpoints)
        actual_binned = apply_reference_bins(actual, breakpoints)
        return float(toad.metrics.PSI(actual_binned, expected_binned))

    auc_rows: List[Dict[str, Any]] = []
    ks_rows: List[Dict[str, Any]] = []
    auc_timings: List[Dict[str, Any]] = []
    ks_timings: List[Dict[str, Any]] = []

    for split_name, split_frame in splits.items():
        y_true = split_frame[TARGET_COL].to_numpy()
        for score_col in SCORE_COLUMNS:
            score = pd.to_numeric(split_frame[score_col], errors="coerce").to_numpy()
            auc_rows.append(
                {
                    "split": split_name,
                    "score": score_col,
                    "value": float(toad.metrics.AUC(score, y_true)),
                }
            )
            ks_rows.append(
                {
                    "split": split_name,
                    "score": score_col,
                    "value": float(toad.metrics.KS(score, y_true)),
                }
            )
            auc_timing = benchmark_callable(
                lambda values=score, labels=y_true: toad.metrics.AUC(values, labels),
                repeat=repeat,
                warmup=warmup,
            )
            ks_timing = benchmark_callable(
                lambda values=score, labels=y_true: toad.metrics.KS(values, labels),
                repeat=repeat,
                warmup=warmup,
            )
            auc_timings.append(
                {"metric": "auc", "split": split_name, "score": score_col, **auc_timing}
            )
            ks_timings.append(
                {"metric": "ks", "split": split_name, "score": score_col, **ks_timing}
            )

    psi_score_rows: List[Dict[str, Any]] = []
    psi_feature_rows: List[Dict[str, Any]] = []
    psi_timings: List[Dict[str, Any]] = []

    for base_split, compare_split in PSI_SPLITS:
        base_frame = splits[base_split]
        compare_frame = splits[compare_split]

        for score_col in SCORE_COLUMNS:
            base_values = base_frame[score_col].to_numpy()
            compare_values = compare_frame[score_col].to_numpy()
            psi_score_rows.append(
                {
                    "base_split": base_split,
                    "compare_split": compare_split,
                    "feature": score_col,
                    "value": aligned_toad_psi(base_values, compare_values),
                }
            )
            timing = benchmark_callable(
                lambda expected=base_values, actual=compare_values: aligned_toad_psi(
                    expected,
                    actual,
                ),
                repeat=repeat,
                warmup=warmup,
            )
            psi_timings.append(
                {
                    "metric": "psi",
                    "scope": "score",
                    "base_split": base_split,
                    "compare_split": compare_split,
                    "feature": score_col,
                    **timing,
                }
            )

        for feature in numeric_features:
            psi_feature_rows.append(
                {
                    "base_split": base_split,
                    "compare_split": compare_split,
                    "feature": feature,
                    "value": aligned_toad_psi(
                        base_frame[feature].to_numpy(),
                        compare_frame[feature].to_numpy(),
                    ),
                }
            )

        timing = benchmark_callable(
            lambda expected=base_frame, actual=compare_frame: [
                aligned_toad_psi(
                    expected[feature].to_numpy(),
                    actual[feature].to_numpy(),
                )
                for feature in numeric_features
            ],
            repeat=repeat,
            warmup=warmup,
        )
        psi_timings.append(
            {
                "metric": "psi",
                "scope": "all_numeric_features",
                "base_split": base_split,
                "compare_split": compare_split,
                "feature": "__all__",
                **timing,
            }
        )

    raw_iv = toad.quality(iv_input, target=TARGET_COL, iv_only=True)
    if isinstance(raw_iv, pd.DataFrame):
        if "iv" in raw_iv.columns:
            iv_series = raw_iv["iv"]
        else:
            iv_series = raw_iv.squeeze(axis=1)
    else:
        iv_series = raw_iv

    iv_rows = [
        {
            "feature": feature,
            "kind": "numeric" if feature in numeric_features else "categorical",
            "value": float(iv_series.get(feature, np.nan)),
        }
        for feature in feature_columns
    ]

    iv_timing = benchmark_callable(
        lambda: toad.quality(iv_input, target=TARGET_COL, iv_only=True),
        repeat=repeat,
        warmup=warmup,
    )

    return {
        "metadata": {
            "toad_runtime": {
                "status": "ok",
                "python": sys.version.split()[0],
                "executable": sys.executable,
                "platform": platform.platform(),
                "toad_version": toad.__version__,
            }
        },
        "correctness": {
            "auc": auc_rows,
            "ks": ks_rows,
            "psi_scores": psi_score_rows,
            "psi_features": psi_feature_rows,
            "iv": iv_rows,
        },
        "efficiency": {
            "auc": auc_timings,
            "ks": ks_timings,
            "psi": psi_timings,
            "iv": [{"implementation": "toad_quality_prepared", **iv_timing}],
        },
    }


def build_report(
    binary: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    feature_columns: Sequence[str],
    numeric_features: Sequence[str],
    newt_results: Dict[str, Any],
    toad_results: Dict[str, Any],
    data_path: Path,
    iv_buckets: int,
    psi_buckets: int,
) -> Dict[str, Any]:
    """Combine Newt and toad results into one benchmark report."""
    toad_status = (
        toad_results.get("metadata", {}).get("toad_runtime", {}).get("status", "error")
    )

    correctness: Dict[str, Any] = {}
    efficiency: Dict[str, Any] = {}

    correctness["auc"] = {
        "rows": compare_value_rows(
            newt_results["correctness"]["auc"],
            toad_results.get("correctness", {}).get("auc", []),
            keys=("split", "score"),
        )
    }
    correctness["auc"]["summary"] = summarize_differences(correctness["auc"]["rows"])

    correctness["ks"] = {
        "rows": compare_value_rows(
            newt_results["correctness"]["ks"],
            toad_results.get("correctness", {}).get("ks", []),
            keys=("split", "score"),
        )
    }
    correctness["ks"]["summary"] = summarize_differences(correctness["ks"]["rows"])

    correctness["psi_scores"] = {
        "rows": compare_value_rows(
            newt_results["correctness"]["psi_scores"],
            toad_results.get("correctness", {}).get("psi_scores", []),
            keys=("base_split", "compare_split", "feature"),
        )
    }
    correctness["psi_scores"]["summary"] = summarize_differences(
        correctness["psi_scores"]["rows"]
    )

    correctness["psi_features"] = {
        "rows": compare_value_rows(
            newt_results["correctness"]["psi_features"],
            toad_results.get("correctness", {}).get("psi_features", []),
            keys=("base_split", "compare_split", "feature"),
        )
    }
    correctness["psi_features"]["summary"] = summarize_differences(
        correctness["psi_features"]["rows"]
    )
    correctness["psi_features"]["top_diffs"] = sort_top_rows(
        correctness["psi_features"]["rows"]
    )

    correctness["iv"] = {
        "rows": compare_value_rows(
            newt_results["correctness"]["iv"],
            toad_results.get("correctness", {}).get("iv", []),
            keys=("feature",),
        )
    }
    correctness["iv"]["summary"] = summarize_differences(correctness["iv"]["rows"])
    correctness["iv"]["top_diffs"] = sort_top_rows(correctness["iv"]["rows"])

    paired_timings = []
    paired_timings.extend(
        compare_timing_rows(
            newt_results["efficiency"]["auc"],
            toad_results.get("efficiency", {}).get("auc", []),
            keys=("metric", "split", "score"),
        )
    )
    paired_timings.extend(
        compare_timing_rows(
            newt_results["efficiency"]["ks"],
            toad_results.get("efficiency", {}).get("ks", []),
            keys=("metric", "split", "score"),
        )
    )
    paired_timings.extend(
        compare_timing_rows(
            newt_results["efficiency"]["psi"],
            toad_results.get("efficiency", {}).get("psi", []),
            keys=("metric", "scope", "base_split", "compare_split", "feature"),
        )
    )

    efficiency["paired"] = paired_timings
    efficiency["iv"] = list(newt_results["efficiency"]["iv"]) + list(
        toad_results.get("efficiency", {}).get("iv", [])
    )

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_path": str(data_path),
            "dataset": {
                "binary_rows": int(len(binary)),
                "binary_bad_rate": float(binary[TARGET_COL].mean()),
                "split_rows": {
                    split_name: int(len(split_frame))
                    for split_name, split_frame in splits.items()
                },
                "feature_count": len(feature_columns),
                "numeric_feature_count": len(numeric_features),
                "iv_buckets": iv_buckets,
                "psi_buckets": psi_buckets,
            },
            "newt_runtime": newt_results["metadata"]["runtime"],
            "toad_runtime": toad_results.get("metadata", {}).get("toad_runtime", {}),
            "toad_status": toad_status,
        },
        "correctness": correctness,
        "efficiency": efficiency,
    }


def format_float(value: Any, digits: int = 6) -> str:
    """Format a float-like value for markdown."""
    if value is None:
        return "n/a"
    value = float(value)
    if math.isnan(value):
        return "n/a"
    return "{:.{digits}f}".format(value, digits=digits)


def format_speedup(value: Any) -> str:
    """Format a speedup value for markdown."""
    if value is None:
        return "n/a"
    value = float(value)
    if math.isnan(value):
        return "n/a"
    return "{:.2f}x".format(value)


def build_markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    """Build a simple markdown table."""
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]

    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the benchmark report as markdown."""
    meta = report["metadata"]
    toad_runtime = meta.get("toad_runtime", {})

    lines = [
        "# Newt vs toad Metric Benchmark",
        "",
        "Generated: `{}`".format(meta["generated_at"]),
        "",
        "Data: `{}`".format(meta["data_path"]),
        "",
        "Binary rows: `{}`".format(meta["dataset"]["binary_rows"]),
        "",
        "Split sizes: `train={train}`, `test={test}`, `oot={oot}`".format(
            **meta["dataset"]["split_rows"]
        ),
        "",
        "Feature counts: `all={}`, `numeric={}`".format(
            meta["dataset"]["feature_count"],
            meta["dataset"]["numeric_feature_count"],
        ),
        "",
        "Newt runtime: Python `{}` on `{}`".format(
            meta["newt_runtime"]["python"],
            meta["newt_runtime"]["platform"],
        ),
        "",
        "Toad runtime status: `{}`".format(meta["toad_status"]),
    ]

    if meta["toad_status"] == "ok":
        lines.extend(
            [
                "",
                "Toad Python: `{}`".format(toad_runtime.get("python", "n/a")),
                "",
                "Toad executable: `{}`".format(toad_runtime.get("executable", "n/a")),
                "",
                "Toad version: `{}`".format(toad_runtime.get("toad_version", "n/a")),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Toad benchmark failed.",
            ]
        )
        error = toad_runtime.get("error")
        if error:
            lines.extend(["", f"Reason: `{error}`"])

    auc_rows = [
        {
            "split": row["split"],
            "score": row["score"],
            "newt": format_float(row["newt"]),
            "toad": format_float(row["toad"]),
            "abs_diff": format_float(row["abs_diff"]),
        }
        for row in report["correctness"]["auc"]["rows"]
    ]
    lines.extend(
        [
            "",
            "## Correctness",
            "",
            "AUC summary: mean abs diff `{}`, max abs diff `{}`".format(
                format_float(report["correctness"]["auc"]["summary"]["mean_abs_diff"]),
                format_float(report["correctness"]["auc"]["summary"]["max_abs_diff"]),
            ),
            "",
            build_markdown_table(
                auc_rows, ["split", "score", "newt", "toad", "abs_diff"]
            ),
        ]
    )

    ks_rows = [
        {
            "split": row["split"],
            "score": row["score"],
            "newt": format_float(row["newt"]),
            "toad": format_float(row["toad"]),
            "abs_diff": format_float(row["abs_diff"]),
        }
        for row in report["correctness"]["ks"]["rows"]
    ]
    lines.extend(
        [
            "",
            "KS summary: mean abs diff `{}`, max abs diff `{}`".format(
                format_float(report["correctness"]["ks"]["summary"]["mean_abs_diff"]),
                format_float(report["correctness"]["ks"]["summary"]["max_abs_diff"]),
            ),
            "",
            build_markdown_table(
                ks_rows, ["split", "score", "newt", "toad", "abs_diff"]
            ),
        ]
    )

    psi_score_rows = [
        {
            "pair": "{}->{}".format(row["base_split"], row["compare_split"]),
            "feature": row["feature"],
            "newt": format_float(row["newt"]),
            "toad": format_float(row["toad"]),
            "abs_diff": format_float(row["abs_diff"]),
        }
        for row in report["correctness"]["psi_scores"]["rows"]
    ]
    lines.extend(
        [
            "",
            "PSI score summary: mean abs diff `{}`, max abs diff `{}`".format(
                format_float(
                    report["correctness"]["psi_scores"]["summary"]["mean_abs_diff"]
                ),
                format_float(
                    report["correctness"]["psi_scores"]["summary"]["max_abs_diff"]
                ),
            ),
            "",
            build_markdown_table(
                psi_score_rows, ["pair", "feature", "newt", "toad", "abs_diff"]
            ),
        ]
    )

    iv_top_rows = [
        {
            "feature": row["feature"],
            "newt": format_float(row["newt"]),
            "toad": format_float(row["toad"]),
            "abs_diff": format_float(row["abs_diff"]),
        }
        for row in report["correctness"]["iv"]["top_diffs"]
    ]
    lines.extend(
        [
            "",
            "IV summary: mean abs diff `{}`, max abs diff `{}`".format(
                format_float(report["correctness"]["iv"]["summary"]["mean_abs_diff"]),
                format_float(report["correctness"]["iv"]["summary"]["max_abs_diff"]),
            ),
            "",
            "Top IV diffs:",
            "",
            build_markdown_table(iv_top_rows, ["feature", "newt", "toad", "abs_diff"]),
        ]
    )

    psi_feature_top_rows = [
        {
            "pair": "{}->{}".format(row["base_split"], row["compare_split"]),
            "feature": row["feature"],
            "newt": format_float(row["newt"]),
            "toad": format_float(row["toad"]),
            "abs_diff": format_float(row["abs_diff"]),
        }
        for row in report["correctness"]["psi_features"]["top_diffs"]
    ]
    lines.extend(
        [
            "",
            "Top PSI feature diffs:",
            "",
            build_markdown_table(
                psi_feature_top_rows,
                ["pair", "feature", "newt", "toad", "abs_diff"],
            ),
        ]
    )

    paired_timing_rows = [
        {
            "metric": row["metric"],
            "scope": row.get("score") or row.get("feature") or row.get("scope"),
            "pair": "{}->{}".format(
                row.get("base_split", row.get("split", "")),
                row.get("compare_split", ""),
            ).strip("->"),
            "newt_ms": format_float(row["newt_ms"], digits=3),
            "toad_ms": format_float(row["toad_ms"], digits=3),
            "newt_speedup": format_speedup(row["newt_speedup"]),
        }
        for row in report["efficiency"]["paired"]
    ]
    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            build_markdown_table(
                paired_timing_rows,
                ["metric", "scope", "pair", "newt_ms", "toad_ms", "newt_speedup"],
            ),
        ]
    )

    iv_timing_rows = [
        {
            "implementation": row["implementation"],
            "median_ms": format_float(row.get("median_ms"), digits=3),
            "status": row.get("status", "ok"),
            "note": row.get("error", ""),
        }
        for row in report["efficiency"]["iv"]
    ]
    lines.extend(
        [
            "",
            "IV timing details:",
            "",
            build_markdown_table(
                iv_timing_rows,
                ["implementation", "median_ms", "status", "note"],
            ),
            "",
        ]
    )

    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=str(default_data_path()),
        help="Path to the parquet dataset used for the benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory that will receive JSON and Markdown benchmark reports.",
    )
    parser.add_argument("--repeat", type=int, default=5, help="Benchmark repeats.")
    parser.add_argument("--warmup", type=int, default=1, help="Benchmark warmup runs.")
    parser.add_argument(
        "--iv-buckets",
        type=int,
        default=10,
        help="Number of buckets used for aligned IV comparisons.",
    )
    parser.add_argument(
        "--psi-buckets",
        type=int,
        default=10,
        help="Number of buckets used for PSI calculations.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark CLI."""
    args = parse_args(argv)
    data_path = Path(args.data_path).resolve()

    binary, splits = load_benchmark_dataset(data_path)
    feature_columns = get_feature_columns(binary)
    numeric_features = get_numeric_feature_columns(binary)

    newt_results = compute_newt_results(
        binary=binary,
        splits=splits,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        repeat=args.repeat,
        warmup=args.warmup,
        iv_buckets=args.iv_buckets,
        psi_buckets=args.psi_buckets,
    )

    toad_results = compute_toad_results(
        binary=binary,
        splits=splits,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        repeat=args.repeat,
        warmup=args.warmup,
        iv_buckets=args.iv_buckets,
        psi_buckets=args.psi_buckets,
    )

    report = build_report(
        binary=binary,
        splits=splits,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        newt_results=newt_results,
        toad_results=toad_results,
        data_path=data_path,
        iv_buckets=args.iv_buckets,
        psi_buckets=args.psi_buckets,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / BENCHMARK_OUTPUT_JSON
    markdown_path = output_dir / BENCHMARK_OUTPUT_MD

    json_path.write_text(
        json.dumps(make_serializable(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    print("Wrote benchmark JSON to {}".format(json_path))
    print("Wrote benchmark Markdown to {}".format(markdown_path))

    return 0 if report["metadata"]["toad_status"] == "ok" else 1
