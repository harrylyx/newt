"""Batch IV calculation helpers with a Rust-backed engine."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from newt.config import BINNING


def calculate_batch_iv(
    X: pd.DataFrame,
    y: pd.Series,
    features: Optional[Sequence[str]] = None,
    bins: int = BINNING.DEFAULT_BUCKETS,
    epsilon: float = BINNING.DEFAULT_EPSILON,
    engine: str = "rust",
) -> pd.DataFrame:
    """Calculate IV for many features."""
    feature_names = list(features) if features is not None else X.columns.tolist()
    target = pd.to_numeric(y, errors="coerce")
    valid_target = target.isin([0, 1])
    target_values = target.loc[valid_target].astype(int).tolist()

    if engine == "python":
        values = [
            _calculate_single_iv(
                pd.to_numeric(X.loc[valid_target, feature], errors="coerce"),
                target.loc[valid_target].astype(int),
                bins=bins,
                epsilon=epsilon,
            )
            for feature in feature_names
        ]
        return pd.DataFrame({"feature": feature_names, "iv": values})

    if engine != "rust":
        raise ValueError("engine must be 'rust' or 'python'")

    rust_module = _load_rust_extension()
    feature_vectors = [
        [
            None if pd.isna(value) else float(value)
            for value in pd.to_numeric(
                X.loc[valid_target, feature],
                errors="coerce",
            ).tolist()
        ]
        for feature in feature_names
    ]
    values = rust_module.calculate_batch_iv(
        feature_vectors,
        target_values,
        int(bins),
        float(epsilon),
    )
    return pd.DataFrame({"feature": feature_names, "iv": values})


def _calculate_single_iv(
    series: pd.Series,
    target: pd.Series,
    bins: int,
    epsilon: float,
) -> float:
    """Python reference implementation for IV."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = numeric.dropna()
    if non_missing.empty or non_missing.nunique() <= 1:
        return 0.0

    edges = _build_quantile_edges(non_missing.to_numpy(dtype=float), bins)
    good_counts = np.zeros(len(edges) - 1 + 1, dtype=float)
    bad_counts = np.zeros(len(edges) - 1 + 1, dtype=float)

    for value, label in zip(numeric.tolist(), target.tolist()):
        index = _bin_index(value, edges)
        if label == 1:
            bad_counts[index] += 1
        else:
            good_counts[index] += 1

    total_good = good_counts.sum()
    total_bad = bad_counts.sum()
    if total_good == 0 or total_bad == 0:
        return 0.0

    dist_good = np.maximum(good_counts / total_good, epsilon)
    dist_bad = np.maximum(bad_counts / total_bad, epsilon)
    return float(np.sum((dist_good - dist_bad) * np.log(dist_good / dist_bad)))


def _build_quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    unique = np.unique(values)
    if unique.size <= 1:
        return np.array([-np.inf, np.inf], dtype=float)
    sorted_values = np.sort(values.astype(float))
    unique_bins = min(bins, unique.size)
    positions = [
        int(round((len(sorted_values) - 1) * (index / unique_bins)))
        for index in range(unique_bins + 1)
    ]
    edges = np.unique(sorted_values[positions])
    if edges.size < 2:
        return np.array([-np.inf, np.inf], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_index(value: object, edges: np.ndarray) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return len(edges) - 1
    return int(np.searchsorted(edges[1:-1], float(value), side="right"))


def _load_rust_extension():
    module_name = "newt_iv_rust"
    try:
        return importlib.import_module(module_name)
    except ImportError:
        _build_rust_extension()
        return importlib.import_module(module_name)


def _build_rust_extension() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    manifest_path = repo_root / "rust" / "newt_iv_rust" / "Cargo.toml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Rust manifest not found: {manifest_path}")

    command = [
        str(_resolve_maturin_executable()),
        "develop",
        "--manifest-path",
        str(manifest_path),
        "--release",
        "--quiet",
    ]
    env = dict(os.environ)
    env.setdefault("PYO3_PYTHON", sys.executable)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to build Rust IV extension.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )


def _resolve_maturin_executable() -> Path:
    direct = shutil.which("maturin")
    if direct:
        return Path(direct)
    candidate = Path(sys.executable).resolve().parent / "maturin"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("maturin executable not found in the current environment.")
