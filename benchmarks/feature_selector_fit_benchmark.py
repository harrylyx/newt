"""Benchmark FeatureSelector.fit runtime across engines.

Usage:
    uv run python benchmarks/feature_selector_fit_benchmark.py
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import numpy as np
import pandas as pd

from newt.features.selection import FeatureSelector


def make_dataset(
    n_rows: int,
    n_numeric: int,
    n_categorical: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    numeric = {
        f"num_{idx}": rng.normal(loc=0.0, scale=1.0 + idx * 0.002, size=n_rows)
        for idx in range(n_numeric)
    }
    categorical = {
        f"cat_{idx}": rng.choice(
            ["A", "B", "C", "D", "E", None],
            size=n_rows,
            p=[0.2, 0.22, 0.24, 0.16, 0.12, 0.06],
        )
        for idx in range(n_categorical)
    }

    base = (
        0.9 * numeric["num_0"]
        - 0.7 * numeric["num_3"]
        + 0.5 * numeric["num_9"]
        + rng.normal(scale=0.5, size=n_rows)
    )
    probs = 1.0 / (1.0 + np.exp(-base))
    target = pd.Series((rng.random(n_rows) < probs).astype(int), name="target")

    frame = pd.DataFrame({**numeric, **categorical})
    return frame, target


def bench_once(X: pd.DataFrame, y: pd.Series, engine: str) -> float:
    selector = FeatureSelector(
        engine=engine,
        metrics=["iv", "missing_rate", "ks", "correlation", "lift_10"],
        iv_bins=10,
        corr_method="pearson",
    )
    start = time.perf_counter()
    selector.fit(X, y)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="FeatureSelector fit benchmark")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--numeric", type=int, default=120)
    parser.add_argument("--categorical", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    X, y = make_dataset(
        n_rows=args.rows,
        n_numeric=args.numeric,
        n_categorical=args.categorical,
        seed=args.seed,
    )

    bench_once(X, y, engine="python")
    bench_once(X, y, engine="auto")

    python_times = [bench_once(X, y, engine="python") for _ in range(args.repeat)]
    auto_times = [bench_once(X, y, engine="auto") for _ in range(args.repeat)]

    python_mean = float(np.mean(python_times))
    auto_mean = float(np.mean(auto_times))
    speedup = python_mean / auto_mean if auto_mean > 0 else np.nan

    print("FeatureSelector.fit benchmark")
    print(f"rows={args.rows}, numeric={args.numeric}, categorical={args.categorical}")
    print(f"python mean: {python_mean:.4f}s")
    print(f"auto   mean: {auto_mean:.4f}s")
    print(f"speedup (python/auto): {speedup:.2f}x")
    print(f"python runs: {[round(v, 4) for v in python_times]}")
    print(f"auto runs: {[round(v, 4) for v in auto_times]}")


if __name__ == "__main__":
    main()
