"""Benchmark ChiMerge Python vs Rust engines."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from newt.features.binning import Binner
from newt.features.binning.supervised import ChiMergeBinner

BENCHMARK_OUTPUT_JSON = "chimerge_performance.json"
BENCHMARK_OUTPUT_MD = "chimerge_performance.md"


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def default_output_dir() -> Path:
    """Return benchmark output directory."""
    return repo_root() / "out" / "benchmarks"


def get_peak_rss_mb() -> float:
    """Return process peak RSS in MB."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(usage) / (1024.0 * 1024.0)
        return float(usage) / 1024.0
    except Exception:
        return float("nan")


def python_chimerge_raw(
    X: pd.Series, y: pd.Series, n_bins: int, alpha: float = 0.05
) -> List[float]:
    """A pure Python implementation of ChiMerge for benchmarking baseline.

    This implementation bypasses any Rust engine integration.
    """
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()

    # 1. Sort
    sort_idx = np.argsort(X_arr)
    X_sorted = X_arr[sort_idx]
    y_sorted = y_arr[sort_idx]

    # 2. Initial binning
    unique_vals, counts = np.unique(X_sorted, return_counts=True)

    # Calculate event counts (the slow Python part)
    event_counts = []
    start = 0
    for count in counts:
        end = start + count
        event_counts.append(np.sum(y_sorted[start:end]))
        start = end

    bins = list(zip(unique_vals, counts, event_counts))

    def compute_chi2(b1, b2):
        n1, e1 = b1[1], b1[2]
        n2, e2 = b2[1], b2[2]
        ne1, ne2 = n1 - e1, n2 - e2

        total = n1 + n2
        if total == 0:
            return 0.0

        expected_e1 = (e1 + e2) * n1 / total
        expected_e2 = (e1 + e2) * n2 / total
        expected_ne1 = (ne1 + ne2) * n1 / total
        expected_ne2 = (ne1 + ne2) * n2 / total

        chi2 = 0.0
        for obs, exp in [
            (e1, expected_e1),
            (e2, expected_e2),
            (ne1, expected_ne1),
            (ne2, expected_ne2),
        ]:
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
        return chi2

    # 3. Merge iterations
    threshold = stats.chi2.ppf(1 - alpha, 1)

    while len(bins) > n_bins:
        chi_squares = [compute_chi2(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        if not chi_squares:
            break

        min_chi = min(chi_squares)
        if min_chi >= threshold:
            break

        idx = chi_squares.index(min_chi)
        # Merge idx and idx+1
        b1, b2 = bins[idx], bins[idx + 1]
        merged = (b2[0], b1[1] + b2[1], b1[2] + b2[2])
        bins[idx : idx + 2] = [merged]

    return [float(b[0]) for b in bins[:-1]]


def load_benchmark_data(
    data_path: str,
    target_col: str,
    rows: int,
    feature_count: int,
    seed: int,
    feature_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and sample real data for ChiMerge benchmarking."""
    if feature_names:
        cols_to_load = list(set(feature_names + [target_col]))
    else:
        # First, get all column names to select numeric ones
        sample = pd.read_parquet(data_path, engine="pyarrow").head(100)
        numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()

        cols_to_load = []
        if target_col in numeric_cols:
            cols_to_load.append(target_col)

        other_cols = [c for c in numeric_cols if c != target_col]
        cols_to_load.extend(other_cols[:feature_count])

    df = pd.read_parquet(data_path, columns=cols_to_load)

    if rows < len(df):
        df = df.sample(n=rows, random_state=seed)

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col], errors="ignore")

    # If explicit feature names given, ensure we return only those in correct order
    if feature_names:
        available = [f for f in feature_names if f in X.columns]
        X = X[available]

    return X, y


def benchmark_python_pure(X: pd.DataFrame, y: pd.Series, n_bins: int) -> float:
    """Run ChiMerge using a pure Python implementation."""
    start = time.perf_counter()
    for col in X.columns:
        python_chimerge_raw(X[col], y, n_bins)
    return (time.perf_counter() - start) * 1000.0


def benchmark_rust_sequential(X: pd.DataFrame, y: pd.Series, n_bins: int) -> float:
    """Run ChiMerge using Rust engine via single-threaded sequential fit."""
    start = time.perf_counter()
    for col in X.columns:
        binner = ChiMergeBinner(n_bins=n_bins)
        binner.fit(X[col], y)
    return (time.perf_counter() - start) * 1000.0


def benchmark_rust_parallel(
    X: pd.DataFrame, y: pd.Series, n_bins: int, show_progress: bool = False
) -> float:
    """Run ChiMerge using the new Rust parallel engine via Binner.fit."""
    binner = Binner()
    start = time.perf_counter()
    binner.fit(X, y, method="chi", n_bins=n_bins, show_progress=show_progress)
    return (time.perf_counter() - start) * 1000.0


def build_markdown_report(payload: Dict[str, Any]) -> str:
    """Build Markdown benchmark report."""
    meta = payload["metadata"]
    lines = [
        "# ChiMerge Engine Benchmark",
        "",
        "Comparing Pure Python implementation vs Rust sequential "
        "and Rust parallel engines.",
        "",
        "## Environment",
        "",
        f"- Generated at: `{meta['generated_at']}`",
        f"- Python: `{meta['python_version']}`",
        f"- Platform: `{meta['platform']}`",
        f"- Data Source: `{meta.get('data_source', 'Synthetic')}`",
        "",
        "## Performance Metrics",
        "",
        "| rows | features | n_bins | python_pure_ms | rust_seq_ms | "
        "rust_par_ms | rust_vs_py_speedup |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for s in payload["scenarios"]:
        speedup = s["python_ms"] / s["rust_ms"] if s["rust_ms"] > 0 else 0
        lines.append(
            f"| {s['rows']} | {s['features']} | {s['n_bins']} | "
            f"{s['python_ms']:.2f} | {s['rust_ms']:.2f} | "
            f"{s['rust_par_ms']:.2f} | {speedup:.2f}x |"
        )

    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    if meta.get("data_source") and meta["data_source"] != "Synthetic":
        lines.append(
            f"python benchmarks/chimerge_performance.py "
            f"--data-path {meta['data_source']} "
            f"--rows {meta['rows']} --features {meta['features']}"
        )
    else:
        lines.append(
            "python benchmarks/chimerge_performance.py --rows 100000 --features 10"
        )
    lines.append("```")

    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark ChiMerge Python vs Rust")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to Parquet data file.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Target column name for real data.",
    )
    parser.add_argument(
        "--rows",
        default="10000,50000,100000",
        help="Comma-separated row counts.",
    )
    parser.add_argument(
        "--features",
        default="5,10",
        help="Comma-separated feature counts.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Target number of bins.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Benchmark repeat count.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Output directory.",
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip pure Python benchmark (use if data is very large).",
    )
    parser.add_argument(
        "--feature-names",
        type=str,
        default=None,
        help="Comma-separated list of feature names to benchmark. "
        "Overrides --features count.",
    )
    args = parser.parse_args(argv)

    rows_list = [int(r.strip()) for r in args.rows.split(",") if r.strip()]

    if args.feature_names:
        feat_list = [1]  # Dummy count, we will use the actual names
        feature_names = [f.strip() for f in args.feature_names.split(",") if f.strip()]
    else:
        feat_list = [int(f.strip()) for f in args.features.split(",") if f.strip()]
        feature_names = None

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []
    for rows in rows_list:
        for feats in feat_list:
            if feature_names:
                print(
                    f"Running scenario: rows={rows}, "
                    f"specified features={len(feature_names)}..."
                )
                X, y_series = load_benchmark_data(
                    args.data_path,
                    args.target_col,
                    rows,
                    len(feature_names),
                    seed=42,
                    feature_names=feature_names,
                )
            else:
                print(f"Running scenario: rows={rows}, features={feats}...")
                if args.data_path:
                    X, y_series = load_benchmark_data(
                        args.data_path, args.target_col, rows, feats, seed=42
                    )
                else:
                    X = pd.DataFrame(
                        {f"f{i}": np.random.randn(rows) for i in range(feats)}
                    )
                    y_series = pd.Series(np.random.randint(0, 2, rows))

            # Python Pure
            if not args.skip_python:
                print("  Running Pure Python...")
                py_ms = benchmark_python_pure(X, y_series, args.n_bins)
            else:
                py_ms = 0.0

            # Rust Sequential
            print("  Running Rust Sequential...")
            rs_seq_ms = benchmark_rust_sequential(X, y_series, args.n_bins)

            # Rust Parallel
            print("  Running Rust Parallel...")
            rs_par_ms = benchmark_rust_parallel(X, y_series, args.n_bins)

            scenarios.append(
                {
                    "rows": rows,
                    "features": len(X.columns),
                    "n_bins": args.n_bins,
                    "python_ms": py_ms,
                    "rust_ms": rs_seq_ms,
                    "rust_par_ms": rs_par_ms,
                    "peak_rss_mb": get_peak_rss_mb(),
                }
            )

    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "data_source": args.data_path or "Synthetic",
            "rows": args.rows,
            "features": args.features,
        },
        "scenarios": scenarios,
    }

    json_path = output_dir / BENCHMARK_OUTPUT_JSON
    md_path = output_dir / BENCHMARK_OUTPUT_MD

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown_report(payload))

    print(f"Benchmark results written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
