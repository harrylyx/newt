"""Benchmark Stepwise Selection performance (Rust Parallel vs Python Serial)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
from sklearn.datasets import make_classification

from newt.features.selection import StepwiseSelector


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


def generate_data(n_samples: int, n_features: int, n_informative: int, seed: int):
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_informative,
        random_state=seed,
    )
    feature_names = [f"feat_{i:03d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    return df, pd.Series(y)


def run_scenario(
    n_samples: int,
    n_features: int,
    n_informative: int,
    direction: str,
    criterion: str,
    seed: int,
):
    """Run a single stepwise benchmark scenario."""
    X, y = generate_data(n_samples, n_features, n_informative, seed)

    # 1. Python Engine
    selector_py = StepwiseSelector(
        direction=direction, criterion=criterion, engine="python", verbose=False
    )
    start_py = time.perf_counter()
    selector_py.fit(X, y)
    duration_py_ms = (time.perf_counter() - start_py) * 1000.0

    # 2. Rust Engine
    selector_rust = StepwiseSelector(
        direction=direction, criterion=criterion, engine="rust", verbose=False
    )
    start_rust = time.perf_counter()
    selector_rust.fit(X, y)
    duration_rust_ms = (time.perf_counter() - start_rust) * 1000.0

    # Consistency Check
    py_set = set(selector_py.selected_features_)
    rust_set = set(selector_rust.selected_features_)
    overlap = len(py_set & rust_set)
    total_distinct = len(py_set | rust_set)
    overlap_pct = (overlap / total_distinct * 100.0) if total_distinct > 0 else 100.0

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_informative": n_informative,
        "direction": direction,
        "criterion": criterion,
        "python_ms": duration_py_ms,
        "rust_ms": duration_rust_ms,
        "speedup": duration_py_ms / duration_rust_ms if duration_rust_ms > 0 else 0,
        "overlap_pct": overlap_pct,
        "n_selected": len(rust_set),
        "peak_rss_mb": get_peak_rss_mb(),
    }


def build_markdown_report(payload: Dict[str, object]) -> str:
    """Build Markdown benchmark report."""
    lines = [
        "# Stepwise Selection Performance Benchmark",
        "",
        "Compares Python (Statsmodels) vs Rust (Parallel IRLS) engines.",
        "",
        "## Environment",
        f"- Generated at: `{payload['metadata']['generated_at']}`",
        f"- Python: `{payload['metadata']['python_version']}`",
        f"- Platform: `{payload['metadata']['platform']}`",
        f"- OS CPU count: `{payload['metadata']['cpu_count']}`",
        "",
        "## Scenarios",
        "",
        "| rows | features | direction | python_ms | rust_ms | speedup | "
        "overlap | rss_mb |",
        "| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sc in payload["scenarios"]:
        lines.append(
            f"| {sc['n_samples']} | {sc['n_features']} | {sc['direction']} | "
            f"{sc['python_ms']:.2f} | {sc['rust_ms']:.2f} | {sc['speedup']:.1f}x | "
            f"{sc['overlap_pct']:.1f}% | {sc['peak_rss_mb']:.1f} |"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Benchmark Stepwise Selection")
    parser.add_argument("--samples", default="5000,15000", help="Row counts.")
    parser.add_argument("--features", default="50,150", help="Feature counts.")
    parser.add_argument(
        "--direction", default="forward", choices=["forward", "backward", "both"]
    )
    parser.add_argument("--criterion", default="aic", choices=["aic", "bic", "pvalue"])
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    args = parser.parse_args(argv)

    samples_list = [int(x) for x in args.samples.split(",")]
    features_list = [int(x) for x in args.features.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []
    for rows in samples_list:
        for feats in features_list:
            n_informative = min(feats // 5, 20)
            print(f"Running: {rows} rows, {feats} features...")
            res = run_scenario(
                rows, feats, n_informative, args.direction, args.criterion, args.seed
            )
            scenarios.append(res)

    payload = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "scenarios": scenarios,
    }

    md_path = output_dir / "stepwise_performance.md"
    md_path.write_text(build_markdown_report(payload), encoding="utf-8")

    json_path = output_dir / "stepwise_performance.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nReport written to: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
