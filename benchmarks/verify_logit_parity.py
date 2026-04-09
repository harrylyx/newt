"""Benchmark and verify parity between Rust and Statsmodels Logistic Regression."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import statsmodels.api as sm
from sklearn.datasets import make_classification

from newt._newt_iv_rust import fit_logistic_regression_numpy


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


def generate_data(n_samples: int, n_features: int, seed: int):
    """Generate synthetic data for parity check."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_features, 10),
        n_redundant=0,
        random_state=seed,
    )
    # Add constant for intercept
    X_with_const = np.column_stack([np.ones(n_samples), X])
    return X_with_const, y.astype(float)


def run_benchmark_scenario(
    n_samples: int, n_features: int, repeat: int, warmup: int, seed: int
):
    """Run a single parity scenario."""
    X, y = generate_data(n_samples, n_features, seed)

    # Warmup
    for _ in range(warmup):
        sm.Logit(y, X).fit(disp=False)
        fit_logistic_regression_numpy(X, y)

    # 1. Statsmodels
    sm_times = []
    sm_res = None
    for _ in range(repeat):
        start = time.perf_counter()
        sm_res = sm.Logit(y, X).fit(disp=False)
        sm_times.append((time.perf_counter() - start) * 1000.0)

    # 2. Rust
    rust_times = []
    rust_res = None
    for _ in range(repeat):
        start = time.perf_counter()
        rust_res = fit_logistic_regression_numpy(X, y)
        rust_times.append((time.perf_counter() - start) * 1000.0)

    # Calculate differences
    sm_params = sm_res.params
    rust_params = np.array(rust_res["coefficients"])
    max_coef_diff = np.abs(sm_params - rust_params).max()

    sm_pvalues = sm_res.pvalues
    rust_pvalues = np.array(rust_res["p_values"])
    max_p_diff = np.abs(sm_pvalues - rust_pvalues).max()

    aic_diff = abs(sm_res.aic - rust_res["aic"])

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "sm_median_ms": np.median(sm_times),
        "rust_median_ms": np.median(rust_times),
        "max_coef_diff": float(max_coef_diff),
        "max_p_diff": float(max_p_diff),
        "aic_diff": float(aic_diff),
        "peak_rss_mb": get_peak_rss_mb(),
    }


def build_markdown_report(payload: Dict[str, object]) -> str:
    """Build Markdown report."""
    lines = [
        "# Logistic Regression Parity Report",
        "",
        "Verifies numerical parity and performance between Rust IRLS and Statsmodels.",
        "",
        "## Environment",
        f"- Generated at: `{payload['metadata']['generated_at']}`",
        f"- Python: `{payload['metadata']['python_version']}`",
        f"- Platform: `{payload['metadata']['platform']}`",
        "",
        "## Scenarios",
        "",
        "| samples | features | sm_ms | rust_ms | speedup | max_coef_diff | "
        "max_p_diff | aic_diff |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sc in payload["scenarios"]:
        speedup = (
            sc["sm_median_ms"] / sc["rust_median_ms"] if sc["rust_median_ms"] > 0 else 0
        )
        lines.append(
            f"| {sc['n_samples']} | {sc['n_features']} | {sc['sm_median_ms']:.2f} | "
            f"{sc['rust_median_ms']:.2f} | {speedup:.2f}x | "
            f"{sc['max_coef_diff']:.2e} | "
            f"{sc['max_p_diff']:.2e} | {sc['aic_diff']:.2e} |"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Verify Rust Logistic Regression parity"
    )
    parser.add_argument(
        "--samples", default="5000,20000", help="Comma-separated sample counts."
    )
    parser.add_argument(
        "--features", default="10,50", help="Comma-separated feature counts."
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    args = parser.parse_args(argv)

    samples_list = [int(x) for x in args.samples.split(",")]
    features_list = [int(x) for x in args.features.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []
    for n_samples in samples_list:
        for n_features in features_list:
            print(f"Running scenario: {n_samples} samples, {n_features} features...")
            res = run_benchmark_scenario(
                n_samples, n_features, args.repeat, args.warmup, args.seed
            )
            scenarios.append(res)

    payload = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "scenarios": scenarios,
    }

    md_path = output_dir / "logit_parity.md"
    md_path.write_text(build_markdown_report(payload), encoding="utf-8")

    json_path = output_dir / "logit_parity.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nReport written to: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
