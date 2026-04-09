"""Benchmark PSI scalar vs batch engines."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from newt.metrics.psi import calculate_psi, calculate_psi_batch

BENCHMARK_OUTPUT_JSON = "psi_performance.json"
BENCHMARK_OUTPUT_MD = "psi_performance.md"


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def default_output_dir() -> Path:
    """Return benchmark output directory."""
    return repo_root() / "out" / "benchmarks"


def parse_int_list(value: str) -> List[int]:
    """Parse comma-separated integer values."""
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer value")
    return parsed


def parse_float_list(value: str) -> List[float]:
    """Parse comma-separated float values."""
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one float value")
    return parsed


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


def generate_scenario_data(
    rows: int,
    feature_count: int,
    group_count: int,
    missing_rate: float,
    seed: int,
) -> Dict[str, object]:
    """Generate synthetic PSI benchmark data."""
    rng = np.random.default_rng(seed)
    features = [f"feature_{index + 1}" for index in range(feature_count)]
    groups = [f"group_{index + 1}" for index in range(group_count)]

    reference: Dict[str, np.ndarray] = {}
    actual_groups: Dict[str, Dict[str, np.ndarray]] = {group: {} for group in groups}

    for feature_index, feature_name in enumerate(features):
        base = rng.normal(loc=0.0, scale=1.0, size=rows).astype(float)
        if missing_rate > 0.0:
            base_mask = rng.random(rows) < missing_rate
            base[base_mask] = np.nan
        reference[feature_name] = base

        for group_index, group_name in enumerate(groups):
            shift = 0.03 * (group_index + 1)
            noise = rng.normal(
                loc=shift,
                scale=0.25 + 0.01 * feature_index,
                size=rows,
            )
            values = base + noise
            if missing_rate > 0.0:
                missing_mask = rng.random(rows) < missing_rate
                values[missing_mask] = np.nan
            actual_groups[group_name][feature_name] = values.astype(float)

    return {
        "features": features,
        "groups": groups,
        "reference": reference,
        "actual_groups": actual_groups,
    }


def benchmark_callable(
    func,
    repeat: int,
    warmup: int,
) -> Dict[str, float]:
    """Benchmark a callable and return summary stats in milliseconds."""
    for _ in range(warmup):
        func()

    timings: List[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        func()
        timings.append((time.perf_counter() - started) * 1000.0)

    return {
        "repeat": float(repeat),
        "warmup": float(warmup),
        "median_ms": float(statistics.median(timings)),
        "min_ms": float(min(timings)),
        "max_ms": float(max(timings)),
    }


def run_scalar_loop(
    features: Sequence[str],
    groups: Sequence[str],
    reference: Dict[str, np.ndarray],
    actual_groups: Dict[str, Dict[str, np.ndarray]],
    buckets: int,
    nan_strategy: str,
) -> None:
    """Run scalar PSI loop implementation."""
    for feature in features:
        expected = reference[feature]
        for group in groups:
            calculate_psi(
                expected=expected,
                actual=actual_groups[group][feature],
                buckets=buckets,
                nan_strategy=nan_strategy,
            )


def run_batch_engine(
    features: Sequence[str],
    groups: Sequence[str],
    reference: Dict[str, np.ndarray],
    actual_groups: Dict[str, Dict[str, np.ndarray]],
    buckets: int,
    nan_strategy: str,
    engine: str,
) -> None:
    """Run batch PSI engine implementation."""
    for feature in features:
        calculate_psi_batch(
            expected=reference[feature],
            actual_groups=[actual_groups[group][feature] for group in groups],
            buckets=buckets,
            nan_strategy=nan_strategy,
            engine=engine,
        )


def collect_scalar_values(
    features: Sequence[str],
    groups: Sequence[str],
    reference: Dict[str, np.ndarray],
    actual_groups: Dict[str, Dict[str, np.ndarray]],
    buckets: int,
    nan_strategy: str,
) -> List[float]:
    """Collect scalar PSI values for every feature/group pair."""
    values: List[float] = []
    for feature in features:
        expected = reference[feature]
        for group in groups:
            value = calculate_psi(
                expected=expected,
                actual=actual_groups[group][feature],
                buckets=buckets,
                nan_strategy=nan_strategy,
            )
            values.append(float(value))
    return values


def collect_batch_values(
    features: Sequence[str],
    groups: Sequence[str],
    reference: Dict[str, np.ndarray],
    actual_groups: Dict[str, Dict[str, np.ndarray]],
    buckets: int,
    nan_strategy: str,
    engine: str,
) -> List[float]:
    """Collect batch PSI values for every feature/group pair."""
    values: List[float] = []
    for feature in features:
        batch_values = calculate_psi_batch(
            expected=reference[feature],
            actual_groups=[actual_groups[group][feature] for group in groups],
            buckets=buckets,
            nan_strategy=nan_strategy,
            engine=engine,
        )
        values.extend(float(item) for item in batch_values)
    return values


def diff_stats(
    baseline: Sequence[float], compared: Sequence[float]
) -> Dict[str, float]:
    """Return max/mean absolute diff between two value sequences."""
    if len(baseline) != len(compared):
        raise ValueError("Cannot compare PSI outputs with different lengths")

    diffs: List[float] = []
    for left, right in zip(baseline, compared):
        if np.isnan(left) and np.isnan(right):
            diffs.append(0.0)
        elif np.isnan(left) or np.isnan(right):
            diffs.append(float("inf"))
        else:
            diffs.append(abs(float(left) - float(right)))

    if not diffs:
        return {"max_abs_diff": 0.0, "mean_abs_diff": 0.0}
    return {
        "max_abs_diff": float(np.max(diffs)),
        "mean_abs_diff": float(np.mean(diffs)),
    }


def build_markdown_report(payload: Dict[str, object]) -> str:
    """Build Markdown benchmark report."""
    lines: List[str] = []
    lines.append("# PSI Engine Benchmark")
    lines.append("")
    lines.append(
        "This benchmark compares scalar `calculate_psi` against "
        "`calculate_psi_batch(engine='python')` and "
        "`calculate_psi_batch(engine='rust')`."
    )
    lines.append("")
    metadata = payload["metadata"]
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Generated at: `{metadata['generated_at']}`")
    lines.append(f"- Python: `{metadata['python_version']}`")
    lines.append(f"- Platform: `{metadata['platform']}`")
    lines.append(f"- CPU count: `{metadata['cpu_count']}`")
    lines.append("")
    lines.append("## Scenarios")
    lines.append("")
    lines.append(
        "| rows | features | groups | missing_rate | scalar_ms | batch_python_ms | "
        "batch_rust_ms | rust_speedup_vs_scalar | python_speedup_vs_scalar | "
        "py_batch_max_abs_diff | py_batch_mean_abs_diff | rust_batch_max_abs_diff | "
        "rust_batch_mean_abs_diff | avg_scalar_ms_per_feat_group | "
        "avg_rust_ms_per_feat_group | peak_rss_mb |"
    )
    lines.append(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: |"
    )

    for scenario in payload["scenarios"]:
        lines.append(
            "| {rows} | {feature_count} | {group_count} | {missing_rate:.2f} | "
            "{scalar_median_ms:.3f} | {batch_python_median_ms:.3f} | "
            "{batch_rust_median_ms:.3f} | {rust_speedup_vs_scalar:.2f}x | "
            "{python_speedup_vs_scalar:.2f}x | {python_batch_max_abs_diff:.3e} | "
            "{python_batch_mean_abs_diff:.3e} | {rust_batch_max_abs_diff:.3e} | "
            "{rust_batch_mean_abs_diff:.3e} | {scalar_per_fg_ms:.6f} | "
            "{rust_per_fg_ms:.6f} | {peak_rss_mb:.2f} |".format(**scenario)
        )

    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python benchmarks/psi_performance.py")
    lines.append("```")
    lines.append("")
    lines.append(
        "Use CLI options (`--rows`, `--features`, `--groups`, `--missing-rates`) "
        "to customize scenario grid."
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run PSI benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark PSI scalar vs batch engines"
    )
    parser.add_argument(
        "--rows",
        default="100000,500000",
        help="Comma-separated row counts per scenario.",
    )
    parser.add_argument(
        "--features",
        default="20,100",
        help="Comma-separated feature counts per scenario.",
    )
    parser.add_argument(
        "--groups",
        default="6,12",
        help="Comma-separated compare-group counts per scenario.",
    )
    parser.add_argument(
        "--missing-rates",
        default="0.00,0.05,0.20",
        help="Comma-separated missing rates per scenario.",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=10,
        help="PSI bucket count.",
    )
    parser.add_argument(
        "--nan-strategy",
        default="separate",
        choices=["separate", "exclude"],
        help="Missing-value strategy for PSI.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Benchmark repeat count per scenario.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup count per scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260406,
        help="Random seed for synthetic data generation.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Output directory for benchmark artifacts.",
    )
    args = parser.parse_args(argv)

    rows_list = parse_int_list(args.rows)
    feature_list = parse_int_list(args.features)
    group_list = parse_int_list(args.groups)
    missing_rates = parse_float_list(args.missing_rates)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / BENCHMARK_OUTPUT_JSON
    output_md = output_dir / BENCHMARK_OUTPUT_MD

    scenarios: List[Dict[str, float]] = []
    scenario_index = 0
    for rows in rows_list:
        for feature_count in feature_list:
            for group_count in group_list:
                for missing_rate in missing_rates:
                    data = generate_scenario_data(
                        rows=rows,
                        feature_count=feature_count,
                        group_count=group_count,
                        missing_rate=missing_rate,
                        seed=args.seed + scenario_index,
                    )
                    scenario_index += 1

                    features = data["features"]
                    groups = data["groups"]
                    reference = data["reference"]
                    actual_groups = data["actual_groups"]

                    scalar_stats = benchmark_callable(
                        func=partial(
                            run_scalar_loop,
                            features=features,
                            groups=groups,
                            reference=reference,
                            actual_groups=actual_groups,
                            buckets=args.buckets,
                            nan_strategy=args.nan_strategy,
                        ),
                        repeat=args.repeat,
                        warmup=args.warmup,
                    )
                    batch_python_stats = benchmark_callable(
                        func=partial(
                            run_batch_engine,
                            features=features,
                            groups=groups,
                            reference=reference,
                            actual_groups=actual_groups,
                            buckets=args.buckets,
                            nan_strategy=args.nan_strategy,
                            engine="python",
                        ),
                        repeat=args.repeat,
                        warmup=args.warmup,
                    )
                    batch_rust_stats = benchmark_callable(
                        func=partial(
                            run_batch_engine,
                            features=features,
                            groups=groups,
                            reference=reference,
                            actual_groups=actual_groups,
                            buckets=args.buckets,
                            nan_strategy=args.nan_strategy,
                            engine="rust",
                        ),
                        repeat=args.repeat,
                        warmup=args.warmup,
                    )

                    scalar_values = collect_scalar_values(
                        features=features,
                        groups=groups,
                        reference=reference,
                        actual_groups=actual_groups,
                        buckets=args.buckets,
                        nan_strategy=args.nan_strategy,
                    )
                    python_batch_values = collect_batch_values(
                        features=features,
                        groups=groups,
                        reference=reference,
                        actual_groups=actual_groups,
                        buckets=args.buckets,
                        nan_strategy=args.nan_strategy,
                        engine="python",
                    )
                    rust_batch_values = collect_batch_values(
                        features=features,
                        groups=groups,
                        reference=reference,
                        actual_groups=actual_groups,
                        buckets=args.buckets,
                        nan_strategy=args.nan_strategy,
                        engine="rust",
                    )
                    python_diff = diff_stats(scalar_values, python_batch_values)
                    rust_diff = diff_stats(scalar_values, rust_batch_values)

                    operations = max(feature_count * group_count, 1)
                    scalar_median = float(scalar_stats["median_ms"])
                    batch_python_median = float(batch_python_stats["median_ms"])
                    batch_rust_median = float(batch_rust_stats["median_ms"])

                    scenarios.append(
                        {
                            "rows": int(rows),
                            "feature_count": int(feature_count),
                            "group_count": int(group_count),
                            "missing_rate": float(missing_rate),
                            "scalar_median_ms": scalar_median,
                            "batch_python_median_ms": batch_python_median,
                            "batch_rust_median_ms": batch_rust_median,
                            "python_speedup_vs_scalar": (
                                scalar_median / batch_python_median
                                if batch_python_median > 0
                                else float("nan")
                            ),
                            "rust_speedup_vs_scalar": (
                                scalar_median / batch_rust_median
                                if batch_rust_median > 0
                                else float("nan")
                            ),
                            "scalar_per_fg_ms": scalar_median / float(operations),
                            "python_per_fg_ms": batch_python_median / float(operations),
                            "rust_per_fg_ms": batch_rust_median / float(operations),
                            "python_batch_max_abs_diff": float(
                                python_diff["max_abs_diff"]
                            ),
                            "python_batch_mean_abs_diff": float(
                                python_diff["mean_abs_diff"]
                            ),
                            "rust_batch_max_abs_diff": float(rust_diff["max_abs_diff"]),
                            "rust_batch_mean_abs_diff": float(
                                rust_diff["mean_abs_diff"]
                            ),
                            "peak_rss_mb": get_peak_rss_mb(),
                        }
                    )

    payload = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "nan_strategy": args.nan_strategy,
            "buckets": args.buckets,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "rows": rows_list,
            "features": feature_list,
            "groups": group_list,
            "missing_rates": missing_rates,
        },
        "scenarios": scenarios,
    }

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=True)

    output_md.write_text(build_markdown_report(payload), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
