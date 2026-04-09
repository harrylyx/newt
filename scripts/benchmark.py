#!/usr/bin/env python
"""Unified Benchmark Orchestrator for Newt.

Usage:
    python scripts/benchmark.py <command> [options]

Commands:
    metrics     Validate Newt metrics (AUC, KS, IV, PSI) against toad.
    psi         Measure PSI calculation performance (Scalar vs Batch Python/Rust).
    chi         Measure ChiMerge binning performance (Python vs Rust).
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path to allow importing from 'benchmarks' and 'src'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from benchmarks.chimerge_performance import main as chi_main
    from benchmarks.metric_vs_toad import main as metrics_main
    from benchmarks.psi_performance import main as psi_main
except ImportError as e:
    print(f"Error: Failed to import benchmark modules. {e}")
    sys.exit(1)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for benchmark orchestration."""
    parser = argparse.ArgumentParser(
        description="Unified Benchmark Orchestrator for Newt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Benchmark command to run")

    # Command: metrics
    subparsers.add_parser(
        "metrics", help="Validate metrics against toad", add_help=False
    )

    # Command: psi
    subparsers.add_parser("psi", help="Measure PSI engine performance", add_help=False)

    # Command: chi
    subparsers.add_parser(
        "chi", help="Measure ChiMerge engine performance", add_help=False
    )

    args, unknown = parser.parse_known_args(argv)

    if args.command == "metrics":
        return metrics_main(unknown)
    elif args.command == "psi":
        return psi_main(unknown)
    elif args.command == "chi":
        return chi_main(unknown)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
