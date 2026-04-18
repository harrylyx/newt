# Newt

A lightweight Python toolkit for efficient feature analysis and statistical diagnostics in credit risk modeling.

## Key Features

- **6 Binning Algorithms**: ChiMerge, Decision Tree, K-Means, Equal Frequency, Equal Width, Optimal.
- **Monotonic Support**: Comprehensive monotonic binning support (ascending, descending, auto-detect).
- **Feature Analysis**: Robust binning statistics and transformation workflows.
- **Feature Selection**: Pipeline-style feature selection (IV, PSI, VIF, Stepwise).
- **Scorecard Generation**: End-to-end scorecard generation and scoring.
- **High Performance**: High-performance Rust engine for core performance paths (Binning, Selection, IV, PSI).

## Installation

```bash
pip install newt
```

## Quick Start

- [User Guide](user_guide.md)
- [API Reference](api/reference.md)

Recommended workflow: fit `Binner`, call `binner.woe_transform(X)`, then build the scorecard with `Scorecard.from_model(model, binner)`. For persistence, use `LogisticModel.dump/load` and `Scorecard.dump/load`.
## Benchmarks
- [Performance vs Toad](benchmarks/metric_vs_toad.md)
- [PSI Performance](benchmarks/psi_performance.md)
- [Stepwise Selection Performance](benchmarks/stepwise_performance.md)
- [ChiMerge Performance](benchmarks/chimerge_performance.md)
