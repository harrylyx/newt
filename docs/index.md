# Newt

A lightweight Python toolkit for efficient feature analysis and statistical diagnostics in credit risk modeling.

## Key Features

- **6 Binning Algorithms**: ChiMerge, Decision Tree, K-Means, Equal Frequency, Equal Width, Optimal.
- **Monotonic Support**: Comprehensive monotonic binning support (ascending, descending, auto-detect).
- **WOE/IV Analysis**: Robust WOE encoding and Information Value (IV) calculation.
- **Feature Selection**: Pipeline-style feature selection (IV, PSI, VIF, Stepwise).
- **Scorecard Generation**: End-to-end scorecard generation and scoring.
- **High Performance**: Rust-backed engine for IV calculation.

## Installation

```bash
pip install newt
```

## Quick Start

- [User Guide](user_guide.md)
- [API Reference](api/reference.md)
- [PSI Engine Performance](benchmarks/psi_performance.md)
- [Stepwise Selection Performance](benchmarks/stepwise_performance.md)
