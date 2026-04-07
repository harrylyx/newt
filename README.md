# Newt

A lightweight Python toolkit for efficient feature analysis and statistical diagnostics in credit risk modeling.

## Features

- **6 Binning Algorithms**: ChiMerge, Decision Tree, K-Means, Equal Frequency, Equal Width, Optimal Binning
- **Monotonic Binning Support**: Ascending, descending, and auto-detect monotonic trends
- **WOE/IV Analysis**: Comprehensive Weight of Evidence and Information Value calculations
- **Feature Selection**: Pre-filtering (IV, missing rate, correlation), post-filtering (PSI, VIF), and stepwise selection
- **Logistic Regression**: Statsmodels-based with scikit-learn-like interface
- **Scorecard Generation**: Traditional credit scorecard with customizable base score and PDO
- **Pipeline Workflow**: Fluent API for end-to-end scorecard development
- **Visualization Tools**: Binning plots, IV ranking, WOE patterns, PSI comparison
- **Excel Model Report**: Multi-sheet model report generation with overview, model design, variable analysis, and model performance output
ß
## Installation

Newt supports Python `>=3.8.5,<3.13` (Python `3.8.5` through `3.12.x`).

```bash
pip install newt
```

### Install from GitHub Release

Prebuilt wheels with the Rust-backed IV engine are available from
[GitHub Releases](https://github.com/harrylyx/newt/releases). Download the
wheel matching your platform and Python version, then install directly:

```bash
pip install newt-<version>-<platform-wheel>.whl
```

**Supported platforms:**

| OS | Architecture |
|---|---|
| macOS | arm64 |
| Windows | x86_64 |
| Linux | x86_64, arm64 |

No Rust toolchain is required when installing from an official wheel.

### Optional Dependencies

Install the optional optimal binning stack only when you need `method='opt'`:

```bash
pip install "newt[optbinning]"
```

`optbinning` extras are available on Python `<3.12`. The Newt core package still supports Python `3.8.5` through `3.12.x`.

### Default Local Environments

This repo uses two repo-local environments by default:

- `.venv`: Python `3.8.5` for development, tests, linting, and package edits.
- `.venv-benchmark-3.10`: Python `3.10.19` for `newt-benchmark` and `toad`.

Do not create extra repo-local `.venv-*` directories unless you are debugging a specific issue and plan to delete them afterward.

### Development

For development with uv:

```bash
# Clone the repository
git clone https://github.com/harrylyx/newt.git
cd newt
```

Then recreate the default dev environment with:

```bash
UV_PROJECT_ENVIRONMENT=.venv UV_PYTHON_INSTALL_DIR=.uv-python uv sync --python 3.8.5 --group dev --frozen
```

To recreate the benchmark environment, use:

```bash
UV_PROJECT_ENVIRONMENT=.venv-benchmark-3.10 UV_PYTHON_INSTALL_DIR=.uv-python uv sync --python 3.10.19 --group dev --group benchmark --frozen
```

Use `.venv` for normal development work, tests, and linting. Use `.venv-benchmark-3.10` for benchmark runs.

If you need a one-off interpreter check, use a temporary environment outside the repository and delete it afterward.

For Excel report development and validation (including `openpyxl`, `pyarrow`, and `lightgbm`), use the default `.venv` environment.

Build the Rust extension for development after syncing the environment:

```bash
maturin develop --manifest-path rust/newt_iv_rust/Cargo.toml --release
```

### IV Engine Defaults

Single-feature and batch IV both use the Rust engine by default.

```python
from newt.features.analysis import calculate_batch_iv, calculate_iv

# Default: Rust engine
single = calculate_iv(df, target="target", feature="age")
batch = calculate_batch_iv(X, y)

# Explicit Python fallback
single_py = calculate_iv(df, target="target", feature="age", engine="python")
batch_py = calculate_batch_iv(X, y, engine="python")
```

## Quick Start

```python
import pandas as pd
from newt import Binner, FeatureSelector, WOEEncoder
from newt.modeling import LogisticModel, Scorecard
from newt.pipeline import ScorecardPipeline

# Load data
df = pd.read_csv('credit_data.csv')
target = 'default'

# Option 1: Using the pipeline (recommended)
pipeline = (
    ScorecardPipeline(X_train, y_train, X_test, y_test)
    .prefilter(iv_threshold=0.02)
    .bin(method='quantile', n_bins=5)
    .woe_transform()
    .postfilter(psi_threshold=0.25, vif_threshold=10.0)
    .stepwise(direction='both', criterion='aic')
    .build_model()
    .generate_scorecard(base_score=600, pdo=50)
)

# Score new data
scores = pipeline.score(X_new)

# Option 2: Using individual components
# Feature selection
selector = FeatureSelector()
selector.fit(df, df[target])
selector.select(iv_threshold=0.02)
X = selector.transform(df)

# Binning
binner = Binner()
binner.fit(X, df[target], method='chi', n_bins=5)

# Access binning results
binner['age'].stats        # View statistics
binner['age'].plot()       # Plot results
binner['age'].woe_map()    # Get WOE mapping

# WOE transformation
X_binned = binner.transform(X)
X_woe = X_binned.copy()
for col in X_binned.columns:
    encoder = WOEEncoder()
    encoder.fit(X_binned[col].astype(str), df[target])
    X_woe[col] = encoder.transform(X_binned[col].astype(str))

# Model building
model = LogisticModel()
model.fit(X_woe, df[target])
print(model.summary())

# Scorecard generation
scorecard = Scorecard(base_score=600, pdo=50)
scorecard.from_model(model, binner, binner.woe_encoders_)
scores = scorecard.score(X_new)
```

## Binning Methods

Newt supports 6 binning algorithms:

- **ChiMerge (`chi`)**: Chi-square based supervised binning (default)
- **Decision Tree (`dt`)**: Decision tree based supervised binning
- **Optimal Binning (`opt`)**: Optimal binning with constraints
  (install with `pip install "newt[optbinning]"`)
- **K-Means (`kmean`)**: Unsupervised K-Means clustering
- **Equal Frequency (`quantile`)**: Equal frequency bins (unsupervised)
- **Equal Width (`step`)**: Equal width bins (unsupervised)

### Monotonic Binning

```python
# Auto-detect monotonic trend
binner.fit(X, y, method='chi', monotonic=True)

# Force ascending trend
binner.fit(X, y, method='dt', monotonic='ascending')

# Force descending trend
# Requires the optional optbinning extra
binner.fit(X, y, method='opt', monotonic='descending')
```

## Feature Selection

### Pre-Filtering (EDA-based)
```python
selector = FeatureSelector(
    metrics=['iv', 'missing_rate', 'ks', 'correlation']
)
selector.fit(df, df[target])
selector.select(
    iv_threshold=0.02,
    missing_threshold=0.9,
    corr_threshold=0.8
)
X = selector.transform(df)
```

### Stepwise Selection
```python
from newt.features.selection import StepwiseSelector

stepwise = StepwiseSelector(
    direction='both',
    criterion='aic',
    p_enter=0.05,
    p_remove=0.10
)
X_selected = stepwise.fit_transform(X_woe, y)
```

### Post-Filtering (PSI & VIF)
```python
from newt.features.selection import PostFilter

postfilter = PostFilter(
    psi_threshold=0.25,
    vif_threshold=10.0
)
X_filtered = postfilter.fit_transform(X_train_woe, X_test_woe)
```

## Metrics

Newt provides comprehensive evaluation metrics:

```python
from newt.metrics import (
    calculate_auc,
    calculate_gini,
    calculate_ks,
    calculate_lift,
    calculate_lift_at_k,
    calculate_psi,
    calculate_vif
)

# AUC
auc = calculate_auc(y_true, y_pred_proba)

# Gini coefficient
gini = calculate_gini(y_true, y_pred_proba)

# Kolmogorov-Smirnov statistic
ks = calculate_ks(y_true, y_pred_proba)

# Lift table
lift_df = calculate_lift(y_true, y_pred_proba, bins=10)

# Lift at top K
lift_at_10pct = calculate_lift_at_k(y_true, y_pred_proba, k=0.1)

# Population Stability Index
psi = calculate_psi(expected, actual, buckets=10)

# Variance Inflation Factor
vif_values = calculate_vif(X)
```

## Visualization

```python
from newt.visualization import (
    plot_binning_result,
    plot_iv_ranking,
    plot_woe_pattern,
    plot_psi_comparison
)

# Binning visualization
fig = plot_binning_result(binner, X, y, 'age')

# IV ranking
fig = plot_iv_ranking(iv_dict, top_n=20, threshold=0.02)

# WOE pattern
fig = plot_woe_pattern(woe_encoder['age'], 'age')

# PSI comparison
fig = plot_psi_comparison(psi_dict, threshold=0.25)
```

## Documentation

- [User Guide (English)](docs/user_guide.md) - Comprehensive end-to-end workflow guide
- [用户指南 (中文)](docs/user_guide_zh.md) - Chinese version of the user guide
- [Recent Release Notes (English)](docs/release_notes.md) - Summary of `v0.1.1` to `v0.1.4`
- [最近版本更新说明 (中文)](docs/release_notes_zh.md) - `v0.1.1` 到 `v0.1.4` 版本汇总
- [Benchmark Guide](docs/benchmarks/metric_vs_toad.md) - Compare Newt metrics against toad on the bundled sample dataset
- [PSI Engine Benchmark](docs/benchmarks/psi_performance.md) - Compare scalar PSI with Python/Rust batch engines
- [Examples](examples/) - Jupyter notebook examples

## Benchmark

Run the bundled benchmark on `examples/data/test_data/all_data.pq` from the benchmark environment:

```bash
./.venv-benchmark-3.10/bin/newt-benchmark
```

This writes:

- `out/benchmarks/metric_vs_toad.json`
- `out/benchmarks/metric_vs_toad.md`

The benchmark runs Newt and toad in the same benchmark environment. It does not create a separate worker virtual environment. In this repository, the default host environment for that command is `.venv-benchmark-3.10`.

Run PSI engine performance benchmark:

```bash
./.venv-benchmark-3.10/bin/newt-benchmark-psi
```

This writes:

- `out/benchmarks/psi_performance.json`
- `out/benchmarks/psi_performance.md`

## Dependencies

**Core Dependencies:**
- Python >= 3.8.5, < 3.13
- numpy < 2.0.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- scipy >= 1.10.0
- statsmodels >= 0.14.0
- matplotlib >= 3.7.0
- seaborn >= 0.13.0

**Optional Dependencies (for optimal binning):**
- optbinning >= 0.20.0
- ortools < 9.8
- cvxpy >= 1.3, < 1.5

**Development Dependencies:**
- pytest >= 7.0.0
- flake8 >= 6.0.0
- coverage >= 7.0.0
- pytest-cov >= 4.0.0

## Development

```bash
# Install development dependencies
uv sync --group dev

# Format code
uv run black .
uv run isort .

# Run linter
uv run flake8 src tests

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/newt --cov-report=html
```

## Project Structure

```
newt/
├── src/newt/                      # Source code
│   ├── features/
│   │   ├── binning/              # Binning algorithms
│   │   ├── selection/           # Feature selection
│   │   └── analysis/             # WOE/IV analysis
│   ├── modeling/                 # Logistic regression & scorecard
│   ├── metrics/                  # Evaluation metrics
│   ├── pipeline/                 # Pipeline workflow
│   ├── visualization/            # Plotting functions
│   ├── utils/                    # Utilities
│   └── config.py                 # Configuration constants
├── tests/                         # Tests
│   ├── unit/                     # Unit tests
│   └── cross_val/                # Cross-validation tests
├── docs/                         # Documentation
├── examples/                      # Jupyter notebooks
└── pyproject.toml                # uv/PEP 621 configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [AGENTS.md](AGENTS.md) for detailed development guidelines.

## License

MIT License - see LICENSE file for details.

## Citation

If you use Newt in your research, please cite:

```bibtex
@software{newt2026,
  title = {Newt: Credit Scorecard Development Toolkit},
  author = {harrylyx},
  year = {2026},
  url = {https://github.com/harrylyx/newt}
}
```
