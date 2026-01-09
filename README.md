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

## Installation

```bash
pip install newt
```

For development with Poetry:

```bash
# Clone the repository
git clone https://github.com/harrylyx/newt.git
cd newt

# Install dependencies
python -m poetry install
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
    .bin(method='opt', n_bins=5)
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
- **Optimal Binning (`opt`)**: Optimal binning with constraints (requires `optbinning`)
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
- [Examples](examples/) - Jupyter notebook examples

## Dependencies

**Core Dependencies:**
- Python >= 3.8.1, < 4.0
- numpy < 2.0.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- scipy >= 1.10.0
- statsmodels >= 0.14.0
- matplotlib >= 3.7.0
- seaborn >= 0.13.0

**Optional Dependencies (for optimal binning):**
- optbinning >= 0.20.0
- ortools < 9.12
- cvxpy >= 1.3, < 1.5

**Development Dependencies:**
- pytest >= 7.0.0
- flake8 >= 6.0.0
- coverage >= 7.0.0
- pytest-cov >= 4.0.0

## Development

```bash
# Install development dependencies
python -m poetry install

# Format code
python -m poetry run black .
python -m poetry run isort .

# Run linter
python -m poetry run flake8 src tests

# Run tests
python -m poetry run pytest

# Run tests with coverage
python -m poetry run pytest --cov=src/newt --cov-report=html
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
└── pyproject.toml                # Poetry configuration
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
