# Newt - Agent Development Guide

## 1. Project Overview

Newt is a lightweight Python toolkit for efficient feature analysis and statistical diagnostics in credit risk modeling.

- **Python Version**: >=3.8.1, <4.0
- **Dependency Management**: Poetry
- **Core Features**:
  - 6 binning algorithms (ChiMerge, Decision Tree, K-Means, Equal Frequency, Equal Width, Optimal)
  - Monotonic binning support (ascending, descending, auto-detect)
  - WOE/IV analysis and encoding
  - Feature selection (pre-filtering, post-filtering, stepwise regression)
  - Logistic regression modeling
  - Scorecard generation
  - Pipeline-style workflow
  - Visualization tools (binning plots, IV ranking, WOE patterns, PSI comparison)

## 2. Development Environment

**IMPORTANT:** Always use `python -m poetry` prefix for all commands - no environment variables configured.

- **OS**: Windows 11
- **Terminal**: PowerShell
- **Environment**: No global environment variables configured
- **Command Prefix**: Always use `python -m` for all Python commands

```bash
# ❌ Do NOT use direct poetry commands
poetry run pytest
poetry run python -m pytest
pytest

# ✅ Always use python -m prefix
python -m pytest
python -m poetry
```

## 3. Project Architecture

### 3.1 Directory Structure

```
newt/
├── src/newt/                      # Source code
│   ├── features/
│   │   ├── binning/              # Binning (Mixin pattern, monotonic support)
│   │   │   ├── base.py          # BaseBinner (abstract, monotonic adjustment)
│   │   │   ├── binner.py        # Binner (unified interface)
│   │   │   ├── binner_mixins.py # Stats, IO, WOE mixins
│   │   │   ├── binning_stats.py # Statistical calculations
│   │   │   ├── woe_storage.py   # WOE encoder management
│   │   │   ├── supervised.py    # ChiMerge, DecisionTree, OptBinning
│   │   │   └── unsupervised.py  # EqualWidth, EqualFrequency, KMeans
│   │   ├── selection/
│   │   │   ├── selector.py      # FeatureSelector (EDA + filtering)
│   │   │   ├── postfilter.py    # PostFilter (PSI, VIF)
│   │   │   └── stepwise.py     # StepwiseSelector
│   │   └── analysis/
│   │       ├── woe_calculator.py # WOEEncoder
│   │       ├── iv_calculator.py   # calculate_iv
│   │       └── correlation.py    # Correlation matrix
│   ├── modeling/
│   │   ├── logistic.py          # LogisticModel (statsmodels wrapper)
│   │   └── scorecard.py         # Scorecard generation
│   ├── metrics/
│   │   ├── auc.py              # calculate_auc
│   │   ├── gini.py             # calculate_gini
│   │   ├── ks.py               # calculate_ks
│   │   ├── lift.py             # calculate_lift
│   │   ├── psi.py              # calculate_psi
│   │   └── vif.py              # calculate_vif
│   ├── pipeline/
│   │   └── pipeline.py         # ScorecardPipeline (fluent API)
│   ├── utils/
│   │   └── decorators.py       # @requires_fit decorator
│   ├── visualization/
│   │   ├── binning_viz.py     # Visualization functions
│   │   └── binning.py          # Plotting utilities
│   ├── config.py                # Configuration constants
│   └── __init__.py              # Package exports
├── tests/                         # Tests
│   ├── unit/                    # Unit tests
│   │   ├── features/
│   │   │   ├── binning/
│   │   │   ├── analysis/
│   │   │   └── selection/
│   │   ├── metrics/
│   │   ├── modeling/
│   │   ├── pipeline/
│   │   ├── utils/
│   │   └── visualization/
│   ├── cross_val/                # Cross-validation tests
│   │   └── compare_with_toad.py
│   └── conftest.py              # Shared fixtures
├── docs/                         # Documentation
│   ├── user_guide.md           # English user guide
│   └── user_guide_zh.md        # Chinese user guide
├── examples/                      # Example notebooks
│   ├── 01-basic-usage.ipynb
│   ├── 02-advanced-analysis.ipynb
│   ├── 03-production-pipeline.ipynb
│   └── data/                   # Sample datasets
│       └── statlog+german+credit+data/
├── pyproject.toml                # Poetry configuration
├── README.md                     # Package README
└── AGENTS.md                    # This file
```

### 3.2 Core Modules

#### features/binning/
- `base.py`: BaseBinner (abstract class, defines binning interface)
  - **Monotonic support**: `monotonic` parameter (True, False, "ascending", "descending", "auto")
  - `_adjust_monotonicity()`: PAVA-based monotonic adjustment algorithm
  - `MONOTONIC_TRENDS`: Valid monotonic trend constants
- `binner.py`: Binner (unified interface, combines 3 mixins)
  - `BinningResult`: Proxy object for accessing feature binning results
  - Supports `__getitem__` for `binner['feature']` access
- `binner_mixins.py`: BinnerStatsMixin, BinnerIOMixin, BinnerWOEMixin
- `binning_stats.py`: Statistical calculation functions
- `woe_storage.py`: WOE encoder management
- `supervised.py`: ChiMergeBinner, DecisionTreeBinner, OptBinningBinner
- `unsupervised.py`: EqualWidthBinner, EqualFrequencyBinner, KMeansBinner

#### features/selection/
- `selector.py`: FeatureSelector (Unified EDA and filtering: IV, missing rate, correlation)
  - Supports multiple metrics: IV, missing_rate, ks, correlation
  - `corr_matrix`: Feature-to-feature correlation matrix property
  - `select()`: Apply thresholds for feature filtering
  - `report()`: Generate comprehensive analysis report
- `postfilter.py`: PostFilter (PSI, VIF)
- `stepwise.py`: StepwiseSelector (forward, backward, bidirectional)

#### features/analysis/
- `woe_calculator.py`: WOEEncoder (WOE encoding and IV calculation)
- `iv_calculator.py`: calculate_iv function
- `correlation.py`: Correlation matrix and high correlation pairs

#### modeling/
- `logistic.py`: LogisticModel (logistic regression wrapper)
  - Uses statsmodels for detailed statistical output
  - Provides scikit-learn-like interface
  - `to_dict()`: Export model parameters (intercept, coefficients)
  - `get_significant_features()`: Filter features by p-value
- `scorecard.py`: Scorecard (scorecard generation)
  - `from_model()`: Build from fitted model, binner, WOE encoder
  - Supports multiple model types (sklearn, statsmodels, custom)
  - `score()`: Calculate scores for new data

#### metrics/
- `auc.py`: calculate_auc (AUC metric)
- `gini.py`: calculate_gini (Gini coefficient)
- `ks.py`: calculate_ks (Kolmogorov-Smirnov statistic)
- `lift.py`: calculate_lift (Lift metric)
- `psi.py`: calculate_psi (Population Stability Index)
- `vif.py`: calculate_vif (Variance Inflation Factor)

#### pipeline/
- `pipeline.py`: ScorecardPipeline (fluent API for end-to-end workflow)
  - Chainable methods: `.prefilter()`, `.bin()`, `.woe_transform()`, `.postfilter()`, `.stepwise()`, `.build_model()`, `.generate_scorecard()`
  - Access to intermediate results via properties
  - `summary()`: Get pipeline execution summary

#### visualization/
- `binning_viz.py`: Visualization functions
  - `plot_binning_result()`: Binning histogram with bad rate line
  - `plot_iv_ranking()`: IV ranking bar chart
  - `plot_woe_pattern()`: WOE pattern visualization
  - `plot_psi_comparison()`: PSI comparison chart
- `binning.py`: Alternative binning plotting utilities

#### utils/
- `decorators.py`: @requires_fit decorator (unified fitted state check)

#### config.py
Configuration constants (dataclasses):
- `BinningConfig`: DEFAULT_N_BINS, DEFAULT_BUCKETS, DEFAULT_EPSILON, MIN_SAMPLES_LEAF
- `FilteringConfig`: DEFAULT_IV_THRESHOLD, DEFAULT_MISSING_THRESHOLD, DEFAULT_CORR_THRESHOLD, DEFAULT_PSI_THRESHOLD, DEFAULT_VIF_THRESHOLD
- `ModelingConfig`: DEFAULT_P_ENTER, DEFAULT_P_REMOVE, DEFAULT_CLASSIFICATION_THRESHOLD
- `ScorecardConfig`: DEFAULT_PDO, DEFAULT_BASE_SCORE, DEFAULT_BASE_ODDS

## 4. Development Standards

### 4.1 Naming Conventions

**Strict conventions:**

- **Class names**: PascalCase
  ```python
  class Binner:
      pass

  class WOEEncoder:
      pass
  ```

- **Function/Method names**: snake_case
  ```python
  def calculate_iv():
      pass

  def fit_transform():
      pass

  def _update_all_stats():
      pass
  ```

- **Constants**: UPPER_CASE
  ```python
  DEFAULT_N_BINS = 5
  DEFAULT_IV_THRESHOLD = 0.02
  MONOTONIC_TRENDS = frozenset(["ascending", "descending", "auto"])
  ```

- **Private members**: Prefix with underscore
  ```python
  self._X = X.copy()
  self._fit_splits()
  ```

- **❌ Prohibited**: camelCase
  ```python
  # ❌ Incorrect
  def calculateIV():
      pass

  # ✅ Correct
  def calculate_iv():
      pass
  ```

### 4.2 Type Annotations

**Use typing module:**

- **Must annotate return types**
  ```python
  def fit(self, X: pd.DataFrame, y: pd.Series) -> "Binner":
      pass
  ```

- **Optional for optional parameters**
  ```python
  def fit(self, X: pd.Series, y: Optional[pd.Series] = None) -> "BaseBinner":
      pass
  ```

- **Final for constants**
  ```python
  DEFAULT_N_BINS: Final[int] = 5
  ```

- **Avoid Any** unless absolutely necessary

### 4.3 Docstrings

**Use Google-style consistently:**

```python
def calculate_iv(
    df: pd.DataFrame,
    target: str,
    feature: str,
    buckets: int = 10,
    epsilon: float = 1e-8,
) -> Dict[str, Union[float, pd.DataFrame]]:
    """Calculate Information Value (IV) for a feature.

    Args:
        df: Input DataFrame.
        target: Target column name (binary 0/1).
        feature: Feature column name.
        buckets: Number of buckets for numerical features.
        epsilon: Small constant to avoid division by zero or log(0).

    Returns:
        Dict containing 'iv' (float) and 'woe_table' (pd.DataFrame).
    """
```

### 4.4 Code Formatting

- Use `black` for code formatting
- Use `isort` for import sorting
- Pass `flake8` checks
- Pre-commit automation enabled

### 4.5 Configuration Usage

**Use constants from config.py to avoid magic numbers:**

```python
# ✅ Recommended
from newt.config import BINNING, FILTERING, MODELING, SCORECARD

def __init__(self, n_bins: int = BINNING.DEFAULT_N_BINS):
    self.n_bins = n_bins

# ❌ Avoid
def __init__(self, n_bins: int = 5):
    self.n_bins = n_bins
```

## 5. Testing Standards

### 5.1 Test Structure

```
tests/
├── conftest.py                   # Shared fixtures
├── unit/                         # Unit tests
│   ├── features/
│   │   ├── binning/
│   │   ├── analysis/
│   │   └── selection/
│   ├── metrics/
│   ├── modeling/
│   ├── pipeline/
│   ├── utils/
│   └── visualization/
├── cross_val/                    # Cross-validation tests
│   └── compare_with_toad.py
└── integration/                   # Integration tests (to be added)
    ├── test_performance.py
    └── test_pipeline.py
```

### 5.2 Test Naming

- Use `test_<function>_<scenario>` format
- Use parametrized tests with `@pytest.mark.parametrize`

### 5.3 Coverage Target

- **Target**: 70% coverage
- New features must have tests with >60% coverage
- Bug fixes must include regression tests

## 6. Git Standards

### 6.1 Branch Strategy

- `main`: Primary branch
- `feature/<name>`: Feature development branches
- `bugfix/<num>`: Bugfix branches

### 6.2 Commit Messages

- Format: `<type>(<scope>): <subject>`
- Types: `feat`, `fix`, `docs`, `refactor`, `test`
- Examples:
  - `feat(binning): add monotonic binning support`
  - `fix(woe): handle missing values in transform`
  - `docs: update AGENTS.md`

### 6.3 PR Process

1. Create branch from `main`
2. Develop and test
3. Pass CI checks
4. Create Pull Request
5. Code review
6. Merge to `main`

## 7. Common Commands

```bash
# Install dependencies
python -m poetry install

# Format code
python -m poetry run black .
python -m poetry run isort .

# Run linter
python -m poetry run flake8 src tests

# Run all tests
python -m poetry run pytest

# Run specific test
python -m poetry run pytest tests/unit/features/binning/test_binning.py

# Run tests with coverage
python -m poetry run pytest --cov=src/newt --cov-report=html

# Run pre-commit checks
pre-commit run --all-files
```

## 8. CI/CD

**GitHub Actions Workflows:**

- `.github/workflows/ci.yml`:
  - Runs on push to main/master or PR
  - Tests Python 3.8, 3.9, 3.10
  - Runs flake8 and pytest with coverage

- `.github/workflows/release.yml`:
  - Runs on GitHub release creation
  - Publishes to PyPI

## 9. Key Design Patterns

### 9.1 Binning Monotonicity

**Monotonic Binning Support:**

- **Parameter**: `monotonic` (bool or str: "ascending", "descending", "auto")
- **Algorithm**: PAVA (Pool Adjacent Violators) with greedy merging
- **Implementation**: `_adjust_monotonicity()` in `BaseBinner`
- **Usage**:
  ```python
  binner = Binner()
  binner.fit(X, y, method='chi', monotonic=True)              # Auto-detect
  binner.fit(X, y, method='dt', monotonic='ascending')      # Force ascending
  binner.fit(X, y, method='opt', monotonic='descending')    # Force descending
  ```

### 9.2 Scorecard Model Support

**Multiple Model Types Supported:**

- Custom `LogisticModel` wrapper (via `to_dict()`)
- Dictionary format: `{'intercept': float, 'coefficients': {feature: coef}}`
- scikit-learn models (LogisticRegression, SGDClassifier, etc.)
- statsmodels results (Logit, GLM fit results)

**Usage:**
```python
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Sklearn model
lr = LogisticRegression()
lr.fit(X_woe, y)
scorecard.from_model(lr, binner, woe_encoder)

# Statsmodels model
model = sm.Logit(y, X).fit()
scorecard.from_model(model, binner, woe_encoder)

# Dictionary format
model_dict = {'intercept': -2.5, 'coefficients': {'age': 0.3}}
scorecard.from_model(model_dict, binner, woe_encoder)
```

### 9.3 Binning Result Access Pattern

**Convenient Feature Access:**

```python
binner = Binner()
binner.fit(X, y)

# Access via __getitem__
result = binner['age']        # Returns BinningResult
print(result.stats)           # View statistics
result.plot()                 # Plot binning
print(result.woe_map())      # Get WOE mapping

# Loop through features
for feat in binner:
    print(feat)
```

### 9.4 Visualization Integration

**Plotting Functions:**

```python
from newt.visualization import (
    plot_binning_result,
    plot_iv_ranking,
    plot_woe_pattern,
    plot_psi_comparison
)

# Binning plot
fig = plot_binning_result(binner, X, y, 'age')

# IV ranking
iv_dict = selector.eda_summary_.set_index('feature')['iv'].to_dict()
fig = plot_iv_ranking(iv_dict)

# WOE pattern
fig = plot_woe_pattern(woe_encoder['age'], 'age')
```

## 10. Development Guidelines

### 10.1 When Adding New Features

1. **Core functionality**: Add to `src/newt/` appropriate module
2. **Exports**: Update `__init__.py` files to export new classes/functions
3. **Configuration**: Add constants to `config.py` if needed
4. **Documentation**: Update user guides in `docs/`
5. **Tests**: Add unit tests to `tests/unit/` appropriate directory
6. **AGENTS.md**: Update this file with new structure/info

### 10.2 When Modifying Core Components

1. **Maintain backward compatibility** for existing APIs
2. **Update tests** to cover new functionality
3. **Run linting**: `python -m poetry run flake8 src tests`
4. **Run tests**: `python -m poetry run pytest --cov=src/newt`
5. **Check coverage**: Ensure >70% coverage target maintained

### 10.3 Documentation Updates

When adding new features:
1. Update `docs/user_guide.md` (English)
2. Update `docs/user_guide_zh.md` (Chinese)
3. Update `README.md` if API changes
4. Add example in `examples/` notebooks if applicable

## 11. Dependencies

**Key External Dependencies:**

- **Core**: pandas, numpy
- **Scientific**: scipy, scikit-learn, statsmodels
- **Binning**: optbinning (optional, for optimal binning)
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest, pytest-cov, coverage
- **Code Quality**: black, isort, flake8

**Note**: `optbinning` requires additional dependencies:
- ortools: <9.12
- cvxpy: >=1.3,<1.5

## 12. Performance Considerations

### 12.1 Memory Efficiency

- **WOE storage**: Uses `WOEStorage` class to manage multiple encoders
- **Binning**: Binner stores splits (lightweight) rather than full data
- **Visualization**: Avoid storing large datasets in memory during plotting

### 12.2 Computational Efficiency

- **Monotonic adjustment**: Greedy merging with early termination
- **Feature selection**: Vectorized operations in pandas
- **Scorecard scoring**: Direct mapping for O(n) scoring on new data

### 12.3 Scalability

- **Large datasets**: Use `n_bins` to control complexity
- **High-dimensional features**: Pre-filtering before expensive operations
- **Parallel processing**: Not yet implemented (future enhancement)
