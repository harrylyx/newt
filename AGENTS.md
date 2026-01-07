# Newt - Agent Development Guide

## 1. Project Overview

Newt is a lightweight Python toolkit for efficient feature analysis and statistical diagnostics in credit risk modeling.

- **Python Version**: >=3.8.1
- **Dependency Management**: Poetry
- **Core Features**:
  - 6 binning algorithms (ChiMerge, Decision Tree, K-Means, Equal Frequency, Equal Width, Optimal)
  - WOE/IV analysis and encoding
  - Feature selection (pre-filtering, post-filtering, stepwise regression)
  - Logistic regression modeling
  - Scorecard generation
  - Pipeline-style workflow

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
├── src/newt/              # Source code
│   ├── features/
│   │   ├── binning/      # Binning (Mixin pattern)
│   │   ├── selection/    # Feature selection
│   │   └── analysis/     # WOE/IV analysis
│   ├── modeling/         # Modeling
│   ├── metrics/          # Evaluation metrics
│   ├── pipeline/         # Pipeline
│   ├── utils/            # Utilities (@requires_fit decorator)
│   └── config.py        # Configuration constants
├── tests/               # Tests
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── conftest.py     # Shared fixtures
├── docs/                # Documentation
└── examples/            # Examples (currently empty)
```

### 3.2 Core Modules

#### features/binning/
- `base.py`: BaseBinner (abstract class, defines binning interface)
- `binner.py`: Binner (unified interface, combines 3 mixins)
- `binner_mixins.py`: BinnerStatsMixin, BinnerIOMixin, BinnerWOEMixin
- `binning_stats.py`: Statistical calculation functions
- `woe_storage.py`: WOE encoder management
- `supervised.py`: ChiMergeBinner, DecisionTreeBinner, OptBinningBinner
- `unsupervised.py`: EqualWidthBinner, EqualFrequencyBinner, KMeansBinner

#### features/selection/
- `selector.py`: FeatureSelector (Unified EDA and filtering: IV, missing rate, correlation)
- `postfilter.py`: PostFilter (PSI, VIF)
- `stepwise.py`: StepwiseSelector (forward, backward, bidirectional)

#### features/analysis/
- `woe_calculator.py`: WOEEncoder (WOE encoding and IV calculation)
- `iv_calculator.py`: calculate_iv function
- `correlation.py`: Correlation matrix and high correlation pairs

#### modeling/
- `logistic.py`: LogisticModel (logistic regression wrapper)
- `scorecard.py`: Scorecard (scorecard generation)

#### metrics/
- `auc.py`: calculate_auc (AUC metric)
- `gini.py`: calculate_gini (Gini coefficient)
- `ks.py`: calculate_ks (Kolmogorov-Smirnov statistic)
- `lift.py`: calculate_lift (Lift metric)
- `psi.py`: calculate_psi (Population Stability Index)
- `vif.py`: calculate_vif (Variance Inflation Factor)

#### pipeline/
- `pipeline.py`: ScorecardPipeline (fluent API for end-to-end workflow)

#### utils/
- `decorators.py`: @requires_fit decorator (unified fitted state check)

#### config.py
Configuration constants (dataclasses):
- `BinningConfig`: DEFAULT_N_BINS, DEFAULT_EPSILON, MIN_SAMPLES_LEAF
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
from newt.config import BINNING

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
├── conftest.py              # Shared fixtures
├── unit/                   # Unit tests
│   ├── features/
│   │   ├── binning/
│   │   ├── analysis/
│   │   └── selection/
│   ├── metrics/
│   └── ...
└── integration/            # Integration tests
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
  - `feat(binning): add new binning method`
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
