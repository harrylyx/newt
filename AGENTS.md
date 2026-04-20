# Newt - Agent Development Guide (Compressed)

This guide defines the core rules, standards, and patterns for developing Newt. **Follow these strictly.**

## 1. Development Environment (CRITICAL)

**Tooling**: Use `uv` exclusively for environment and dependency management.

- `.venv`: Python 3.8.5. Core dev, tests, linting.
- `.venv-benchmark-3.10`: Python 3.10.19. Benchmarks and `toad` parity.

**Rules**:
- **No extra venvs**: Keep only these two repo-local environments.
- **Use `uv run`**: Avoid bare `python` or `pytest`.
- **Sync Command**: `UV_PROJECT_ENVIRONMENT=.venv-benchmark-3.10 UV_PYTHON_INSTALL_DIR=.uv-python uv sync --python 3.10.19 --group dev --group benchmark --frozen`

## 2. Architecture & Directory Structure

```
newt/
├── src/newt/              # Core Source
│   ├── features/          # Binning (Mixin pattern), Selection, Analysis (WOE/IV)
│   ├── modeling/          # LogisticModel, Scorecard (Statsmodels wrappers)
│   ├── metrics/           # AUC, KS, Gini, PSI, VIF
│   ├── reporting/         # Excel Report Engine (xlsxwriter)
│   ├── results/           # Stable result objects for downstream
│   ├── visualization/     # Matplotlib/Seaborn plots
│   └── config.py          # Central constants (dataclasses)
├── benchmarks/            # Performance scripts (Root directory)
├── tests/                 # Unit, Integration, Packaging tests
└── rust/newt_native/      # High-performance Native engine (Rust)
```

**Key API Patterns**:
- **Binner**: Uses a Mixin pattern (`BinnerStatsMixin`, etc.) and `__getitem__` for `binner['feat']` access.
- **Rust First**: Core performance paths (Binning, Selection, IV, PSI) default to Rust with Python fallback.

## 3. Standards

### 3.1 Naming & Types
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case` (Private: `_prefix`)
- **Constants**: `UPPER_CASE`
- **Typing**: Mandatory return type annotations. Avoid `Any`.
- **Formatting**: `black`, `isort`, `flake8`.

### 3.2 Documentation
Use **Google-style** docstrings consistently.
```python
def calculate_iv(df: pd.DataFrame, target: str, feature: str) -> Dict[str, Union[float, pd.DataFrame]]:
    """Calculate IV for a feature.
    Args:
        df: Input DataFrame.
    Returns:
        Dict with 'iv' and 'woe_table'.
    """
```

## 4. Fundamental Patterns

### 4.1 Binning Monotonicity (PAVA)
- **Parameter**: `monotonic` (bool, "ascending", "descending", "auto").
- **Implementation**: `_adjust_monotonicity()` in `BaseBinner`. Greedy merging logic.

### 4.2 Scorecard Model Support
- Supports `LogisticModel`, `sklearn` models, `statsmodels` results, and raw `dict` formats (`{'intercept': float, 'coefficients': {feat: coef}}`).
- Build scorecards with `Scorecard.from_model(model, binner)` after fitting the model on the binner's transformed output.
- Treat `Binner` as the source of truth for WOE state; use `binner.get_woe_map(feature)` and `binner.export()` rather than maintaining separate WOE storage objects.

### 4.3 Results Access
```python
binner = Binner().fit(X, y)
res = binner['age']  # BinningResult
res.stats            # pd.DataFrame
res.plot()           # Visualization
```

For batch WOE workflows, prefer `binner.woe_transform(X)` over hand-written encoder loops.

## 5. Development Workflow

### 5.1 Common Commands
```bash
uv sync --group dev                # Install
uv run black .                     # Format
uv run flake8 src tests            # Lint
uv run pytest                      # Test
uv run pytest --cov=src/newt      # Coverage (Target: 70%)
```

### 5.2 Release & CI/CD
- **Version**: Managed in `pyproject.toml`, `src/newt/__init__.py`, `uv.lock`.
- **CI**: GitHub Actions runs on py3.8, 3.9, 3.10.
- **Wheels**: Built for `cp38`-`cp312` across Windows, macOS (arm64), Linux (x86_64, aarch64).

### 5.3 Release Flow
1. Bump the version in `pyproject.toml` and `src/newt/__init__.py`, then let `uv` refresh `uv.lock` if needed.
2. Run the relevant tests and lint checks before release.
3. Commit the version bump and code changes.
4. Create a Git tag for the release, and include the concrete change summary in the tag message.
5. Create a GitHub Release from that tag with the same change summary.
6. Push the branch and the tag to `origin`, then verify the release page exists.

## 6. Guidelines
1. **Adding Features**: Update `src/`, `__init__.py`, `config.py`, `tests/`, and `docs/`.
2. **Backward Compatibility**: Ensure APIs remain stable.
3. **Performance**: Avoid large copies in loops. Use Rust for batch paths.
4. **Clean Code**: Remove scratch files and throwaway venvs after use.
