Welcome to the Newt repository. This file contains the main points for new contributors.

## Repository overview

- **Source code**: `src/newt/` contains the implementation.
    - `features/binning`: Binning algorithms (ChiMerge, DecisionTree, KMeans, etc.) and `Combiner`.
    - `features/analysis`: Analysis tools like `WOEEncoder`.
    - `visualization`: Visualization tools like `plot_binning`.
    - `metrics`: Statistical metrics implementation.
- **Tests**: `tests/`.
- **Examples**: under `examples/`.
- **Documentation**: markdown pages live in `docs/` (e.g., `user_guide.md`, `binning_guide.md`).
- **Configuration**: `pyproject.toml` (Poetry), `.pre-commit-config.yaml` (Linters).

## Local workflow

This project uses [Poetry](https://python-poetry.org/) for dependency management and [pre-commit](https://pre-commit.com/) for code quality.

1. **Install Dependencies**:

   ```bash
   poetry install
   ```

2. **Format and Lint**:

   We use `black`, `isort`, and `flake8`.

   ```bash
   # Run all checks via pre-commit
   pre-commit run --all-files
   
   # Or run manually
   poetry run black .
   poetry run isort .
   poetry run flake8 .
   ```

3. **Run Tests**:

   ```bash
   poetry run pytest
   ```

4. **Build Documentation**:

   Documentation is in Markdown format in `docs/`.

## Pull request expectations

PRs should use the template located at `.github/PULL_REQUEST_TEMPLATE/pull_request_template.md` (if available). Provide a summary, test plan and issue number if applicable, then check that:

- New tests are added when needed.
- Documentation is updated.
- `pre-commit` passes without errors.
- The full test suite passes.

## Style notes

- **Formatting**: Code must be formatted with `black` and imports sorted with `isort`.
- **Linting**: Code must pass `flake8` checks.
- **Docstrings**: Public classes and methods should have clear docstrings.

## New Features (Recent)

- **Binning**: Unified `Combiner` class for various binning strategies.
- **WOE/IV**: `WOEEncoder` for weight of evidence transformation and information value calculation.
- **Visualization**: `plot_binning` for interactive combo charts (Plotly).