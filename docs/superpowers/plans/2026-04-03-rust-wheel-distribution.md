# Rust Wheel Distribution Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Newt as a mixed Python/Rust package whose GitHub Release artifacts include prebuilt wheels that let users use Rust-backed IV immediately after installation.

**Architecture:** Replace the current pure-Python packaging path with a `maturin` mixed-project build that installs the Rust extension inside the `newt` package. Build multi-platform wheels with `cibuildwheel`, verify them in clean environments, and publish the artifacts to GitHub Releases while keeping the existing Python fallback explicit and predictable.

**Tech Stack:** Python, Rust, PyO3, maturin, cibuildwheel, GitHub Actions, uv, pytest

---

### Task 1: Move Packaging To A Mixed Python/Rust Build

**Files:**
- Modify: `pyproject.toml`
- Modify: `rust/newt_iv_rust/Cargo.toml`
- Modify: `rust/newt_iv_rust/src/lib.rs`
- Modify: `src/newt/features/analysis/batch_iv.py`
- Test: `tests/unit/features/analysis/test_batch_iv.py`
- Create: `tests/unit/features/analysis/test_batch_iv_packaging.py`

- [ ] **Step 1: Write the failing packaging test**

```python
from importlib import import_module


def test_batch_iv_uses_package_internal_rust_module():
    module = import_module("newt._newt_iv_rust")
    assert hasattr(module, "calculate_batch_iv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/features/analysis/test_batch_iv_packaging.py::test_batch_iv_uses_package_internal_rust_module -v`

Expected: FAIL because `newt._newt_iv_rust` does not exist yet.

- [ ] **Step 3: Write the failing runtime behavior test**

```python
import pytest

from newt.features.analysis.batch_iv import _load_rust_extension


def test_load_rust_extension_does_not_trigger_local_build_when_missing(monkeypatch):
    monkeypatch.setattr("newt.features.analysis.batch_iv.importlib.import_module", side_effect=ImportError())
    with pytest.raises(ImportError):
        _load_rust_extension()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/unit/features/analysis/test_batch_iv_packaging.py::test_load_rust_extension_does_not_trigger_local_build_when_missing -v`

Expected: FAIL because current code calls `maturin develop`.

- [ ] **Step 5: Update Rust module naming and package config**

Change the package to a mixed `maturin` build:

- switch `[build-system]` from `hatchling.build` to `maturin`
- add `[tool.maturin]` entries for `python-source = "src"` and `module-name = "newt._newt_iv_rust"`
- remove the old Hatch wheel target configuration
- update Rust crate metadata so the extension builds to the same module name

- [ ] **Step 6: Update Python-side Rust loading**

Change `src/newt/features/analysis/batch_iv.py` so:

- Rust engine imports `newt._newt_iv_rust`
- installed environments do not run `maturin develop`
- requesting `engine="rust"` without the compiled extension raises a clear import/runtime error
- `engine="python"` continues to work unchanged

- [ ] **Step 7: Add minimal regression coverage**

Extend the batch IV tests to cover:

- package-internal module import path
- no hidden local compilation attempt in installed-style code paths
- Rust and Python engines still match numerically on the same data

- [ ] **Step 8: Run focused tests**

Run: `uv run pytest tests/unit/features/analysis/test_batch_iv.py tests/unit/features/analysis/test_batch_iv_packaging.py -v`

Expected: PASS.

- [ ] **Step 9: Build local artifacts to verify mixed packaging**

Run: `uv build`

Expected:

- build succeeds with `maturin`
- wheel contains the compiled Rust extension
- source distribution still builds successfully

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml rust/newt_iv_rust/Cargo.toml rust/newt_iv_rust/src/lib.rs src/newt/features/analysis/batch_iv.py tests/unit/features/analysis/test_batch_iv.py tests/unit/features/analysis/test_batch_iv_packaging.py
git commit -m "feat(packaging): build mixed Rust wheels with maturin"
```

### Task 2: Add Install-From-Wheel Verification

**Files:**
- Create: `tests/packaging/test_install_from_wheel.py`
- Modify: `tests/unit/features/analysis/test_batch_iv_packaging.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing wheel verification test**

```python
def test_installed_wheel_can_run_rust_batch_iv():
    ...
```

The test should:

- create or use a built wheel path
- install it into a clean venv
- run a short Python snippet that imports `calculate_batch_iv(..., engine="rust")`
- assert the command exits successfully

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/packaging/test_install_from_wheel.py::test_installed_wheel_can_run_rust_batch_iv -v`

Expected: FAIL until the build and test harness are wired correctly.

- [ ] **Step 3: Implement the wheel smoke-test harness**

Add a packaging test helper that:

- builds or locates the wheel under `dist/`
- creates a temporary virtual environment
- installs the wheel with `pip`
- runs a Python snippet that imports Newt and executes Rust-backed IV

- [ ] **Step 4: Run the packaging test**

Run: `uv run pytest tests/packaging/test_install_from_wheel.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/packaging/test_install_from_wheel.py pyproject.toml tests/unit/features/analysis/test_batch_iv_packaging.py
git commit -m "test(packaging): verify installed wheel uses Rust IV"
```

### Task 3: Build Multi-Platform Wheels In GitHub Actions

**Files:**
- Create: `.github/workflows/build-wheels.yml`
- Modify: `.github/workflows/release.yml`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `docs/user_guide.md`
- Modify: `docs/user_guide_zh.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Write the failing workflow validation test**

Create a lightweight repository test that checks for the expected workflow structure, for example:

```python
def test_release_workflow_references_cibuildwheel():
    content = Path(".github/workflows/build-wheels.yml").read_text()
    assert "cibuildwheel" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/packaging/test_workflows.py::test_release_workflow_references_cibuildwheel -v`

Expected: FAIL because the new workflow file does not exist yet.

- [ ] **Step 3: Add the wheel build workflow**

Create `.github/workflows/build-wheels.yml` that:

- runs on GitHub Actions across macOS, Windows, and Linux
- builds the requested architectures with `cibuildwheel`
- targets Python 3.8+
- uploads wheel artifacts
- runs an installed-wheel smoke test command

- [ ] **Step 4: Update the release workflow**

Change `.github/workflows/release.yml` so it:

- reuses or depends on the built wheel artifacts
- uploads wheels plus sdist to GitHub Releases
- does not publish to PyPI in this phase

- [ ] **Step 5: Keep CI lean but aware of packaging**

Update `.github/workflows/ci.yml` so standard PR CI stays fast while still checking the packaging configuration is valid.

- [ ] **Step 6: Document the new release model**

Update user-facing docs to explain:

- GitHub Releases are the primary binary distribution channel for this phase
- supported OS / architecture / Python matrix
- ordinary users do not need Rust installed when using official wheels

- [ ] **Step 7: Run local validation**

Run:

```bash
uv run pytest tests/packaging/test_workflows.py -v
uv run pytest tests/unit/features/analysis/test_batch_iv.py tests/unit/features/analysis/test_batch_iv_packaging.py tests/packaging/test_install_from_wheel.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/build-wheels.yml .github/workflows/release.yml .github/workflows/ci.yml README.md docs/user_guide.md docs/user_guide_zh.md AGENTS.md tests/packaging/test_workflows.py
git commit -m "feat(ci): build and release Rust wheels on GitHub"
```

### Task 4: End-To-End Artifact Verification

**Files:**
- Modify: `tests/packaging/test_install_from_wheel.py`
- Modify: `README.md`
- Modify: `docs/user_guide.md`
- Modify: `docs/user_guide_zh.md`

- [ ] **Step 1: Build final local artifacts**

Run: `uv build`

Expected: wheels and sdist are produced successfully under `dist/`.

- [ ] **Step 2: Inspect wheel contents**

Run a script or test that opens the built wheel and verifies the Rust extension file is present under the `newt` package.

- [ ] **Step 3: Run clean-environment install verification**

Run the packaging smoke test against the final wheel set and confirm:

- `import newt` works
- `calculate_batch_iv(..., engine="rust")` works
- no local build is triggered

- [ ] **Step 4: Record final install guidance**

Document the exact GitHub Release installation flow in the README and guides.

- [ ] **Step 5: Run full verification**

Run:

```bash
uv run pytest tests/unit/features/analysis/test_batch_iv.py tests/unit/features/analysis/test_batch_iv_packaging.py tests/packaging/test_install_from_wheel.py tests/packaging/test_workflows.py -v
uv build
```

Expected: PASS, with wheel artifacts present and verified.

- [ ] **Step 6: Commit**

```bash
git add tests/packaging/test_install_from_wheel.py README.md docs/user_guide.md docs/user_guide_zh.md
git commit -m "docs(release): document GitHub wheel installation"
```
