# Rust Wheel Distribution Design

## Summary

Newt currently ships as a pure Python wheel. The package includes the Python wrapper for batch IV calculation, but it does not ship the compiled Rust extension. As a result, users who install the published wheel cannot use the Rust-backed IV path immediately.

This design changes Newt into a proper mixed Python/Rust package that ships prebuilt wheels for the supported operating systems, architectures, and Python versions. The first release target is GitHub Releases. PyPI support will be added later on top of the same wheel build pipeline.

## Goal

After a user installs Newt from an official release artifact, Rust-backed IV calculation works immediately without requiring a local Rust toolchain, `maturin`, or any first-run compilation step.

## Non-Goals

- Publishing to PyPI in this phase
- Supporting unofficial platforms outside the defined matrix
- Keeping the current runtime behavior that silently compiles Rust code on first import
- Replacing the Python IV implementation; it remains as a fallback path for explicit use and testing

## Target Support Matrix

The release pipeline will produce wheels for the following targets:

- `macOS x86_64`
- `macOS arm64`
- `Windows x86_64`
- `Linux x86_64`
- `Linux arm64`
- `Python 3.8+`

The initial implementation should build for `cp38` through the actively supported CPython versions configured in the wheel builder.

## Current State

Newt currently builds with `hatchling` and produces:

- one pure Python wheel
- one source distribution

The current wheel does not contain the compiled Rust extension. The Rust IV path depends on importing `newt_iv_rust` and, if that import fails, running `maturin develop` locally from the source tree. This is acceptable for local development but not for end-user installation from released artifacts.

The current GitHub release workflow also builds on a single Ubuntu runner, which cannot produce the cross-platform wheel set required by the support matrix.

## Chosen Approach

Newt will move to a mixed Python/Rust packaging model built by `maturin`, with multi-platform wheel creation handled by `cibuildwheel`.

This is the recommended approach because:

- it matches the actual package shape: Python package plus Rust extension
- it produces installable binary wheels directly
- it avoids runtime local compilation for end users
- it scales cleanly to GitHub Release assets and later PyPI publishing

## Alternatives Considered

### 1. Keep `hatchling` and manually bolt Rust artifacts into the wheel

This was rejected because it creates a fragile custom packaging path. The project would need to manually coordinate Python packaging, Rust compilation, wheel contents, and platform tagging.

### 2. Keep runtime local compilation

This was rejected because it fails the main requirement. Users would still need Rust toolchains and build dependencies on their own machines.

### 3. Ship source distribution only

This was rejected because source-only distribution pushes all compilation work to users and makes installation quality inconsistent across platforms.

## Packaging Design

### 1. Build Backend

The project build backend will switch from `hatchling` to `maturin`.

The package will be built as a mixed project:

- Python sources stay under `src/newt`
- the Rust extension is compiled and packaged as part of the installed Python package

### 2. Rust Module Placement

The Rust extension will no longer be treated as an external top-level module that happens to live beside the Python package.

Instead, it will be installed inside the Newt package namespace as an internal module, for example:

- `newt._newt_iv_rust`

This gives the package a stable internal import path and ensures the compiled module is part of the wheel contents.

### 3. Python-Side Loading

The Python IV wrapper will stop trying to compile Rust code on first import in installed environments.

The new behavior will be:

- when the official wheel is installed, the Rust extension imports directly
- when the Python fallback is explicitly requested, Newt uses the Python implementation
- if the Rust engine is requested but the compiled extension is missing, Newt raises a clear error instead of attempting hidden local compilation

This keeps production installs predictable and keeps the Python implementation available for testing and fallback use.

### 4. Distribution Artifacts

Each release will contain:

- platform-specific binary wheels for all supported targets
- one source distribution

The wheels become the primary installation artifact for both GitHub-based internal distribution and future PyPI publishing.

## CI/CD Design

### 1. Continuous Integration

The existing CI workflow continues to run unit tests on Linux across the supported Python versions. It remains the fast correctness check.

Additional packaging validation will be added so the repository verifies that the mixed build configuration remains healthy.

### 2. Wheel Build Workflow

A dedicated GitHub Actions workflow will use `cibuildwheel` to build wheels across:

- macOS
- Windows
- Linux

with the required architecture matrix.

This workflow will:

- build release wheels
- run a smoke test against the produced wheel
- confirm the installed package can import the Rust extension
- confirm Rust-backed IV calculation executes successfully in a clean environment

### 3. GitHub Release Publishing

For the first phase, release artifacts will be uploaded to GitHub Releases only.

The release workflow will:

- build all wheels and the source distribution
- attach them to the GitHub Release
- preserve them as downloadable assets for internal users

PyPI publishing will be added later once the wheel pipeline is stable.

## Installation Experience

With this design in place, the normal installation path becomes:

```bash
pip install newt
```

or installation from a GitHub Release wheel:

```bash
pip install newt-<version>-<platform-wheel>.whl
```

The installed package works immediately for Rust-backed IV on supported targets.

## Implementation Plan

### Phase 1. Restructure Packaging

- move the Rust extension to a package-internal module name
- switch the build backend to `maturin`
- configure mixed Python/Rust packaging metadata
- remove installed-environment runtime compilation behavior

### Phase 2. Add Wheel Build Pipeline

- add `cibuildwheel`
- define the platform, architecture, and Python version matrix
- add wheel smoke tests

### Phase 3. Add GitHub Release Distribution

- replace the current single-runner pure Python release flow
- publish all built wheels and the source distribution to GitHub Releases

### Phase 4. Prepare PyPI Follow-Up

- keep the release artifacts and workflow layout compatible with later PyPI publishing
- defer actual PyPI upload until after GitHub Release distribution is stable

## Verification

The solution is complete only when all of the following are true:

- the built wheel actually contains the compiled Rust extension
- installing a released wheel in a clean environment does not trigger local compilation
- `calculate_batch_iv(..., engine="rust")` works immediately after install
- the source distribution still builds correctly
- GitHub Release artifacts include the full wheel set for the supported matrix

## Risks

### 1. Build Matrix Size

The support matrix is large. Build times and release runtime will increase significantly compared with the current single-runner release flow.

### 2. Cross-Platform Rust Packaging Details

Platform-specific wheel tagging, macOS architecture handling, and Linux arm64 builds need careful CI configuration. These are solvable, but they must be verified with real artifacts.

### 3. Python Version Breadth

Supporting Python `3.8+` increases the number of wheels and the chance of platform-specific dependency issues. This is acceptable but should be treated as an explicit maintenance cost.

## Future Work

After GitHub Release distribution is stable:

- add PyPI publishing on top of the same wheel artifacts
- optionally add Trusted Publishing for PyPI
- optionally add a smaller support policy if maintenance cost becomes too high
