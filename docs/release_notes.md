# Newt Release Notes (Recent)

This page summarizes the recent production changes from `v0.1.1` to `v0.1.4`.

## v0.1.4 (2026-04-04)

### CI/CD and release pipeline

- Linux wheel builds were split into two native jobs:
  - `Linux x86_64` on `ubuntu-latest`
  - `Linux aarch64` on `ubuntu-24.04-arm`
- QEMU-based Linux emulation was removed from both `Build Wheels` and `Release` workflows.
- Wheel matrix remains `cp38` through `cp312` (`3.8.5` to `3.12.x`).

### Why this release

- Previous Linux ARM builds were limited by cross-architecture emulation behavior.
- Native ARM runners reduce architecture mismatch risk and make failures easier to isolate.

## v0.1.3 (2026-04-04)

### Packaging and workflow hardening

- Introduced the intermediate workflow fix to unblock Linux `aarch64` wheel jobs.
- Carried forward Python support contract `>=3.8.5,<3.13` and `cp38-cp312` wheel builds.

## v0.1.2 (2026-04-04)

### Reporting fixes

- Report output was normalized across sheets.
- Score-direction handling and table layout issues were corrected in report generation.
- Benchmark and report docs were synchronized with the actual runtime environments.

## v0.1.1 (2026-04-04)

### IV engine and metric alignment

- `calculate_iv` and batch IV switched to Rust as the default engine.
- Python engine remains available when explicitly requested (`engine="python"`).
- Newt IV computation was aligned with `toad`-compatible smoothing behavior.
- Rust extension gained categorical IV support so default Rust mode works for both numeric and categorical features.

### Benchmark and compatibility policy

- Benchmark now runs directly in `.venv-benchmark-3.10` with `toad==0.1.5`.
- Worker fallback virtual environments were removed from benchmark flow.
- Official Python support window is now explicit: `3.8.5` through `3.12.x`.

## Current baseline after v0.1.4

- **Python support (core):** `>=3.8.5,<3.13`
- **Wheel targets:** `cp38`, `cp39`, `cp310`, `cp311`, `cp312`
- **Default local environments:**
  - `.venv` (Python `3.8.5`) for development
  - `.venv-benchmark-3.10` (Python `3.10.19`) for benchmark
- **IV default:** Rust engine for both single-feature and batch IV
- **Benchmark parity goal:** Newt IV vs toad IV should be near-zero diff on the prepared benchmark input
