# Newt vs toad Benchmark

This benchmark validates Newt's `AUC`, `KS`, `IV`, and `PSI` implementations against `toad` using the bundled sample dataset at `examples/data/test_data/all_data.pq`.

For broader release-level context, see the project [README.md](../../README.md).

## What It Checks

- `AUC` and `KS` on `score_p` and `score_old_p`
- `PSI` for:
  - `train -> test`
  - `train -> oot`
  - the score columns above
  - all numeric features
- `IV` for all candidate features
  - numeric columns are first aligned to fixed 10-quantile bins
  - categorical columns are compared directly

The benchmark also records median runtimes for:

- Newt `AUC` and `KS`
- Newt and toad `PSI`
- Newt single-column `IV`
- Newt batch `IV` with the Python engine
- Newt batch `IV` with the Rust engine when available
- toad's `quality(..., iv_only=True)` path

## Run It

Run the benchmark from the benchmark environment:

```bash
./.venv-benchmark-3.10/bin/newt-benchmark
```

## Output

The command writes:

- `out/benchmarks/metric_vs_toad.json`
- `out/benchmarks/metric_vs_toad.md`

The JSON file contains the full machine-readable report. The Markdown file is a short human-readable summary.

## Python and Dependency Notes

- Newt core remains compatible with Python `3.8.5` through `3.12.x`
- The benchmark environment is `.venv-benchmark-3.10`
- That environment includes `toad==0.1.5` through the `benchmark` dependency group
- The benchmark runs `toad` directly in the same environment; no worker venv or Python fallback is used

## Current Comparison Policy

- `AUC` and `KS` compare the public metric functions directly
- `PSI` uses Newt's reference quantile bucket logic on both sides so the comparison measures the metric calculation, not bucket-definition drift
- `IV` uses the same prepared feature frame for both libraries to avoid mixing metric differences with different default binning rules

## Latest Verified Run

Latest benchmark run (generated on `2026-04-04T21:22:44`) from `.venv-benchmark-3.10`:

- IV mean absolute diff: `4.949931178722187e-16`
- IV max absolute diff: `1.4033219031261979e-13`
- `userinfo_24` IV:
  - Newt: `1.727905`
  - toad: `1.727905`

This confirms the current IV implementation is aligned with toad on the prepared
benchmark input.
