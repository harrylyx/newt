# Newt vs toad Benchmark

This benchmark validates Newt's `AUC`, `KS`, `IV`, and `PSI` implementations against `toad` using the bundled sample dataset at `examples/data/test_data/all_data.pq`.

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

Use the default project environment:

```bash
uv run newt-benchmark
```

If you want to run Newt itself under Python `3.12`, sync the environment first:

```bash
uv sync --group dev --python 3.12
uv run newt-benchmark
```

## Output

The command writes:

- `out/benchmarks/metric_vs_toad.json`
- `out/benchmarks/metric_vs_toad.md`

The JSON file contains the full machine-readable report. The Markdown file is a short human-readable summary.

## Python and Dependency Notes

- Newt core remains compatible with Python `3.8+`
- The benchmark prefers Python `3.12` for the isolated toad worker
- If `toad` cannot be installed under Python `3.12`, the benchmark automatically falls back to Python `3.10` and records the failed attempt in the report metadata
- The optional `optbinning` dependency stack stays pinned to Python `< 3.12` in project metadata
  - on the current `macOS arm64` environment, `optbinning + ortools + cvxpy` does not pass a clean Python `3.12` smoke test

## Current Comparison Policy

- `AUC` and `KS` compare the public metric functions directly
- `PSI` uses Newt's reference quantile bucket logic on both sides so the comparison measures the metric calculation, not bucket-definition drift
- `IV` uses the same prepared feature frame for both libraries to avoid mixing metric differences with different default binning rules
