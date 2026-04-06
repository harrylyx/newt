# PSI Engine Benchmark

This benchmark compares three PSI computation paths:

- Scalar loop: repeated `calculate_psi(...)`
- Batch Python: `calculate_psi_batch(..., engine="python")`
- Batch Rust: `calculate_psi_batch(..., engine="rust")`

The benchmark is synthetic and focuses on throughput and memory behavior under
feature/group batch workloads.

## Run

Use the benchmark environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-benchmark-3.10 UV_PYTHON_INSTALL_DIR=.uv-python uv run newt-benchmark-psi
```

You can customize scenario grids:

```bash
UV_PROJECT_ENVIRONMENT=.venv-benchmark-3.10 UV_PYTHON_INSTALL_DIR=.uv-python uv run newt-benchmark-psi \
  --rows 100000,500000 \
  --features 20,100 \
  --groups 6,12 \
  --missing-rates 0.00,0.05,0.20 \
  --repeat 1 \
  --warmup 1
```

## Outputs

- `out/benchmarks/psi_performance.json`
- `out/benchmarks/psi_performance.md`

The Markdown output includes:

- total runtime per engine path
- batch-vs-scalar consistency (`max/mean abs diff`)
- per feature-group average runtime
- speedup vs scalar loop
- peak RSS memory

## Notes

- `engine="rust"` automatically falls back to Python if the extension is unavailable.
- Use the same machine/profile for before-vs-after comparisons.

## Example Quick Run

Command:

```bash
UV_PROJECT_ENVIRONMENT=.venv-benchmark-3.10 UV_PYTHON_INSTALL_DIR=.uv-python uv run newt-benchmark-psi \
  --rows 5000 \
  --features 5 \
  --groups 3 \
  --missing-rates 0.00,0.20 \
  --repeat 1 \
  --warmup 0
```

Sample output (`out/benchmarks/psi_performance.md`):

| rows | features | groups | missing_rate | scalar_ms | batch_python_ms | batch_rust_ms | rust_speedup_vs_scalar | python_speedup_vs_scalar | py_batch_max_abs_diff | py_batch_mean_abs_diff | rust_batch_max_abs_diff | rust_batch_mean_abs_diff | avg_scalar_ms_per_feat_group | avg_rust_ms_per_feat_group | peak_rss_mb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20000 | 10 | 4 | 0.00 | 121.794 | 76.480 | 443.767 | 0.27x | 1.59x | 0.000e+00 | 0.000e+00 | 3.469e-18 | 2.602e-19 | 3.044838 | 11.094166 | 189.96 |
| 20000 | 10 | 4 | 0.20 | 84.817 | 54.088 | 418.689 | 0.20x | 1.57x | 0.000e+00 | 0.000e+00 | 2.776e-17 | 4.163e-18 | 2.120414 | 10.467213 | 192.07 |
