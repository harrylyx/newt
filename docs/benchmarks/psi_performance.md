# PSI Engine Benchmark

This benchmark compares scalar `calculate_psi` against `calculate_psi_batch(engine='python')` and `calculate_psi_batch(engine='rust')`.

## Environment

- Generated at: `2026-04-06T03:53:15.075948Z`
- Python: `3.10.19`
- Platform: `macOS-26.2-arm64-arm-64bit`
- CPU count: `10`

## Scenarios

| rows | features | groups | missing_rate | scalar_ms | batch_python_ms | batch_rust_ms | rust_speedup_vs_scalar | python_speedup_vs_scalar | py_batch_max_abs_diff | py_batch_mean_abs_diff | rust_batch_max_abs_diff | rust_batch_mean_abs_diff | avg_scalar_ms_per_feat_group | avg_rust_ms_per_feat_group | peak_rss_mb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100000 | 20 | 6 | 0.00 | 1443.598 | 744.934 | 163.198 | 8.85x | 1.94x | 0.000e+00 | 0.000e+00 | 6.939e-18 | 4.770e-19 | 12.029984 | 1.359984 | 319.91 |
| 100000 | 20 | 6 | 0.05 | 1345.450 | 617.339 | 193.774 | 6.94x | 2.18x | 0.000e+00 | 0.000e+00 | 1.388e-17 | 6.939e-19 | 11.212087 | 1.614786 | 425.50 |
| 100000 | 20 | 6 | 0.20 | 1179.149 | 534.981 | 150.942 | 7.81x | 2.20x | 0.000e+00 | 0.000e+00 | 2.776e-17 | 2.082e-18 | 9.826244 | 1.257850 | 431.47 |

## Reproduce

```bash
newt-benchmark-psi
```

Use CLI options (`--rows`, `--features`, `--groups`, `--missing-rates`) to customize scenario grid.
