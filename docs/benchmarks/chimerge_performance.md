```shell
python benchmarks/chimerge_performance.py \
  --data-path examples/data/test_data/all_data.pq \
  --rows 1000000 \
  --feature-names "thirdparty_info_period2_6,userinfo_14,thirdparty_info_period1_6,userinfo_16,thirdparty_info_period5_1,thirdparty_info_period5_2,thirdparty_info_period1_10,webloginfo_15,thirdparty_info_period4_6,thirdparty_info_period6_1"
```


# ChiMerge Engine Benchmark

Comparing Pure Python implementation vs Rust sequential and Rust parallel engines.

## Environment

- Generated at: `2026-04-09T19:33:30.500297`
- Python: `3.8.5`
- Platform: `macOS-26.2-x86_64-i386-64bit`
- Data Source: `examples/data/test_data/all_data.pq`

## Performance Metrics

| rows | features | n_bins | python_pure_ms | rust_seq_ms | rust_par_ms | rust_vs_py_speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 10 | 5 | 71.11 | 13.84 | 80.94 | 5.14x |
| 1000000 | 10 | 5 | 1432.52 | 694.71 | 2213.20 | 2.06x |

## Reproduce

```bash
python benchmarks/chimerge_performance.py --data-path examples/data/test_data/all_data.pq --rows 1000,1000000 --features 5,10
```
