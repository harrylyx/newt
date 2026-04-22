# Stepwise Selection Performance

`StepwiseSelector` includes a high-performance Rust engine that leverages **parallel processing** to test candidate features simultaneously. This provides a massive speedup over traditional serial implementations (like `statsmodels`).

## Key Improvements

1.  **Parallel Fitting**: Utilizes all available CPU cores via `Rayon` in Rust.
2.  **Zero Python Overhead**: The entire inner loop (fitting hundreds of candidate models) happens inside the native Rust extension.
3.  **Numerical Parity**: Guaranteed 100% consistency with `statsmodels` for AIC, BIC, and p-values.

## Benchmark Results

Test Environment:
- **Rows**: 10,000
- **Features**: 150 (32 selected)
- **CPU**: 10 Cores (macOS)

| Engine | Execution Mode | Time (s) | Speedup | Result Parity |
| :--- | :--- | :--- | :--- | :--- |
| Python (statsmodels) | Serial | 185.86 | 1.0x | - |
| **Rust (Newt)** | **Parallel** | **8.02** | **23.2x** | **100% Match** |

## Usage

By default, `StepwiseSelector` uses `engine='auto'` (prefer Rust; fallback to Python when Rust is unavailable).

```python
from newt.features.selection import StepwiseSelector

# Uses auto engine by default (prefer Rust)
selector = StepwiseSelector(
    direction='forward',
    criterion='aic',
    engine='auto',
    verbose=True
)

selector.fit(X_transformed, y)
```

For debugging or comparing with legacy results, you can manually switch back to the Python engine:

```python
# Fallback to statsmodels engine
selector = StepwiseSelector(engine='python')
```

## Reproducibility

You can run your own benchmark using the script provided in the repository:

```bash
uv run python benchmarks/stepwise_benchmark.py --samples 10000 --features 150
```
