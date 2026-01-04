import numpy as np
import pandas as pd
from src.credit_risk.metrics import (
    calculate_ks,
    calculate_ks_fast,
    calculate_auc,
    calculate_lift,
    calculate_psi,
    calculate_gini
)

def test_metrics():
    print("Generating synthetic data...")
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    # Generate probabilities correlated with y_true
    y_prob = np.random.rand(n_samples)
    y_prob = y_prob + y_true * 0.5
    y_prob = y_prob / y_prob.max()
    
    print("\nTesting KS...")
    ks = calculate_ks(y_true, y_prob)
    ks_fast = calculate_ks_fast(y_true, y_prob)
    print(f"KS (scipy): {ks:.4f}")
    print(f"KS (fast): {ks_fast:.4f}")
    assert abs(ks - ks_fast) < 1e-4
    
    print("\nTesting AUC...")
    auc = calculate_auc(y_true, y_prob)
    print(f"AUC: {auc:.4f}")
    
    print("\nTesting Gini...")
    gini = calculate_gini(y_true, y_prob)
    print(f"Gini: {gini:.4f}")
    assert abs(gini - (2*auc - 1)) < 1e-9
    
    print("\nTesting Lift...")
    lift_df = calculate_lift(y_true, y_prob)
    print("Lift table head:")
    print(lift_df.head(3))
    
    print("\nTesting PSI...")
    expected = np.random.rand(1000)
    actual = np.random.rand(1000) * 1.1 # slightly shifted
    psi = calculate_psi(expected, actual)
    print(f"PSI: {psi:.4f}")

if __name__ == "__main__":
    try:
        test_metrics()
        print("\nAll checks passed!")
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Please ensure dependencies are installed (numpy, pandas, scikit-learn, scipy).")
    except Exception as e:
        print(f"An error occurred: {e}")
