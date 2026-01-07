import os
import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath("src"))

from newt import Binner  # noqa: E402


def test_binning_methods():
    # 1. Generate Synthetic Data
    np.random.seed(42)
    N = 1000
    X = pd.DataFrame(
        {"score": np.random.normal(0, 1, N), "age": np.random.randint(20, 80, N)}
    )
    # Target related to score
    y_prob = 1 / (1 + np.exp(-(X["score"] * 2 + np.random.normal(0, 0.5, N))))
    y = (y_prob > 0.5).astype(int)
    X["target"] = y

    print("Data generated.")

    # 2. Test FastChiMerge (method='chi')
    print("\nTesting FastChiMerge (method='chi')...")
    c_chi = Binner()
    try:
        c_chi.fit(X, y="target", method="chi", n_bins=5, cols=["score"])
        splits = c_chi.rules_["score"]
        print(f"Chi splits: {splits}")

        # Verify monotonicity (optional but good)
        df_trans = c_chi.transform(X)
        df_chk = pd.DataFrame({"bin": df_trans["score"], "y": y})
        bad_rate = df_chk.groupby("bin")["y"].mean()
        print("Chi Bad Rates per bin:")
        print(bad_rate)

    except Exception as e:
        print(f"FAILED ChiMerge: {e}")
        import traceback

        traceback.print_exc()

    # 3. Test OptBinning (method='opt')
    print("\nTesting OptBinning (method='opt')...")
    c_opt = Binner()
    try:
        c_opt.fit(X, y="target", method="opt", n_bins=5, cols=["score"])
        splits = c_opt.rules_["score"]
        print(f"Opt splits: {splits}")

        df_trans = c_opt.transform(X)
        df_chk = pd.DataFrame({"bin": df_trans["score"], "y": y})
        bad_rate = df_chk.groupby("bin")["y"].mean()
        print("Opt Bad Rates per bin:")
        print(bad_rate)

    except Exception as e:
        print(f"FAILED OptBinning: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_binning_methods()
