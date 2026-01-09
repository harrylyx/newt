import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("src"))  # noqa: E402

from newt import Binner  # noqa: E402


def test_optbinning_params():
    print("Testing OptBinning parameters via Binner.fit...")

    np.random.seed(42)
    N = 200
    X = pd.DataFrame({"score": np.random.normal(0, 1, N)})
    y = np.random.randint(0, 2, N)
    X["target"] = y

    c = Binner()
    # Pass arbitrary param to check propagation
    # OptBinning accepts 'min_prebin_size'
    try:
        c.fit(X, y="target", method="opt", n_bins=5, min_prebin_size=0.123)

        # Check if key exists in binners_
        if "score" not in c.binners_:
            print("FAILURE: 'score' not found in binners_. Fit likely failed silently.")
            return

        binner = c.binners_["score"]
        print(f"Binner kwargs found: {binner.kwargs}")

        if binner.kwargs.get("min_prebin_size") == 0.123:
            print("SUCCESS: Parameter propagated.")
        else:
            print(
                f"FAILURE: Parameter mismatch. "
                f"Got {binner.kwargs.get('min_prebin_size')}"
            )

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_optbinning_params()
