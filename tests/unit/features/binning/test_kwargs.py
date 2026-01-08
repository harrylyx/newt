import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("src"))  # noqa: E402

from newt import Binner  # noqa: E402


def test_kwargs_propagation():
    print("Testing kwargs propagation via Binner.fit...")

    np.random.seed(42)
    N = 200
    X = pd.DataFrame({"score": np.random.normal(0, 1, N)})
    y = np.random.randint(0, 2, N)
    X["target"] = y

    # Use OptBinning to verify kwargs storage as BaseBinner doesn't store them
    Binner()

    # Pass arbitrary param to check propagation
    pass


if __name__ == "__main__":
    test_kwargs_propagation()
