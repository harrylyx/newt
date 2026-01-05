import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('src'))
from newt.features.binning.supervised import ChiMergeBinner, OptBinningBinner

def debug_chimerge():
    print("Debugging ChiMergeBinner...")
    np.random.seed(42)
    X = pd.Series(np.random.normal(0, 1, 100))
    y = pd.Series(np.random.randint(0, 2, 100))
    
    binner = ChiMergeBinner(n_bins=5)
    try:
        splits = binner._fit_splits(X, y)
        print(f"ChiMerge Splits: {splits}")
    except Exception as e:
        print(f"ChiMerge Failed: {e}")
        import traceback
        traceback.print_exc()

def debug_optbinning():
    print("\nDebugging OptBinningBinner...")
    np.random.seed(42)
    X = pd.Series(np.random.normal(0, 1, 100))
    y = pd.Series(np.random.randint(0, 2, 100))
    
    binner = OptBinningBinner(n_bins=5)
    try:
        splits = binner._fit_splits(X, y)
        print(f"OptBinning Splits: {splits}")
    except Exception as e:
        print(f"OptBinning Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chimerge()
    debug_optbinning()
