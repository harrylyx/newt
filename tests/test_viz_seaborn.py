import pandas as pd
import numpy as np
import sys
import os
import matplotlib.figure
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath('src'))

from newt import Binner
from newt.visualization import plot_binning

def test_viz_seaborn():
    # Generate synthetic data
    np.random.seed(42)
    N = 1000
    X = pd.DataFrame({
        'score': np.random.normal(0, 1, N),
    })
    # Target related to score
    y_prob = 1 / (1 + np.exp(-(X['score'] * 0.5 + np.random.normal(0, 0.5, N))))
    y = (y_prob > 0.5).astype(int)
    X['target'] = y
    
    # 1. Fit Binner
    c = Binner()
    c.fit(X, y='target', method='chi', n_bins=5)
    
    # 2. Plot
    print("Generating plot...")
    try:
        fig = plot_binning(
            combiner=c,
            data=X,
            feature='score',
            target='target',
            decimals=3,
            bar_mode='total_dist'
        )
        
        # Verify type
        if isinstance(fig, matplotlib.figure.Figure):
            print("Success: Return type is matplotlib.figure.Figure")
        else:
            print(f"Failure: Return type is {type(fig)}")
            return

        # Save to PNG
        output_file = 'test_seaborn_plot.png'
        fig.savefig(output_file)
        print(f"Plot saved to {output_file}")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_viz_seaborn()
