import os
import sys
import warnings

import matplotlib.figure
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath("src"))

from newt import Binner  # noqa: E402
from newt.results import BinningPlotData  # noqa: E402
from newt.visualization import plot_binning_result as plot_binning  # noqa: E402
from newt.visualization.binning import plot_binning as plot_binning_legacy  # noqa: E402


def test_viz_seaborn():
    # Generate synthetic data
    np.random.seed(42)
    N = 1000
    X = pd.DataFrame(
        {
            "score": np.random.normal(0, 1, N),
        }
    )
    # Target related to score
    y_prob = 1 / (1 + np.exp(-(X["score"] * 0.5 + np.random.normal(0, 0.5, N))))
    y = (y_prob > 0.5).astype(int)
    X["target"] = y

    # 1. Fit Binner
    c = Binner()
    c.fit(X, y="target", method="chi", n_bins=5)

    # 2. Plot
    print("Generating plot...")
    try:
        fig = plot_binning(
            combiner=c,
            data=X,
            feature="score",
            target="target",
            decimals=3,
            bar_mode="total_dist",
        )

        # Verify type
        if isinstance(fig, matplotlib.figure.Figure):
            print("Success: Return type is matplotlib.figure.Figure")
        else:
            print(f"Failure: Return type is {type(fig)}")
            return

        # Save to PNG
        output_file = "test_seaborn_plot.png"
        fig.savefig(output_file)
        print(f"Plot saved to {output_file}")

    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_viz_seaborn()


def test_legacy_plot_binning_emits_deprecation_warning():
    X = pd.DataFrame(
        {
            "score": np.linspace(0, 1, 20),
            "target": [0] * 10 + [1] * 10,
        }
    )
    binner = Binner()
    binner.fit(X, y="target", method="step", n_bins=3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = plot_binning_legacy(
            combiner=binner,
            data=X,
            feature="score",
            target="target",
        )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert any(item.category is DeprecationWarning for item in caught)


def test_plot_binning_result_accepts_plot_data():
    X = pd.DataFrame(
        {
            "score": np.linspace(0, 1, 40),
            "target": [0] * 20 + [1] * 20,
        }
    )
    binner = Binner()
    binner.fit(X, y="target", method="step", n_bins=4)
    plot_data = BinningPlotData.from_binner(binner, "score")

    fig = plot_binning(plot_data, feature="score")

    assert isinstance(fig, matplotlib.figure.Figure)
