import re
from typing import TYPE_CHECKING

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from ..features.binning import Binner

from ..features.analysis.woe_calculator import WOEEncoder


def plot_binning(
    combiner: "Binner",
    data: pd.DataFrame,
    feature: str,
    target: str,
    labels: bool = True,
    show_iv: bool = True,
    decimals: int = 2,
    bar_mode: str = "total",  # 'total', 'bad', 'total_dist', 'bad_dist'
    figsize: tuple = (10, 6),
) -> matplotlib.figure.Figure:
    """
    Visualize binning results for a specific feature using a fitted Binner.

    Parameters
    ----------
    combiner : Binner
        Fitted Binner object containing binning rules.
    data : pd.DataFrame
        Data containing the feature and target columns.
    feature : str
        Name of the feature column to visualize.
    target : str
        Name of the target column.
    labels : bool
        Whether to show bin labels.
    show_iv : bool
        Whether to show IV in title.
    decimals : int
        Number of decimals to keep in bin labels.
    bar_mode : str
        Metric for bar chart: 'total' (count), 'bad' (bad count),
        'total_dist' (percentage of total), 'bad_dist' (percentage of bads).
    figsize : tuple
        Figure size (width, height).

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object.
    """

    # 1. Validation
    if feature not in data.columns:
        raise ValueError(f"Feature '{feature}' not found in data.")
    if target not in data.columns:
        raise ValueError(f"Target '{target}' not found in data.")

    # Check if feature has rules in Binner
    if feature not in combiner.rules_:
        raise ValueError(f"No binning rules found for feature '{feature}' in Binner.")

    splits = combiner.rules_[feature]

    # 2. Apply Binning
    # Use pd.cut with splits
    if not splits:
        bins = [-np.inf, np.inf]
    else:
        bins = [-np.inf] + sorted(splits) + [np.inf]

    # Create a temporary series for binning
    binned_series = pd.cut(data[feature], bins=bins, include_lowest=True)

    # 3. Calculate Stats using WOEEncoder for consistency
    # We use WOEEncoder to get IV and summary stats
    encoder = WOEEncoder()
    # WOEEncoder.fit converts non-numeric X to string.
    encoder.fit(binned_series, data[target])

    stats = encoder.summary_  # Indexed by bin string (if converted)
    total_iv = encoder.iv_

    # Re-order stats to match bin order
    categories = binned_series.cat.categories
    # Convert categories to string format used by WOEEncoder (astype(str))
    cat_strings = categories.astype(str)

    # Reindex stats and fill 0
    stats = stats.reindex(cat_strings).fillna(0)

    # Recalculate rates/dists for empty bins
    stats["bad_rate"] = stats["bad"] / stats["total"].replace(0, np.nan)
    stats["bad_rate"] = stats["bad_rate"].fillna(0)

    # 4. Format Labels
    def format_interval(cat_str):
        # Regex to parse interval string
        m = re.match(r"[\[\(](.*), (.*)[\]\)]", str(cat_str))
        if not m:
            return str(cat_str)
        left_s, right_s = m.groups()

        try:
            left = float(left_s)
            right = float(right_s)

            l_str = "-inf" if left == -np.inf else f"{left:.{decimals}f}"
            r_str = "inf" if right == np.inf else f"{right:.{decimals}f}"
            return f"({l_str}, {r_str}]"
        except ValueError:
            return str(cat_str)

    stats.index = stats.index.map(format_interval)

    # 5. Plotting with Seaborn/Matplotlib
    # Set style temporarily? or use global defaults.
    # We'll use object-oriented approach.

    fig, ax1 = plt.subplots(figsize=figsize)

    # Bar Chart Data
    bar_name_map = {
        "total": "Total Count",
        "bad": "Bad Count",
        "dist_good": "Good Dist",
        "dist_bad": "Bad Dist",
    }

    if bar_mode == "total_dist":
        bar_y = stats["total"] / stats["total"].sum()
        bar_name = "Total %"
    elif bar_mode == "bad_dist":
        bar_y = stats["dist_bad"]
        bar_name = "Bad %"
    else:
        bar_y = stats.get(bar_mode, stats.get("total"))
        bar_name = bar_name_map.get(bar_mode, bar_mode)

    # Bar Chart (Primary Axis)
    sns.barplot(
        x=stats.index, y=bar_y, ax=ax1, alpha=0.6, color="#636EFA", label=bar_name
    )

    # Bar Labels
    for i, v in enumerate(bar_y):
        text = f"{v:.1%}" if bar_mode in ["total_dist", "bad_dist"] else f"{v:.0f}"
        ax1.text(i, v, text, ha="center", va="bottom", fontsize=9)

    ax1.set_ylabel(bar_name, color="#636EFA", fontsize=12)
    ax1.set_xlabel("Bins", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)

    # Line Chart (Secondary Axis - Bad Rate)
    ax2 = ax1.twinx()

    sns.lineplot(
        x=stats.index,
        y=stats["bad_rate"],
        ax=ax2,
        color="#EF553B",
        marker="o",
        linewidth=2,
        label="Bad Rate",
    )

    # Line Labels
    for i, v in enumerate(stats["bad_rate"]):
        ax2.text(
            i,
            v,
            f"{v:.1%}",
            ha="center",
            va="bottom",
            color="#EF553B",
            fontsize=9,
            fontweight="bold",
        )

    ax2.set_ylabel("Bad Rate", color="#EF553B", fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    # Clean up grid
    ax2.grid(False)  # Turn off secondary grid to avoid clutter

    # Title
    title_text = f"Binning Analysis: {feature}"
    if show_iv:
        title_text += f" (IV: {total_iv:.4f})"
    plt.title(title_text, fontsize=14, pad=20)

    # Layout adjustments
    plt.tight_layout()

    return fig
