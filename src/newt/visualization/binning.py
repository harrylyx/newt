import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from ..features.binning import Combiner

from ..features.analysis.woe_calculator import WOEEncoder


def plot_binning(
    combiner: "Combiner",
    data: pd.DataFrame,
    feature: str,
    target: str,
    labels: bool = True,
    show_iv: bool = True,
    decimals: int = 2,
    bar_mode: str = "total",  # 'total', 'bad', 'total_dist', 'bad_dist'
) -> go.Figure:
    """
    Visualize binning results for a specific feature using a fitted Combiner.

    Parameters
    ----------
    combiner : Combiner
        Fitted Combiner object containing binning rules.
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

    Returns
    -------
    go.Figure
        Plotly figure.
    """

    # 1. Validation
    if feature not in data.columns:
        raise ValueError(f"Feature '{feature}' not found in data.")
    if target not in data.columns:
        raise ValueError(f"Target '{target}' not found in data.")

    # Check if feature has rules in Combiner
    if feature not in combiner.rules_:
        raise ValueError(f"No binning rules found for feature '{feature}' in Combiner.")

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
    # WOEEncoder expects to bin data itself or receive categories.
    # If we pass binned_series (Categorical), WOEEncoder will convert to string.

    # We use WOEEncoder to get IV and summary stats
    encoder = WOEEncoder()
    # We pass the binned series.
    # Note: WOEEncoder.fit converts non-numeric X to string.
    encoder.fit(binned_series, data[target])

    stats = encoder.summary_  # Indexed by bin string (if converted)
    total_iv = encoder.iv_

    # Re-order stats to match bin order
    # binned_series.cat.categories gives the correct order of intervals
    categories = binned_series.cat.categories
    # Convert categories to string format used by WOEEncoder (astype(str))
    cat_strings = categories.astype(str)

    # Reindex stats
    # Filter out empty bins that might not be in stats?
    # WOEEncoder.fit only sees present values.
    # But we want to show all bins defined by splits usually?
    # If a bin is empty, it won't be in summary_.
    # Let's reindex and fill 0.
    stats = stats.reindex(cat_strings).fillna(0)

    # Recalculate rates/dists for empty bins if needed (fillna 0 handles counts)
    # But bad_rate needs calc
    stats["bad_rate"] = stats["bad"] / stats["total"].replace(0, np.nan)
    # If total is 0, bad_rate is NaN. fill with 0?
    stats["bad_rate"] = stats["bad_rate"].fillna(0)

    # WOEEncoder calculates dist_bad and dist_good.
    # It might use epsilon adjustment.
    # The summary_ columns: total, bad, good, dist_good, dist_bad, woe, iv_contribution

    # 4. Format Labels
    # We have the categories (Intervals).
    def format_interval(cat_str):
        # We need to map back to Interval object or parse string?
        # cat_strings corresponds to categories (Interval objects).
        # We can map from index (string) back to Interval via dict?

        # Simple parsing of string interval matching WOEEncoder's string conversion
        # Interval str: "(-inf, 0.5]" or "(0.5, 10.0]"
        # Regex to parse
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

    # 5. Plotting
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar Chart (Histogram)
    bar_y = stats[bar_mode] if bar_mode in stats.columns else stats["total"]

    bar_name_map = {
        "total": "Total Count",
        "bad": "Bad Count",
        "dist_good": "Good Dist",  # WOEEncoder uses dist_good
        "dist_bad": "Bad Dist",
    }
    # Map user bar_mode to WOEEncoder columns if different
    # available: total, bad, good, dist_good, dist_bad, woe, iv_contribution
    # user requested: 'total', 'bad', 'total_dist', 'bad_dist'

    if bar_mode == "total_dist":
        # Calculate manually if not in summary
        bar_y = stats["total"] / stats["total"].sum()
        bar_name = "Total %"
    elif bar_mode == "bad_dist":
        bar_y = stats["dist_bad"]  # WOEEncoder has this
        bar_name = "Bad %"
    else:
        bar_y = stats.get(bar_mode, stats.get("total"))
        bar_name = bar_name_map.get(bar_mode, bar_mode)

    fig.add_trace(
        go.Bar(
            x=stats.index,
            y=bar_y,
            name=bar_name,
            marker_color="#636EFA",
            opacity=0.6,
            text=bar_y if bar_mode in ["total", "bad"] else [f"{v:.1%}" for v in bar_y],
            textposition="auto",
        ),
        secondary_y=False,
    )

    # Line Chart (Bad Rate)
    # Bad Rate calculation
    # stats['bad_rate'] calculated above
    fig.add_trace(
        go.Scatter(
            x=stats.index,
            y=stats["bad_rate"],
            name="Bad Rate",
            mode="lines+markers+text",
            line=dict(color="#EF553B", width=3),
            text=[f"{v:.1%}" for v in stats["bad_rate"]],
            textposition="top center",
        ),
        secondary_y=True,
    )

    # Layout
    title_text = f"Binning Analysis: {feature}"
    if show_iv:
        title_text += f" (IV: {total_iv:.4f})"

    fig.update_layout(
        title=title_text,
        xaxis_title="Bins",
        legend=dict(x=0.5, y=1.1, orientation="h", xanchor="center"),
        template="plotly_white",
        hovermode="x unified",
    )

    # Axis formatting
    fig.update_yaxes(title_text=bar_name, secondary_y=False)
    fig.update_yaxes(title_text="Bad Rate", secondary_y=True, tickformat=".1%")

    return fig
