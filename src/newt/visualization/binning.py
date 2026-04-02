import warnings
from typing import TYPE_CHECKING, Dict

import matplotlib.figure
import pandas as pd

if TYPE_CHECKING:
    from ..features.binning import Binner

from .binning_viz import plot_binning_result


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
    Compatibility wrapper for the legacy binning visualization API.

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
    warnings.warn(
        "plot_binning() is deprecated and will be removed in a future release. "
        "Use newt.visualization.plot_binning_result() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if feature not in data.columns:
        raise ValueError(f"Feature '{feature}' not found in data.")
    if target not in data.columns:
        raise ValueError(f"Target '{target}' not found in data.")
    if feature not in combiner:
        raise ValueError(f"No binning rules found for feature '{feature}' in Binner.")

    y_col_map: Dict[str, str] = {
        "total": "total",
        "bad": "bads",
        "total_dist": "total_prop",
        "bad_dist": "bad_prop",
    }
    if bar_mode not in y_col_map:
        raise ValueError(
            "bar_mode must be one of 'total', 'bad', 'total_dist', or 'bad_dist'"
        )

    title = f"Binning Analysis: {feature}"
    if show_iv:
        title += f" (IV: {combiner.get_iv(feature):.4f})"

    return plot_binning_result(
        binner=combiner,
        X=data[[feature]],
        y=data[target],
        feature=feature,
        figsize=figsize,
        title=title,
        y_col=y_col_map[bar_mode],
        secondary_y_col="bad_rate",
    )
