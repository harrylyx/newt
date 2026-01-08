"""
Binning visualization utilities.

Provides visualization for binning results, WOE/IV analysis, etc.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from newt.config import FILTERING

# Optional imports for visualization
try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. " "Install it with: pip install matplotlib"
        )


def plot_binning_result(
    binner: Any,
    X: pd.DataFrame,
    y: pd.Series,
    feature: str,
    woe_encoder: Optional[Any] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    x_col: str = "bin",
    y_col: Optional[Union[str, List[str]]] = None,
    secondary_y_col: Optional[Union[str, List[str]]] = "bad_rate",
) -> Any:
    """
    Plot binning result for a single feature.

    Supports generic columns for axes. Defaults to Good/Bad bars and Event Rate line.

    Parameters
    ----------
    binner : Binner
        Fitted Binner object.
    X : pd.DataFrame
        Feature data (preserved for API compatibility, not used if binner has stats).
    y : pd.Series
        Target variable (preserved for API compatibility).
    feature : str
        Feature name to plot.
    woe_encoder : WOEEncoder, optional
        WOE encoder (preserved for API compatibility).
    figsize : Tuple[int, int]
        Figure size. Default (12, 6).
    title : str, optional
        Plot title.
    x_col : str
        Column to use for x-axis labels. Default 'bin'.
    y_col : str or list
        Column(s) to plot as bars on primary y-axis. Default ['bad_prop'].
    secondary_y_col : str or list, optional
        Column(s) to plot as lines on secondary y-axis. Default 'bad_rate'.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _check_matplotlib()

    if feature not in binner.binners_:
        raise ValueError(f"Feature '{feature}' not found in binner.")

    # Get pre-calculated statistics
    stats = binner[feature].stats.copy()

    # Ensure stats sorted by bin index if present, else just use as is
    # stats usually comes sorted from binner

    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(stats))

    # Handle primary Y (Bars)
    if y_col is None:
        y_col = ["bad_prop"]
    if isinstance(y_col, str):
        y_cols = [y_col]
    else:
        y_cols = y_col

    # Plot bars
    width = 0.8 / len(y_cols)
    colors = [
        "steelblue",
        "coral",
        "gold",
        "forestgreen",
        "purple",
    ]  # Default color cycle

    for i, col in enumerate(y_cols):
        if col not in stats.columns:
            print(f"Warning: Column '{col}' not in stats, skipping.")
            continue

        offset = (i - len(y_cols) / 2) * width + width / 2
        ax1.bar(
            x_pos + offset,
            stats[col],
            width,
            label=col,
            color=colors[i % len(colors)],
            alpha=0.8,
        )

    ax1.set_xlabel(x_col)
    ax1.set_ylabel(" / ".join(y_cols))
    ax1.set_xticks(x_pos)

    # Format x-labels
    x_labels = [str(x)[:20] for x in stats[x_col]]
    ax1.set_xticklabels(x_labels, rotation=45, ha="right")

    # Legend 1
    handles1, labels1 = ax1.get_legend_handles_labels()

    # Handle secondary Y (Line)
    if secondary_y_col:
        if isinstance(secondary_y_col, str):
            sec_cols = [secondary_y_col]
        else:
            sec_cols = secondary_y_col

        ax2 = ax1.twinx()
        line_colors = ["green", "red", "blue", "black"]

        for i, col in enumerate(sec_cols):
            if col not in stats.columns:
                print(f"Warning: Column '{col}' not in stats, skipping.")
                continue

            ax2.plot(
                x_pos,
                stats[col],
                marker="o",
                linewidth=2,
                label=col,
                color=line_colors[i % len(line_colors)],
            )

            # Auto-scale Y2 to look nice (start from 0)
            if all(stats[col] >= 0) and all(stats[col] <= 1):
                ax2.set_ylim(0, max(stats[col].max() * 1.2, 0.01))
            else:
                # Generic scaling
                ymin, ymax = stats[col].min(), stats[col].max()
                span = ymax - ymin
                ax2.set_ylim(ymin - span * 0.1, ymax + span * 0.1)

        ax2.set_ylabel(" / ".join(sec_cols))

        # Merge legends
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(handles1, labels1, loc="upper left")

    # Title
    if title is None:
        title = f"Binning Result: {feature}"
    plt.title(title)
    plt.tight_layout()

    # Close the figure to prevent automatic display in notebooks (double plotting)
    # The figure object is still returned and can be displayed explicitly or by notebook cell return
    plt.close(fig)

    return fig


def plot_iv_ranking(
    iv_dict: Dict[str, float],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    threshold: float = FILTERING.DEFAULT_IV_THRESHOLD,
) -> Any:
    """
    Plot IV ranking bar chart.

    Parameters
    ----------
    iv_dict : Dict[str, float]
        Dictionary mapping feature names to IV values.
    top_n : int
        Number of top features to show. Default 20.
    figsize : Tuple[int, int]
        Figure size. Default (10, 8).
    title : str, optional
        Plot title.
    threshold : float
        IV threshold line. Default 0.02.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _check_matplotlib()

    # Sort by IV
    sorted_iv = sorted(iv_dict.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        sorted_iv = sorted_iv[:top_n]

    features = [item[0] for item in sorted_iv]
    ivs = [item[1] for item in sorted_iv]

    # Create color map based on IV strength
    colors = []
    for iv in ivs:
        if iv >= 0.5:
            colors.append("darkgreen")  # Suspiciously high
        elif iv >= 0.3:
            colors.append("green")  # Strong
        elif iv >= 0.1:
            colors.append("yellowgreen")  # Medium
        elif iv >= 0.02:
            colors.append("yellow")  # Weak
        else:
            colors.append("lightgray")  # Useless

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, ivs, color=colors, edgecolor="black", alpha=0.8)

    # Add threshold line
    ax.axvline(
        x=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold})",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Top to bottom
    ax.set_xlabel("Information Value (IV)")

    # Add value labels
    for bar, iv in zip(bars, ivs):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{iv:.4f}",
            va="center",
            fontsize=9,
        )

    # Legend for colors
    legend_patches = [
        mpatches.Patch(color="darkgreen", label="IV > 0.5 (Suspiciously High)"),
        mpatches.Patch(color="green", label="0.3 < IV ≤ 0.5 (Strong)"),
        mpatches.Patch(color="yellowgreen", label="0.1 < IV ≤ 0.3 (Medium)"),
        mpatches.Patch(color="yellow", label="0.02 < IV ≤ 0.1 (Weak)"),
        mpatches.Patch(color="lightgray", label="IV ≤ 0.02 (Useless)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    if title is None:
        title = f"Top {len(features)} Features by IV"
    plt.title(title)
    plt.tight_layout()

    return fig


def plot_woe_pattern(
    woe_encoder: Any,
    feature: str,
    figsize: Tuple[int, int] = (10, 5),
    title: Optional[str] = None,
) -> Any:
    """
    Plot WOE pattern for a feature.

    Parameters
    ----------
    woe_encoder : WOEEncoder
        Fitted WOE encoder with summary_ attribute.
    feature : str
        Feature name for title.
    figsize : Tuple[int, int]
        Figure size.
    title : str, optional
        Plot title.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _check_matplotlib()

    if not hasattr(woe_encoder, "summary_") or woe_encoder.summary_.empty:
        raise ValueError("WOE encoder has no summary data.")

    summary = woe_encoder.summary_.reset_index()

    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(summary))

    # Color based on WOE sign
    colors = ["coral" if w < 0 else "steelblue" for w in summary["woe"]]

    ax.bar(x_pos, summary["woe"], color=colors, edgecolor="black", alpha=0.8)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Bin")
    ax.set_ylabel("WOE")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b)[:15] for b in summary["bin"]], rotation=45, ha="right")

    # Add IV annotation
    iv = woe_encoder.iv_ if hasattr(woe_encoder, "iv_") else None
    if iv is not None:
        ax.text(
            0.02,
            0.98,
            f"IV = {iv:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    if title is None:
        title = f"WOE Pattern: {feature}"
    plt.title(title)
    plt.tight_layout()

    return fig


def plot_psi_comparison(
    psi_dict: Dict[str, float],
    threshold: float = FILTERING.DEFAULT_PSI_THRESHOLD,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
) -> Any:
    """
    Plot PSI values as horizontal bar chart.

    Parameters
    ----------
    psi_dict : Dict[str, float]
        Dictionary mapping feature names to PSI values.
    threshold : float
        PSI threshold. Default 0.25.
    figsize : Tuple[int, int]
        Figure size.
    title : str, optional
        Plot title.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _check_matplotlib()

    # Sort by PSI
    sorted_psi = sorted(psi_dict.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_psi]
    psis = [item[1] for item in sorted_psi]

    # Color based on PSI level
    colors = []
    for psi in psis:
        if psi > threshold:
            colors.append("red")  # Significant shift
        elif psi > 0.1:
            colors.append("orange")  # Moderate shift
        else:
            colors.append("green")  # No significant shift

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(features))
    ax.barh(y_pos, psis, color=colors, edgecolor="black", alpha=0.8)

    # Threshold lines
    ax.axvline(x=0.1, color="orange", linestyle="--", linewidth=1.5, label="Moderate (0.1)")
    ax.axvline(
        x=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Significant ({threshold})",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("PSI")
    ax.legend(loc="lower right")

    if title is None:
        title = "Feature PSI Comparison"
    plt.title(title)
    plt.tight_layout()

    return fig
