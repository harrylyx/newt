"""Unified binary classification metrics with exact and binned modes.

Two computation modes:

- ``exact``: Single sklearn roc_curve call for both AUC and KS, single
  argsort for all lift levels.  Produces sklearn-compatible results.
- ``binned``: O(N) histogram + O(B) metric computation from bin counts.
  Significantly faster on large datasets at the cost of slight approximation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from newt._engine import ensure_native_functions, validate_engine
from newt._native import load_native_module
from newt.metrics._common import build_score_edges, prepare_binary_metric_input

PERCENT_LEVELS: Tuple[float, ...] = (0.10, 0.05, 0.02, 0.01)
VALID_METRICS_MODES = frozenset(["exact", "binned"])


def calculate_binary_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_score: Union[pd.Series, np.ndarray],
    lift_use_descending_score: bool = True,
    reverse_auc_label: bool = False,
    metrics_mode: str = "exact",
    bins: int = 10,
    lift_levels: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Calculate AUC, KS, and Lift metrics for a binary label/score pair.

    Args:
        y_true: Binary labels (0/1). May contain non-binary values which
            are excluded from metric computation but counted in totals.
        y_score: Numeric score/probability.
        lift_use_descending_score: If True, higher score = higher risk
            for lift calculation.
        reverse_auc_label: If True, compute AUC as 1 - AUC(y, s) which
            is equivalent to AUC(1-y, s).  KS is unaffected.
        metrics_mode: ``"exact"`` (sklearn roc_curve) or ``"binned"``
            (histogram approximation).
        bins: Number of bins for binned mode.
        lift_levels: Fraction levels for lift-at-k.
            Default: ``(0.10, 0.05, 0.02, 0.01)``.

    Returns:
        Dict with keys: 总, 好, 坏, 坏占比, KS, AUC, {k}%lift ...
    """
    if metrics_mode not in VALID_METRICS_MODES:
        raise ValueError(
            f"metrics_mode must be one of {sorted(VALID_METRICS_MODES)}, "
            f"got: {metrics_mode}"
        )

    levels = tuple(lift_levels) if lift_levels is not None else PERCENT_LEVELS

    prepared = prepare_binary_metric_input(y_true=y_true, y_score=y_score)
    y_clean = prepared.y_clean
    score_clean = prepared.score_clean

    if len(y_clean) == 0 or np.unique(y_clean).size < 2:
        metrics_dict: Dict[str, float] = {"KS": np.nan, "AUC": np.nan}
        lift_dict: Dict[str, float] = {f"{int(lv * 100)}%lift": np.nan for lv in levels}
    elif metrics_mode == "binned":
        metrics_dict, lift_dict = _binned_metrics(
            y_clean,
            score_clean,
            bins=bins,
            levels=levels,
            lift_descending=lift_use_descending_score,
            reverse_auc=reverse_auc_label,
        )
    else:
        metrics_dict, lift_dict = _exact_metrics(
            y_clean,
            score_clean,
            levels=levels,
            lift_descending=lift_use_descending_score,
            reverse_auc=reverse_auc_label,
        )

    return {
        "总": prepared.total,
        "好": prepared.good,
        "坏": prepared.bad,
        "坏占比": prepared.bad_rate,
        **metrics_dict,
        **lift_dict,
    }


# ---------------------------------------------------------------------------
# Exact path (single roc_curve + single argsort)
# ---------------------------------------------------------------------------


def _exact_metrics(
    y_clean: np.ndarray,
    score_clean: np.ndarray,
    levels: Tuple[float, ...],
    lift_descending: bool,
    reverse_auc: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Exact metrics: one roc_curve → AUC + KS, one argsort → all lifts."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_clean, score_clean)
    auc_value = float(np.trapz(tpr, fpr))
    ks_value = float(np.max(np.abs(tpr - fpr)))

    if reverse_auc:
        auc_value = 1.0 - auc_value

    lift_dict = _all_lifts_from_sorted(
        y_clean, score_clean, levels, descending=lift_descending
    )
    return {"KS": ks_value, "AUC": auc_value}, lift_dict


def _all_lifts_from_sorted(
    y_clean: np.ndarray,
    score_clean: np.ndarray,
    levels: Tuple[float, ...],
    descending: bool,
) -> Dict[str, float]:
    """Compute lift at multiple k levels with a single argsort."""
    lift_score = score_clean if descending else -score_clean
    sorted_idx = np.argsort(lift_score)[::-1]
    n = len(y_clean)
    global_event_rate = float(np.mean(y_clean))

    if global_event_rate == 0:
        return {f"{int(lv * 100)}%lift": 0.0 for lv in levels}

    lifts: Dict[str, float] = {}
    for lv in levels:
        n_top = max(1, int(np.ceil(n * lv)))
        top_event_rate = float(np.mean(y_clean[sorted_idx[:n_top]]))
        lifts[f"{int(lv * 100)}%lift"] = top_event_rate / global_event_rate
    return lifts


# ---------------------------------------------------------------------------
# Binned path (O(N) histogram + O(B) computation)
# ---------------------------------------------------------------------------


def _binned_metrics(
    y_clean: np.ndarray,
    score_clean: np.ndarray,
    bins: int,
    levels: Tuple[float, ...],
    lift_descending: bool,
    reverse_auc: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Binned approximation: histogram → AUC, KS, Lift."""
    edges = build_score_edges(score_clean, bins)
    n_bins = len(edges) - 1

    # O(N) bin assignment
    bin_idx = np.clip(
        np.searchsorted(edges[1:-1], score_clean, side="right"),
        0,
        n_bins - 1,
    )

    # O(N) counting via bincount
    bad_mask = y_clean == 1
    good_counts = np.bincount(bin_idx[~bad_mask], minlength=n_bins)[:n_bins].astype(
        float
    )
    bad_counts = np.bincount(bin_idx[bad_mask], minlength=n_bins)[:n_bins].astype(float)

    # Evaluate ROC in descending-score direction to align with exact mode.
    auc_val = calculate_binned_auc(good_counts[::-1], bad_counts[::-1])
    ks_val = calculate_binned_ks(good_counts[::-1], bad_counts[::-1])

    if reverse_auc:
        auc_val = 1.0 - auc_val

    lift_dict = _binned_lifts(
        good_counts,
        bad_counts,
        len(y_clean),
        levels,
        descending=lift_descending,
    )

    return {"KS": ks_val, "AUC": auc_val}, lift_dict


def calculate_binned_auc(
    good_counts: np.ndarray,
    bad_counts: np.ndarray,
) -> float:
    """AUC from bin counts (bins in ascending score order).

    Uses trapezoidal rule on cumulative good/bad proportions
    which approximate the ROC curve (FPR, TPR).
    """
    total_good = float(good_counts.sum())
    total_bad = float(bad_counts.sum())
    if total_good == 0 or total_bad == 0:
        return np.nan

    fpr = np.concatenate([[0.0], np.cumsum(good_counts) / total_good])
    tpr = np.concatenate([[0.0], np.cumsum(bad_counts) / total_bad])
    return float(np.trapz(tpr, fpr))


def calculate_binned_ks(
    good_counts: np.ndarray,
    bad_counts: np.ndarray,
) -> float:
    """KS statistic from bin counts (bins in ascending score order)."""
    total_good = float(good_counts.sum())
    total_bad = float(bad_counts.sum())
    if total_good == 0 or total_bad == 0:
        return np.nan

    cum_good = np.cumsum(good_counts) / total_good
    cum_bad = np.cumsum(bad_counts) / total_bad
    return float(np.max(np.abs(cum_bad - cum_good)))


def _binned_lifts(
    good_counts: np.ndarray,
    bad_counts: np.ndarray,
    total_n: int,
    levels: Tuple[float, ...],
    descending: bool = True,
) -> Dict[str, float]:
    """Approximate lift at k from bin counts."""
    total_counts = good_counts + bad_counts
    global_bad_rate = float(bad_counts.sum() / total_n) if total_n > 0 else 0.0

    if global_bad_rate == 0:
        return {f"{int(k * 100)}%lift": 0.0 for k in levels}

    if descending:
        ordered_total = total_counts[::-1].copy()
        ordered_bad = bad_counts[::-1].copy()
    else:
        ordered_total = total_counts.copy()
        ordered_bad = bad_counts.copy()

    cum_total = np.cumsum(ordered_total)
    cum_bad = np.cumsum(ordered_bad)

    lifts: Dict[str, float] = {}
    for k in levels:
        n_top = max(1, int(np.ceil(total_n * k)))
        idx = int(np.searchsorted(cum_total, n_top, side="right"))
        idx = min(idx, len(cum_total) - 1)

        if cum_total[idx] > 0:
            if idx == 0:
                top_bad_rate = float(cum_bad[0] / cum_total[0])
            else:
                needed = float(n_top - cum_total[idx - 1])
                bin_total = ordered_total[idx]
                if bin_total > 0:
                    bin_bad_rate = ordered_bad[idx] / bin_total
                    prev_bads = cum_bad[idx - 1]
                    interpolated_bads = prev_bads + needed * bin_bad_rate
                    top_bad_rate = float(interpolated_bads / n_top)
                else:
                    top_bad_rate = float(cum_bad[idx] / max(cum_total[idx], 1))
        else:
            top_bad_rate = 0.0

        lifts[f"{int(k * 100)}%lift"] = top_bad_rate / global_bad_rate
    return lifts


# ---------------------------------------------------------------------------
# Batch interface (Phase 2 will add Rust path here)
# ---------------------------------------------------------------------------


def calculate_binary_metrics_batch(
    groups: Sequence[Tuple[np.ndarray, np.ndarray]],
    lift_use_descending_score: bool = True,
    reverse_auc_label: bool = False,
    metrics_mode: str = "exact",
    bins: int = 10,
    lift_levels: Optional[Sequence[float]] = None,
    engine: str = "python",
) -> List[Dict[str, float]]:
    """Calculate binary metrics for multiple (y_true, y_score) groups.

    Args:
        groups: Sequence of (y_true, y_score) array pairs.
        engine: ``"python"``, ``"rust"``, or ``"auto"``.
    """
    levels = tuple(lift_levels) if lift_levels is not None else PERCENT_LEVELS

    try:
        validate_engine(engine)
    except ValueError as exc:
        raise ValueError("engine must be 'python', 'rust' or 'auto'") from exc

    if engine == "rust":
        return _rust_binary_metrics_batch(
            groups=groups,
            lift_use_descending_score=lift_use_descending_score,
            reverse_auc_label=reverse_auc_label,
            metrics_mode=metrics_mode,
            bins=bins,
            levels=levels,
            strict=True,
        )

    if engine == "auto":
        rust_result = _rust_binary_metrics_batch(
            groups=groups,
            lift_use_descending_score=lift_use_descending_score,
            reverse_auc_label=reverse_auc_label,
            metrics_mode=metrics_mode,
            bins=bins,
            levels=levels,
            strict=False,
        )
        if rust_result is not None:
            return rust_result

    return [
        calculate_binary_metrics(
            y_true=y_true,
            y_score=y_score,
            lift_use_descending_score=lift_use_descending_score,
            reverse_auc_label=reverse_auc_label,
            metrics_mode=metrics_mode,
            bins=bins,
            lift_levels=levels,
        )
        for y_true, y_score in groups
    ]


def _rust_binary_metrics_batch(
    groups: Sequence[Tuple[np.ndarray, np.ndarray]],
    lift_use_descending_score: bool,
    reverse_auc_label: bool,
    metrics_mode: str,
    bins: int,
    levels: Tuple[float, ...],
    strict: bool,
) -> Optional[List[Dict[str, float]]]:
    module = _load_rust_module()
    if module is None:
        if strict:
            raise ImportError(
                "Rust engine requested but native extension is unavailable."
            )
        return None

    try:
        ensure_native_functions(
            module,
            ["calculate_binary_metrics_batch_numpy"],
            component="Rust binary metrics engine",
        )
    except RuntimeError:
        if strict:
            raise RuntimeError(
                "Rust engine requested but calculate_binary_metrics_batch_numpy "
                "is unavailable in native extension."
            )
        return None
    fn = module.calculate_binary_metrics_batch_numpy

    try:
        y_groups = [
            np.ascontiguousarray(np.asarray(y_true, dtype=np.float64).ravel())
            for y_true, _ in groups
        ]
        score_groups = [
            np.ascontiguousarray(np.asarray(y_score, dtype=np.float64).ravel())
            for _, y_score in groups
        ]
        values = fn(
            y_groups,
            score_groups,
            list(levels),
            bool(lift_use_descending_score),
            bool(reverse_auc_label),
            str(metrics_mode),
            int(bins),
        )
        return [{str(key): float(val) for key, val in row.items()} for row in values]
    except Exception:
        if strict:
            raise RuntimeError("Rust binary metrics execution failed.")
        return None


def _load_rust_module():
    return load_native_module()


__all__ = [
    "calculate_binary_metrics",
    "calculate_binary_metrics_batch",
    "calculate_binned_auc",
    "calculate_binned_ks",
    "PERCENT_LEVELS",
    "VALID_METRICS_MODES",
]
