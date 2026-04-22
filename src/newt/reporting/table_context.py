"""Shared context objects for report table construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReportBuildOptions:
    """Runtime knobs for report compute path."""

    engine: str = "auto"
    max_workers: int = 4
    parallel_sheets: bool = True
    memory_mode: str = "compact"
    metrics_mode: str = "exact"


@dataclass
class ReportBuildContext:
    """Shared immutable-ish context and caches across sheet builders."""

    data: pd.DataFrame
    tag_col: str
    month_col: str
    options: ReportBuildOptions
    timings: List[Tuple[str, float]] = field(default_factory=list)
    latest_month_psi_cache: Dict[Tuple[str], Dict[object, float]] = field(
        default_factory=dict
    )
    split_metrics_cache: Dict[
        Tuple[object, ...], Tuple[pd.DataFrame, pd.DataFrame]
    ] = field(default_factory=dict)
    group_metrics_cache: Dict[Tuple[object, ...], pd.DataFrame] = field(
        default_factory=dict
    )
    lock: Lock = field(default_factory=Lock, repr=False)

    def record_timing(self, name: str, elapsed: float) -> None:
        self.timings.append((name, float(elapsed)))

    def _copy_frame_if_needed(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.options.memory_mode == "standard":
            return frame.copy()
        return frame

    def cache_get_latest_month_psi(
        self, key: Tuple[str]
    ) -> Optional[Dict[object, float]]:
        with self.lock:
            value = self.latest_month_psi_cache.get(key)
        if value is None:
            return None
        if self.options.memory_mode == "standard":
            return dict(value)
        return value

    def cache_set_latest_month_psi(
        self, key: Tuple[str], value: Dict[object, float]
    ) -> None:
        with self.lock:
            self.latest_month_psi_cache[key] = (
                dict(value) if self.options.memory_mode == "standard" else value
            )

    def cache_get_split_metrics(
        self,
        key: Tuple[object, ...],
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        with self.lock:
            value = self.split_metrics_cache.get(key)
        if value is None:
            return None
        return (
            self._copy_frame_if_needed(value[0]),
            self._copy_frame_if_needed(value[1]),
        )

    def cache_set_split_metrics(
        self,
        key: Tuple[object, ...],
        value: Tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        with self.lock:
            self.split_metrics_cache[key] = (
                self._copy_frame_if_needed(value[0]),
                self._copy_frame_if_needed(value[1]),
            )

    def cache_get_group_metrics(
        self, key: Tuple[object, ...]
    ) -> Optional[pd.DataFrame]:
        with self.lock:
            value = self.group_metrics_cache.get(key)
        if value is None:
            return None
        return self._copy_frame_if_needed(value)

    def cache_set_group_metrics(
        self, key: Tuple[object, ...], value: pd.DataFrame
    ) -> None:
        with self.lock:
            self.group_metrics_cache[key] = self._copy_frame_if_needed(value)


@dataclass
class FeatureComputationArtifacts:
    """Shared per-feature artifacts reused across variable-analysis blocks."""

    edges_by_feature: Dict[str, np.ndarray] = field(default_factory=dict)
    train_bin_stats_by_feature: Dict[str, pd.DataFrame] = field(default_factory=dict)
    oot_bin_stats_by_feature: Dict[str, pd.DataFrame] = field(default_factory=dict)
    use_feature_edges_for_psi_by_feature: Dict[str, bool] = field(default_factory=dict)
