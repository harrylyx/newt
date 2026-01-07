"""
Configuration constants for the newt package.

Provides centralized default values for binning, filtering, and modeling.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class BinningConfig:
    """分箱相关默认配置"""

    DEFAULT_N_BINS: Final[int] = 5
    DEFAULT_BUCKETS: Final[int] = 10
    DEFAULT_EPSILON: Final[float] = 1e-8
    MIN_SAMPLES_LEAF: Final[int] = 10


@dataclass(frozen=True)
class FilteringConfig:
    """过滤相关默认配置"""

    DEFAULT_IV_THRESHOLD: Final[float] = 0.02
    DEFAULT_MISSING_THRESHOLD: Final[float] = 0.9
    DEFAULT_CORR_THRESHOLD: Final[float] = 0.8
    DEFAULT_PSI_THRESHOLD: Final[float] = 0.25
    DEFAULT_VIF_THRESHOLD: Final[float] = 10.0


@dataclass(frozen=True)
class ModelingConfig:
    """建模相关默认配置"""

    DEFAULT_P_ENTER: Final[float] = 0.05
    DEFAULT_P_REMOVE: Final[float] = 0.10
    DEFAULT_CLASSIFICATION_THRESHOLD: Final[float] = 0.5


@dataclass(frozen=True)
class ScorecardConfig:
    """评分卡相关默认配置"""

    DEFAULT_PDO: Final[int] = 20
    DEFAULT_BASE_SCORE: Final[int] = 600
    DEFAULT_BASE_ODDS: Final[float] = 1.0


# Singleton instances for easy access
BINNING = BinningConfig()
FILTERING = FilteringConfig()
MODELING = ModelingConfig()
SCORECARD = ScorecardConfig()
