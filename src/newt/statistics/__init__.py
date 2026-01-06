"""
Statistics module for newt.

Provides exploratory data analysis utilities.
"""

from .eda import (
    ALL_METRICS,
    BASIC_METRICS,
    LABEL_METRICS,
    EDAAnalyzer,
)

__all__ = [
    "EDAAnalyzer",
    "ALL_METRICS",
    "BASIC_METRICS",
    "LABEL_METRICS",
]
