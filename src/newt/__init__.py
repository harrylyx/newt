"""
Newt - Credit Scorecard Development Toolkit

A comprehensive Python library for building credit scorecards with:
- Feature binning (supervised and unsupervised methods)
- WOE/IV analysis
- Variable selection (pre-filtering, post-filtering, stepwise)
- Logistic regression modeling
- Scorecard generation
- Pipeline-style workflow
"""

__version__ = "0.1.0"

# WOE analysis
from newt.features.analysis import WOEEncoder

# Core binning
from newt.features.binning import Binner

# Feature selection
from newt.features.selection import PostFilter, PreFilter, StepwiseSelector

# Metrics
from newt.metrics import (
    calculate_auc,
    calculate_gini,
    calculate_ks,
    calculate_lift,
    calculate_psi,
    calculate_vif,
)

# Modeling
from newt.modeling import LogisticModel, Scorecard

# Pipeline
from newt.pipeline import ScorecardPipeline

# Statistics
from newt.statistics import EDAAnalyzer

__all__ = [
    # Core
    "Binner",
    "WOEEncoder",
    # Selection
    "PreFilter",
    "PostFilter",
    "StepwiseSelector",
    # Metrics
    "calculate_auc",
    "calculate_ks",
    "calculate_gini",
    "calculate_psi",
    "calculate_lift",
    "calculate_vif",
    # Modeling
    "LogisticModel",
    "Scorecard",
    # Pipeline
    "ScorecardPipeline",
    # Statistics
    "EDAAnalyzer",
]

