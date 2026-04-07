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

__version__ = "0.1.6"

# Runtime config loader
from newt.config import load_conf

# WOE analysis
from newt.features.analysis import WOEEncoder

# Core binning
from newt.features.binning import Binner

# Feature selection
from newt.features.selection import FeatureSelector, PostFilter, StepwiseSelector

# Metrics
from newt.metrics import (
    build_reference_quantile_bins,
    calculate_auc,
    calculate_bin_performance_table,
    calculate_binary_metrics,
    calculate_feature_psi_against_base,
    calculate_gini,
    calculate_grouped_binary_metrics,
    calculate_grouped_psi,
    calculate_ks,
    calculate_latest_month_psi,
    calculate_lift,
    calculate_portrait_means_by_score_bin,
    calculate_psi,
    calculate_psi_batch,
    calculate_score_correlation_matrix,
    calculate_vif,
    summarize_label_distribution,
)

# Modeling
from newt.modeling import LogisticModel, Scorecard

# Pipeline
from newt.pipeline import ScorecardPipeline

# Reporting
from newt.reporting import (
    Report,
    calculate_dimensional_comparison,
    calculate_model_comparison,
    calculate_split_metrics,
)

__all__ = [
    # Core
    "Binner",
    "WOEEncoder",
    # Selection
    "FeatureSelector",
    "PostFilter",
    "StepwiseSelector",
    # Metrics
    "calculate_auc",
    "calculate_ks",
    "calculate_gini",
    "calculate_psi",
    "calculate_psi_batch",
    "calculate_grouped_psi",
    "calculate_feature_psi_against_base",
    "calculate_lift",
    "calculate_vif",
    "build_reference_quantile_bins",
    "calculate_bin_performance_table",
    "calculate_binary_metrics",
    "calculate_grouped_binary_metrics",
    "calculate_latest_month_psi",
    "calculate_portrait_means_by_score_bin",
    "calculate_score_correlation_matrix",
    "summarize_label_distribution",
    # Modeling
    "LogisticModel",
    "Scorecard",
    # Reporting
    "Report",
    "calculate_split_metrics",
    "calculate_dimensional_comparison",
    "calculate_model_comparison",
    # Runtime config
    "load_conf",
    # Pipeline
    "ScorecardPipeline",
]
