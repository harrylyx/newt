# Newt Library User Guide

This guide covers the end-to-end workflow for credit scorecard development using `newt`, including binning, feature selection, feature transformation, modeling, and scorecard generation.

The project supports Python `>=3.8.5,<3.13` (Python `3.8.5` through `3.12.x`).

## Installation

### From PyPI

```bash
pip install newt
# or with uv
uv add newt
```

### From GitHub Release (recommended for Native Engine)

Prebuilt wheels with the high-performance Rust engine are available from
[GitHub Releases](https://github.com/harrylyx/newt/releases). Download the
wheel matching your platform and Python version:

```bash
pip install newt-<version>-<platform>.whl
```

Supported platforms for prebuilt wheels:

| OS | Architecture |
|---|---|
| macOS | arm64 |
| Windows | x86_64 |
| Linux | x86_64, arm64 |

No Rust toolchain is required when installing from an official wheel.

### High-Performance Native Engine (Rust)

Newt includes a high-performance Rust extension for various core performance paths, including single-feature IV, batch IV, PSI, ChiMerge, Stepwise selection, and optimized batch Logistic Regression fitting.
When installed from an official wheel, it works immediately:

```python
from newt.features.analysis import calculate_batch_iv, calculate_iv

# Uses the Rust engine by default
single = calculate_iv(df, target="target", feature="age")
batch = calculate_batch_iv(X, y)

# Explicit Python fallback
single_py = calculate_iv(df, target="target", feature="age", engine="python")
batch_py = calculate_batch_iv(X, y, engine="python")
```

If the Rust extension is not available (e.g. source install without Rust
toolchain), requesting `engine="rust"` raises a clear `ImportError` with
instructions. The Python fallback is always available via `engine="python"`.

### Optional `opt` Method Dependency Scope

The optional `opt` binning stack (`newt[optbinning]`) is available on Python
`<3.12` because of upstream dependency limits. This does not change Newt core
support (`3.8.5` through `3.12.x`).

### Default Local Runtime Split

When working inside this repository, keep environments split by purpose:

- `.venv` (Python `3.8.5`) for development, tests, linting, and package edits.
- `.venv-benchmark-3.10` (Python `3.10.19`) for `newt-benchmark` and `toad`.

## Table of Contents

1. [Feature Binning](#1-feature-binning)
2. [Feature Selection](#2-feature-selection)
3. [Binned Feature Analysis](#3-binned-feature-analysis)
4. [Logistic Regression Modeling](#4-logistic-regression-modeling)
5. [Scorecard Generation](#5-scorecard-generation)
6. [Complete Pipeline](#6-complete-pipeline)
7. [Visualization](#7-visualization)
8. [Manual Adjustment](#8-manual-adjustment)
9. [Deployment](#9-deployment)
10. [Metrics](#10-metrics)
11. [Excel Model Report](#11-excel-model-report)

---

## 1. Feature Binning

The core class for binning is `newt.Binner` (or `newt.features.binning.Binner`). It provides a unified interface for various binning algorithms.

### Supported Methods

- `'chi'`: ChiMerge (Chi-square binning) - **Default**, supervised
- `'dt'`: Decision Tree binning - Supervised, finds optimal splits
- `'opt'`: Optimal Binning via Constraints - Supervised, install with
  `pip install "newt[optbinning]"`
- `'kmean'`: K-Means clustering - Unsupervised
- `'quantile'`: Equal frequency binning - Unsupervised
- `'step'`: Equal width binning - Unsupervised

### Basic Usage

```python
import pandas as pd
from newt import Binner

# Load your data
df = pd.read_csv('data.csv')
target = 'target'  # Binary target (0/1)

# Initialize Binner
binner = Binner()

# Fit binning model
# Auto-selects numeric columns and bins them using ChiMerge
binner.fit(df, y=target, method='chi', n_bins=5)

# Transform data to bin codes (0, 1, 2...)
df_binned = binner.transform(df, labels=False)

# Transform data to interval strings ('(-inf, 0.5]', etc.)
df_labels = binner.transform(df, labels=True)

# Export binning rules
rules = binner.export()
# Output example: {'age': {'splits': [25.0, 40.0, 55.0], 'woe': {...}, 'iv': 0.42}, ...}
```

### Accessing Binning Results

```python
# Access binning result for a specific feature using __getitem__
result = binner['age']

# View statistics DataFrame
print(result.stats)

# Plot binning results
result.plot()

# Get WOE mapping
print(result.woe_map())

# Recommended split editing workflow (instead of touching internal binners_)
print(binner.get_splits('age'))
binner.set_splits('age', [25.0, 40.0, 55.0])

# Loop through all features
for feat in binner:
    print(f"Feature: {feat}")
    print(binner[feat].stats)
```

### Binning Specific Features

```python
# Bin specific columns only
binner.fit(df, y=target, method='dt', n_bins=5, cols=['age', 'income'])

# Load pre-defined rules
custom_rules = {'age': [30.0, 50.0, 70.0]}
binner.load(custom_rules)

# Get summary statistics for a feature
stats = binner.stats('age')
```

### Monotonic Binning

Monotonic binning ensures that the bad rate (event rate) changes monotonically across bins.

```python
# Auto-detect monotonic trend from data
binner.fit(df, y=target, method='chi', monotonic=True)

# Force ascending bad rate trend
binner.fit(df, y=target, method='dt', monotonic='ascending')

# Force descending bad rate trend
# Requires the optional optbinning extra
binner.fit(df, y=target, method='opt', monotonic='descending')
```

**Note**: Monotonic adjustment uses PAVA (Pool Adjacent Violators Algorithm) to merge bins until monotonicity is achieved.

### Using Individual Binner Classes

```python
from newt.features.binning import (
    ChiMergeBinner,
    DecisionTreeBinner,
    OptBinningBinner,
    EqualWidthBinner,
    EqualFrequencyBinner,
    KMeansBinner
)

# Individual binner for a single feature
binner = ChiMergeBinner(n_bins=5)
binner.fit(df['age'], df[target])
bins = binner.transform(df['age'])
```

---

## 2. Feature Selection

`newt.features.selection.FeatureSelector` provides comprehensive feature analysis and selection capabilities.

### Pre-Filtering (EDA-based)

```python
from newt.features.selection import FeatureSelector

# Initialize with desired metrics
selector = FeatureSelector(
    metrics=['iv', 'missing_rate', 'ks', 'correlation'],
    iv_bins=10
)

# Calculate statistics
selector.fit(df, df[target])

# View analysis report
print(selector.report())

# View correlation matrix
print(selector.corr_matrix)

# Apply filtering with thresholds
selector.select(
    iv_threshold=0.02,           # Keep features with IV >= 0.02
    missing_threshold=0.9,       # Remove features with >90% missing
    corr_threshold=0.8           # Remove highly correlated pairs
)

# Get selected features
print(f"Selected: {selector.selected_features_}")

# Transform data
X_filtered = selector.transform(df)
```

### Stepwise Selection

`StepwiseSelector` features a high-performance **Rust parallel engine** powered by `Rayon`. It can test hundreds of candidate features in parallel across multiple CPU cores, achieving up to **20x-40x speedups** over traditional implementations.

```python
from newt.features.selection import StepwiseSelector

# Initialize with the high-performance Rust parallel engine (default)
stepwise = StepwiseSelector(
    direction='both',    # 'forward', 'backward', or 'both'
    criterion='aic',     # 'pvalue', 'aic', or 'bic'
    p_enter=0.05,
    p_remove=0.10,
    engine='rust',       # 'rust' (parallel) or 'python' (statsmodels serial)
    verbose=True         # Show tqdm progress bar
)

# Fit and transform (typically on transformed data)
# This will show a real-time progress bar for each selection step
X_selected = stepwise.fit_transform(X_transformed, y)

print(f"Selected features: {stepwise.selected_features_}")
```

For more details on performance gains, see the [Stepwise Performance Benchmark](benchmarks/stepwise_performance.md).

### Post-Filtering (Model-based)

```python
from newt.features.selection import PostFilter

# Post-filter using PSI and VIF
postfilter = PostFilter(
    psi_threshold=0.25,   # Remove features with PSI > 0.25
    vif_threshold=10.0    # Remove features with VIF > 10.0
)

# Fit on training data and test data for PSI
X_filtered = postfilter.fit_transform(X_train_transformed, X_test_transformed)

print(f"Removed by PSI: {postfilter.psi_removed_}")
print(f"Removed by VIF: {postfilter.vif_removed_}")
```

---

## 3. Binned Feature Analysis

Use `Binner` to inspect bin statistics, derive transformed feature matrices, and feed downstream modeling steps.

### Inspect Feature Statistics

```python
binner = Binner()
binner.fit(df, y=target, method='chi')

result = binner['age']
print(result.stats)
result.plot()
```

### Transform Features For Modeling

```python
# Transform all fitted features directly with the binner
X_transformed = binner.woe_transform(df)
```

### Review Binning Summary

```python
for feat in binner:
    print(feat)
    print(binner[feat].stats)
```

---

## 4. Logistic Regression Modeling

`newt.modeling.LogisticModel` provides a scikit-learn-like interface for logistic regression using statsmodels.

### Basic Modeling

```python
from newt.modeling import LogisticModel

# Initialize model
model = LogisticModel(
    fit_intercept=True,
    method='bfgs',          # Optimization method
    maxiter=100
)

# Fit on transformed features
model.fit(X_transformed, df[target])

# Print model summary (similar to R's summary())
print(model.summary())

# Get coefficients DataFrame
coefs = model.get_coefficients()
print(coefs)

# Get significant features (p-value < 0.05)
sig_features = model.get_significant_features(p_threshold=0.05)
print(sig_features)
```

### Prediction

```python
# Predict probabilities
y_pred_proba = model.predict_proba(X_transformed_test)

# Predict class labels (default threshold: 0.5)
y_pred = model.predict(X_transformed_test)

# Custom threshold
y_pred_custom = model.predict(X_transformed_test, threshold=0.3)
```

### Model Export

```python
# Recommended: persist as JSON
model.dump("logistic_model.json")

# Restore lightweight model (no training samples embedded)
restored_model = LogisticModel.load("logistic_model.json")
```

`to_dict/from_dict` remain available for compatibility, but `dump/load` is the recommended path.

### Using with Scikit-learn Models

```python
from sklearn.linear_model import LogisticRegression

# Scikit-learn model
lr = LogisticRegression()
lr.fit(X_transformed, y)

# Can be used directly with Scorecard
scorecard.from_model(lr, binner)
```

### Using with Statsmodels

```python
import statsmodels.api as sm

# Statsmodels model
X_sm = sm.add_constant(X_transformed)
model_sm = sm.Logit(y, X_sm).fit()

# Can be used directly with Scorecard
scorecard.from_model(model_sm, binner)
```

---

## 5. Scorecard Generation

`newt.modeling.Scorecard` converts logistic regression coefficients into a traditional credit scorecard.

### Building a Scorecard

```python
from newt.modeling import Scorecard

# Initialize scorecard
scorecard = Scorecard(
    base_score=600,     # Base score at base odds
    pdo=50,             # Points to double the odds
    base_odds=1/15      # Base odds (good/bad ratio)
)

# Build from fitted model and binner
scorecard.from_model(model, binner)
# Optional (debugging): keep runtime references to original model/binner
# scorecard.from_model(model, binner, keep_training_artifacts=True)

# View scorecard summary
print(scorecard.summary())

# Export complete scorecard
df_scorecard = scorecard.export()
print(df_scorecard)
```

### Calculating Scores

```python
# Calculate scores for new data
# Note: X is raw data (not pre-transformed)
scores = scorecard.score(X_new)

print(f"Scores: {scores}")
```

### Scorecard Export

```python
# Recommended: persist as JSON
scorecard.dump("scorecard.json")

# Restore from JSON
restored_scorecard = Scorecard.load("scorecard.json")
```

`to_dict/from_dict` remain available for compatibility, but `dump/load` is the recommended path.

### Scorecard Parameters

Scorecard points are calculated using the formula:

```
Points = Offset + Factor * ln(odds)
```

Where:
- `Offset = base_score - (pdo / ln(2)) * ln(base_odds)`
- `Factor = pdo / ln(2)`

---

## 6. Complete Pipeline

`newt.pipeline.ScorecardPipeline` provides a fluent API for the complete scorecard development workflow.

### Full Pipeline Example

```python
from newt.pipeline import ScorecardPipeline

# Initialize pipeline with training data
pipeline = (
    ScorecardPipeline(X_train, y_train, X_test, y_test)
    # Step 1: Pre-filtering
    .prefilter(
        iv_threshold=0.02,
        missing_threshold=0.9,
        corr_threshold=0.8
    )
    # Step 2: Binning
    .bin(method='opt', n_bins=5)
    # Step 3: feature transformation
    .woe_transform()
    # Step 4: Post-filtering
    .postfilter(
        psi_threshold=0.25,
        vif_threshold=10.0
    )
    # Step 5: Stepwise selection
    .stepwise(direction='both', criterion='aic')
    # Step 6: Build model
    .build_model()
    # Step 7: Generate scorecard
    .generate_scorecard(base_score=600, pdo=50)
)

# Score new data
scores = pipeline.score(X_new)
```

### Access Intermediate Results

```python
# Pre-filter report
print(pipeline.prefilter_result.report())

# Binning rules
print(pipeline.binner.export())

# Model summary
print(pipeline.model.summary())

# Scorecard summary
print(pipeline.scorecard.summary())

# Pipeline summary
print(pipeline.summary())
```

### Pipeline Summary

```python
# Get complete pipeline execution summary
summary = pipeline.summary()
print(summary)
# Output:
# Pre-filter: 50 features → 20 features
# Binning: ChiMerge with 5 bins
# WOE: Transformed 20 features
# Post-filter: Removed 2 features (PSI)
# Stepwise: Selected 15 features (AIC)
# Model: Logistic Regression
# Scorecard: Base Score=600, PDO=50
```

---

## 7. Visualization

The visualization module provides various plotting functions for model interpretation.

### Binning Visualization

```python
from newt.visualization import plot_binning_result

# Plot binning result for a single feature
fig = plot_binning_result(
    binner=binner,
    X=df,
    y=df[target],
    feature='age',
    figsize=(12, 6)
)

import matplotlib.pyplot as plt
plt.show()
```

### IV Ranking

```python
from newt.visualization import plot_iv_ranking

# Plot top features by IV
fig = plot_iv_ranking(
    iv_dict=pipeline.prefilter_result.eda_summary_.set_index('feature')['iv'].to_dict(),
    top_n=20,
    threshold=0.02
)
plt.show()
```

### PSI Comparison

```python
from newt.visualization import plot_psi_comparison

# Plot PSI values for all features
fig = plot_psi_comparison(
    psi_dict=pipeline.postfilter_result.psi_,
    threshold=0.25
)
plt.show()
```

---

## 8. Manual Adjustment

Sometimes automated binning isn't perfect (e.g., non-monotonic bad rates, business logic constraints). You can manually adjust the splits.

### Step 1: Export Rules

```python
rules = binner.export()
# Output example:
# {
#    'age': {'splits': [25.0, 40.0, 55.0], 'woe': {...}, 'iv': 0.42},
#    'income': {'splits': [50000.0, 100000.0], 'woe': {...}, 'iv': 0.31}
# }
```

### Step 2: Modify Rules

Edit the list of split points:
- **Merge bins**: Remove a split point
- **Split bins**: Add a split point

```python
# Example: Merge the first two bins of 'age' by removing '25.0'
rules['age'] = [40.0, 55.0]

# Example: Add a split point for 'income'
rules['income'] = [30000.0, 75000.0, 150000.0]
```

### Step 3: Load & Re-visualize

```python
# Update existing Binner
binner.load(rules)

# Or create a new one
binner_adjusted = Binner()
binner_adjusted.load(rules)

# Verify with visualization
fig = plot_binning_result(binner_adjusted, df, df[target], 'age')
plt.show()
```

---

## 9. Deployment

Once satisfied with the scorecard, you can save the configuration and use it in production.

### Save Components

```python
import json

# Save binning rules
with open('binning_rules.json', 'w') as f:
    json.dump(binner.export(), f)

# Save lightweight model and scorecard snapshots
model.dump('model_params.json')
scorecard.dump('scorecard.json')
```

### Load and Use in Production

```python
# Load binning rules
with open('binning_rules.json', 'r') as f:
    rules = json.load(f)

# Create binner and load rules
production_binner = Binner()
production_binner.load(rules)

# Load scorecard directly
production_scorecard = Scorecard.load('scorecard.json')

# Optional: load lightweight LogisticModel snapshot for audit/repro checks
production_model = LogisticModel.load('model_params.json')

# Score new data
df_new = pd.read_csv('new_data.csv')
scores = production_scorecard.score(df_new)
```

`to_dict/from_dict` are still supported for compatibility, but `dump/load` is the recommended persistence path.

---

## 10. Metrics

Newt provides comprehensive evaluation metrics for credit risk modeling.

### AUC (Area Under ROC Curve)

```python
from newt.metrics import calculate_auc

auc = calculate_auc(y_true, y_pred_proba)
print(f"AUC: {auc}")
```

### Gini Coefficient

```python
from newt.metrics import calculate_gini

gini = calculate_gini(y_true, y_pred_proba)
print(f"Gini: {gini}")
```

### KS Statistic

```python
from newt.metrics import calculate_ks

ks = calculate_ks(y_true, y_pred_proba)
print(f"KS: {ks}")
```

### Lift Analysis

```python
from newt.metrics import calculate_lift, calculate_lift_at_k

# Lift table by deciles
lift_df = calculate_lift(y_true, y_pred_proba, bins=10)
print(lift_df)

# Lift at top K (e.g., top 10%)
lift_at_10pct = calculate_lift_at_k(y_true, y_pred_proba, k=0.1)
print(f"Lift@10%: {lift_at_10pct}")
```

### PSI (Population Stability Index)

```python
from newt.metrics import (
    calculate_feature_psi_against_base,
    calculate_grouped_psi,
    calculate_psi,
    calculate_psi_batch,
)

# Calculate PSI between training and test distributions
psi_values = calculate_psi(X_train_transformed, X_test_transformed)
print(f"PSI values: {psi_values}")

# Calculate PSI for a single feature
psi_single = calculate_psi(
    X_train_transformed['age'],
    X_test_transformed['age'],
    buckets=10
)
print(f"PSI for age: {psi_single}")

# Batch PSI for multiple groups against one base
psi_batch = calculate_psi_batch(
    expected=X_train_transformed["age"],
    actual_groups=[X_test_transformed["age"], X_oot_transformed["age"]],
    engine="rust",
)
print(f"Batch PSI: {psi_batch}")

# Grouped PSI by month, latest month as reference within each tag
monthly_psi = calculate_grouped_psi(
    data=score_frame,
    group_cols=["month"],
    score_col="score",
    partition_cols=["tag"],
    reference_mode="latest",
    reference_col="month",
    engine="rust",
)
print(monthly_psi.head())

# Business helper: compare many features against a base slice
feature_psi = calculate_feature_psi_against_base(
    data=score_frame,
    feature_cols=["f1", "f2", "f3"],
    base_col="month",
    base_value="202403",
    compare_col="month",
    compare_values=["202401", "202402", "202403"],
    engine="rust",
)
print(feature_psi.head())
```

### VIF (Variance Inflation Factor)

```python
from newt.metrics import calculate_vif

vif_values = calculate_vif(X_transformed)
print(vif_values)
```

### Metric Interpretation Guidelines

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| AUC | > 0.75 | 0.70 - 0.75 | < 0.70 |
| Gini | > 0.50 | 0.40 - 0.50 | < 0.40 |
| KS | > 40 | 30 - 40 | < 30 |
| PSI | < 0.10 | 0.10 - 0.25 | > 0.25 |
| VIF | < 5 | 5 - 10 | > 10 |
| Lift@10% | > 3.0 | 2.0 - 3.0 | < 2.0 |

---

## Complete Example Workflow

```python
import pandas as pd
from newt import Binner
from newt.features.selection import FeatureSelector, PostFilter
from newt.modeling import LogisticModel, Scorecard
from newt.pipeline import ScorecardPipeline
from newt.visualization import plot_binning_result, plot_iv_ranking
from newt.metrics import calculate_auc, calculate_ks

# 1. Load data
df = pd.read_csv('credit_data.csv')
target = 'default'

# 2. Feature selection
selector = FeatureSelector()
selector.fit(df, df[target])
selector.select(iv_threshold=0.02)
X = selector.transform(df)

# 3. Binning with monotonic constraint
binner = Binner()
binner.fit(X, df[target], method='opt', n_bins=5, monotonic=True)

# Access binning results
binner['age'].stats
binner['age'].plot()
X_binned = binner.transform(X, labels=False)

# 4. feature transformation
X_transformed = binner.woe_transform(X)

# 5. Post-filtering
postfilter = PostFilter()
X_transformed = postfilter.fit_transform(X_transformed, X_transformed_test)

# 6. Model building
model = LogisticModel()
model.fit(X_transformed, df[target])
print(model.summary())

# 7. Scorecard generation
scorecard = Scorecard(base_score=600, pdo=50)
scorecard.from_model(model, binner)
print(scorecard.summary())

# 8. Score new data
df_new = pd.read_csv('new_applications.csv')
scores = scorecard.score(df_new)
print(f"Scores: {scores}")

# 9. Calculate metrics
auc = calculate_auc(df[target], model.predict_proba(X_transformed))
ks = calculate_ks(df[target], model.predict_proba(X_transformed))
print(f"AUC: {auc}, KS: {ks}")

# 10. Visualize
fig = plot_iv_ranking(
    iv_dict=selector.eda_summary_.set_index('feature')['iv'].to_dict()
)
```

---

## 11. Excel Model Report

`newt.Report` generates a multi-sheet Excel model report from prepared sample data, a trained model object, and an existing score column. It covers overview, model design, variable analysis, model performance, and optional appendix sheets (dimensional comparison, model comparison, amount metrics, portrait variables) based on your inputs.

```python
from newt import Report

report = Report(
    data=data,
    model=model,
    tag="tag",
    score_col="score_new",
    date_col="obs_date",
    label_list=["target"],
    score_list=["score_old"],
    dim_list=["channel"],
    var_list=["age", "income"],
    prin_bal_amount_col="prin_bal_amount",  # optional; must pair with loan_amount_col
    loan_amount_col="loan_amount",          # optional
    feature_path="./feature_dict.csv",
    report_out_path="./out/model_report.xlsx",
    engine="rust",           # default
    max_workers=8,           # default: min(8, cpu_count)
    parallel_sheets=True,    # default
    memory_mode="compact",   # default: "compact" | "standard"
    metrics_mode="exact",    # default: "exact" | "binned"
)

report.generate()
```

Notes:

- `data` must already include `tag`, `score_col`, `date_col`, `label_list`, and any extra columns referenced by `score_list`, `dim_list`, `var_list`, or `feature_path`
- `model` is used for parameter extraction and feature importance, not for re-scoring
- `tag` is the sample-split column, typically `train`, `test`, `oot`, or `oos`
- `score_col` is the primary score column for the new model
- `date_col` is normalized into `_report_month` in `YYYYMM` format for monthly tables
- `label_list` is the list of target columns; the first label acts as the primary label
- `score_list` is the list of comparison score columns for older or alternative models
- `dim_list` is used for dimensional comparison and is rendered as one table per dimension in the appendix sheet, then mirrored into the overview sheet
- `var_list` is used for portrait-variable comparison and is rendered as one table per variable in the appendix sheet, then mirrored into the overview sheet; it does not drive the variable-analysis sheet
- `feature_path` is an optional feature dictionary CSV. Prepare these 4 headers exactly:

| Variable English Name | Variable Chinese Name | Variable Source | Metric Table English Name |
|---|---|---|---|
| `英文名` | `中文名` | `来源` | `指标表英文名` |
| `thirdparty_info_period1_6` | `近6个月三方查询次数` | `thirdparty` | `thirdparty_info` |

  Compatibility note: legacy dictionaries that still use `表名` are auto-mapped to `指标表英文名`.
- `sheet_list` can optionally select sheets by index `1-5` or by name:
  `1=overview`, `2=model_design`, `3=variable_analysis`, `4=model_performance`, `5=scorecard_details`
- Name selectors include:
  `总览`, `模型设计`, `变量分析`, `模型表现`, `评分卡计算明细`, `分维度对比`, `新老模型对比`, `金额指标`, `画像变量`
- If omitted, all available sheets are generated (availability still depends on inputs, e.g., OOT/tag coverage, benchmark score columns, model family, and amount columns)
- `engine` controls report compute engine: `rust` (default) or `python`
- `max_workers` controls compute parallelism; default is `min(8, cpu_count)`
- `parallel_sheets` enables concurrent sheet computation (Excel write remains sequential)
- `memory_mode` controls runtime memory strategy: `compact` (default) or `standard`. Compact mode significantly reduces memory usage for 10M+ rows by using downcasted types and optimized monthly transformations.
- `metrics_mode` controls metric computation mode: `exact` (default) or `binned` (faster, approximate)
- `prin_bal_amount_col` and `loan_amount_col` can be passed together to enable amount-based metrics and the optional appendix sheet `金额指标`
- If you only need part of the report, pass just the sheet names or indexes you want
- For report development and validation, use `uv sync --group dev`

Optional external runtime config loading:

```python
from newt import load_conf

load_conf("./newt_conf.json")
```

---

## 12. Interactive Table Generation in Jupyter

Beyond the full Excel report, you can also generate individual report modules as Pandas DataFrames directly in an interactive environment like Jupyter Notebooks. The API handles internal transformations (like normalizing dates to month columns) automatically.

### 12.1. Split Metrics (Tag & Month)

Calculate performance metrics grouping by `tag` (e.g. train, oot) and by `month`.

```python
import pandas as pd
from newt import calculate_split_metrics

# Returns two DataFrames: one grouped by tag, one grouped by month
tag_metrics, month_metrics = calculate_split_metrics(
    data=df,
    tag_col='tag',
    date_col='obs_date',      # Only date_col is needed
    label_list=['target'],
    score_col='xgb_score',
    model_name='XGBoost Model',
    score_type='probability',  # 'probability' (higher=more risky) or 'score' (higher=less risky)
    prin_bal_amount_col='prin_bal_amount',  # optional; must be provided with loan_amount_col
    loan_amount_col='loan_amount',          # optional
)

print(tag_metrics)
print(month_metrics)
```

Notes:

- Core output metrics are always headcount-based:
  `总, 好, 坏, 坏占比, KS, AUC, 10%lift, 5%lift, 2%lift, 1%lift`.
- When `prin_bal_amount_col` and `loan_amount_col` are provided together, amount extension metrics are appended:
  `放款金额, 逾期本金, 金额坏占比, 金额AUC, 金额KS, 10%金额lift, 5%金额lift, 2%金额lift, 1%金额lift`.

### 12.2. Dimensional Comparison

Compare metrics split by custom dimensions (only applies to OOT samples typically).

```python
from newt import calculate_dimensional_comparison

# We recommend passing only OOT data for this analysis
dim_metrics = calculate_dimensional_comparison(
    data=df[df['tag'] == 'oot'],
    dim_list=['channel', 'education'],
    label_list=['target'],
    score_model_columns=[('XGBoost Model', 'xgb_score')],
    score_type='probability',
    prin_bal_amount_col='prin_bal_amount',  # optional; must be paired
    loan_amount_col='loan_amount',          # optional
)

print(dim_metrics)
```

Notes:

- Amount extension metrics are appended only when both amount columns are provided:
  `放款金额, 逾期本金, 金额坏占比, 金额AUC, 金额KS, 10%金额lift, 5%金额lift, 2%金额lift, 1%金额lift`.

### 12.3. Model Comparison

Compare the relative performance of a new model against older benchmarks either by `tag` or by `month`.

```python
from newt import calculate_model_comparison

# Head-to-head comparison between two models
compare_metrics = calculate_model_comparison(
    data=df,
    tag_col='tag',
    date_col='obs_date',
    label_list=['target'],
    model_columns=[
        ('New XGBoost Model', 'xgb_score'),
        ('Old Logistic Model', 'old_score')
    ],
    group_mode='month',  # also accepts 'tag'
    score_type='probability',
    prin_bal_amount_col='prin_bal_amount',  # optional; must be paired
    loan_amount_col='loan_amount',          # optional
)

print(compare_metrics)
```

Notes:

- Amount extension metrics are appended only when both amount columns are provided:
  `放款金额, 逾期本金, 金额坏占比, 金额AUC, 金额KS, 10%金额lift, 5%金额lift, 2%金额lift, 1%金额lift`.

### 12.4. Bin Metrics

Calculate bin-level performance metrics directly, with optional amount metrics.

```python
from newt import calculate_bin_metrics

# Quantile bins (default q=10)
bin_metrics = calculate_bin_metrics(
    data=df,
    label_col='target',
    score_col='xgb_score',
    q=10,
    prin_bal_amount_col='prin_bal_amount',  # optional; must be paired
    loan_amount_col='loan_amount',          # optional
)

# Custom split edges
custom_bin_metrics = calculate_bin_metrics(
    data=df,
    label_col='target',
    score_col='xgb_score',
    bins=[-float('inf'), 0.2, 0.5, 0.8, float('inf')],
)
```

Notes:

- `calculate_bin_metrics` keeps the legacy amount output set.
- When amount columns are provided, appended amount columns are:
  `逾期本金, 放款金额, 金额坏占比, 放款金额占比, 逾期本金占比, 金额lift`.

---

## Additional Resources

- **API Documentation**: See individual module docstrings for detailed API reference
- **Examples**: Check the `examples/` directory for more comprehensive examples
- **Configuration**: See `newt.config` for default parameter values
- **Development**: See `AGENTS.md` for development guidelines
