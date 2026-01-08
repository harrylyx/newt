# Newt Library User Guide

This guide covers the end-to-end workflow for credit scorecard development using `newt`, including binning, feature selection, WOE/IV analysis, modeling, and scorecard generation.

## Table of Contents

1. [Feature Binning](#1-feature-binning)
2. [Feature Selection](#2-feature-selection)
3. [WOE & IV Analysis](#3-woe--iv-analysis)
4. [Logistic Regression Modeling](#4-logistic-regression-modeling)
5. [Scorecard Generation](#5-scorecard-generation)
6. [Complete Pipeline](#6-complete-pipeline)
7. [Visualization](#7-visualization)
8. [Manual Adjustment](#8-manual-adjustment)
9. [Deployment](#9-deployment)

---

## 1. Feature Binning

The core class for binning is `newt.Binner` (or `newt.features.binning.Binner`). It provides a unified interface for various binning algorithms.

### Supported Methods

- `'chi'`: ChiMerge (Chi-square binning) - **Default**, supervised
- `'dt'`: Decision Tree binning - Supervised, finds optimal splits
- `'opt'`: Optimal Binning via Constraints - Supervised, requires `optbinning`
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
# Output example: {'age': [25.0, 40.0, 55.0], 'income': [50000.0, 100000.0]}
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

```python
from newt.features.selection import StepwiseSelector

# Initialize stepwise selector
stepwise = StepwiseSelector(
    direction='both',    # 'forward', 'backward', or 'both'
    criterion='aic',     # 'pvalue', 'aic', or 'bic'
    p_enter=0.05,
    p_remove=0.10
)

# Fit and transform (typically on WOE-transformed data)
X_selected = stepwise.fit_transform(X_woe, y)

print(f"Selected features: {stepwise.selected_features_}")
```

### Post-Filtering (Model-based)

```python
from newt.features.selection import PostFilter

# Post-filter using PSI and VIF
postfilter = PostFilter(
    psi_threshold=0.25,   # Remove features with PSI > 0.25
    vif_threshold=10.0    # Remove features with VIF > 10.0
)

# Fit on training data and test data for PSI
X_filtered = postfilter.fit_transform(X_train_woe, X_test_woe)

print(f"Removed by PSI: {postfilter.psi_removed_}")
print(f"Removed by VIF: {postfilter.vif_removed_}")
```

---

## 3. WOE & IV Analysis

`newt.features.analysis.WOEEncoder` handles Weight of Evidence (WOE) and Information Value (IV) calculation.

### Basic WOE Encoding

```python
from newt.features.analysis import WOEEncoder

# Fit on binned data
encoder = WOEEncoder(epsilon=1e-8)
encoder.fit(df_binned['feature_name'], df[target])

# Get IV
print(f"IV: {encoder.iv_}")

# Get summary statistics (Good/Bad distribution, WOE, IV contribution)
print(encoder.summary_)

# Transform to WOE values
df_woe = encoder.transform(df_binned['feature_name'])

# Get WOE mapping dictionary
print(encoder.woe_map_)
```

### Batch WOE Transformation

```python
# Apply WOE to all binned features
X_woe = df_binned.copy()
woe_encoders = {}

for col in df_binned.columns:
    encoder = WOEEncoder()
    encoder.fit(df_binned[col].astype(str), df[target])
    woe_encoders[col] = encoder
    X_woe[col] = encoder.transform(df_binned[col].astype(str))
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

# Fit on WOE-transformed features
model.fit(X_woe, df[target])

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
y_pred_proba = model.predict_proba(X_woe_test)

# Predict class labels (default threshold: 0.5)
y_pred = model.predict(X_woe_test)

# Custom threshold
y_pred_custom = model.predict(X_woe_test, threshold=0.3)
```

### Model Export

```python
# Export model parameters as dictionary
model_dict = model.to_dict()
print(model_dict)
# {'intercept': -2.5, 'coefficients': {'age': 0.3, 'income': 0.5}, ...}
```

---

## 5. Scorecard Generation

`newt.modeling.Scorecard` converts WOE-based logistic regression coefficients into a traditional credit scorecard.

### Building a Scorecard

```python
from newt.modeling import Scorecard

# Initialize scorecard
scorecard = Scorecard(
    base_score=600,     # Base score at base odds
    pdo=50,             # Points to double the odds
    base_odds=1/15      # Base odds (good/bad ratio)
)

# Build from fitted model, binner, and WOE encoder
scorecard.from_model(model, binner, woe_encoders)

# View scorecard summary
print(scorecard.summary())

# Export complete scorecard
df_scorecard = scorecard.export()
print(df_scorecard)
```

### Calculating Scores

```python
# Calculate scores for new data
# Note: X is raw data (not binned, not WOE transformed)
scores = scorecard.score(X_new)

print(f"Scores: {scores}")
```

### Scorecard Export

```python
# Export as dictionary
scorecard_dict = scorecard.to_dict()
```

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
    # Step 3: WOE transformation
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
    woe_encoder=woe_encoders.get('age'),
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

### WOE Pattern

```python
from newt.visualization import plot_woe_pattern

# Plot WOE pattern for a feature
fig = plot_woe_pattern(
    woe_encoder=woe_encoders['age'],
    feature='age'
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
#    'age': [25.0, 40.0, 55.0],
#    'income': [50000.0, 100000.0]
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

# Save model parameters
with open('model_params.json', 'w') as f:
    json.dump(model.to_dict(), f)

# Save scorecard
with open('scorecard.json', 'w') as f:
    json.dump(scorecard.to_dict(), f)
```

### Load and Use in Production

```python
# Load binning rules
with open('binning_rules.json', 'r') as f:
    rules = json.load(f)

# Create binner and load rules
production_binner = Binner()
production_binner.load(rules)

# Load model parameters and recreate
with open('model_params.json', 'r') as f:
    model_params = json.load(f)

# Load WOE encoders (you'll need to save/load these separately)
# ... load woe_encoders dict ...

# Recreate scorecard
production_scorecard = Scorecard(
    base_score=600,
    pdo=50,
    base_odds=1/15
)
production_scorecard.from_model(model_params, production_binner, woe_encoders)

# Score new data
df_new = pd.read_csv('new_data.csv')
scores = production_scorecard.score(df_new)
```

---

## Complete Example Workflow

```python
import pandas as pd
from newt import Binner
from newt.features.selection import FeatureSelector, PostFilter
from newt.features.analysis import WOEEncoder
from newt.modeling import LogisticModel, Scorecard
from newt.pipeline import ScorecardPipeline
from newt.visualization import plot_binning_result, plot_iv_ranking

# 1. Load data
df = pd.read_csv('credit_data.csv')
target = 'default'

# 2. Feature selection
selector = FeatureSelector()
selector.fit(df, df[target])
selector.select(iv_threshold=0.02)
X = selector.transform(df)

# 3. Binning
binner = Binner()
binner.fit(X, df[target], method='opt', n_bins=5)
X_binned = binner.transform(X, labels=False)

# 4. WOE transformation
X_woe = X_binned.copy()
woe_encoders = {}
for col in X_binned.columns:
    encoder = WOEEncoder()
    encoder.fit(X_binned[col].astype(str), df[target])
    woe_encoders[col] = encoder
    X_woe[col] = encoder.transform(X_binned[col].astype(str))

# 5. Post-filtering
postfilter = PostFilter()
X_woe = postfilter.fit_transform(X_woe, X_woe_test)

# 6. Model building
model = LogisticModel()
model.fit(X_woe, df[target])
print(model.summary())

# 7. Scorecard generation
scorecard = Scorecard(base_score=600, pdo=50)
scorecard.from_model(model, binner, woe_encoders)
print(scorecard.summary())

# 8. Score new data
df_new = pd.read_csv('new_applications.csv')
scores = scorecard.score(df_new)
print(f"Scores: {scores}")

# 9. Visualize
fig = plot_iv_ranking(
    iv_dict=selector.eda_summary_.set_index('feature')['iv'].to_dict()
)
```

---

## Additional Resources

- **API Documentation**: See individual module docstrings for detailed API reference
- **Examples**: Check the `examples/` directory for more comprehensive examples
- **Configuration**: See `newt.config` for default parameter values
