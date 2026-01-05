# Newt Library User Guide

This guide covers the end-to-end workflow for feature engineering using `newt`, including binning, WOE/IV analysis, visualization, and manual adjustment.

## 1. Feature Binning (分箱)

The core class for binning is `newt.features.binning.Combiner`. It provides a unified interface for various binning algorithms.

### Supported Methods
- `'chi'`: ChiMerge (Chi-square binning) - **Default**, supervised.
- `'dt'`: Decision Tree binning - Supervised, finds optimal splits.
- `'kmean'`: K-Means clustering - Unsupervised.
- `'quantile'`: Equal frequency binning - Unsupervised.
- `'step'`: Equal width binning - Unsupervised.

### Basic Usage

```python
import pandas as pd
from newt.features.binning import Combiner

# Load your data
df = pd.read_csv('data.csv')
target = 'target' # Binary target (0/1)

# Initialize Combiner
c = Combiner()

# Fit binning model
# Auto-selects numeric columns and bins them using ChiMerge
c.fit(df, y=target, method='chi', n_bins=5)

# Transform data to bin codes (0, 1, 2...)
df_binned = c.transform(df, labels=False)

# Transform data to interval strings ('(-inf, 0.5]', etc.)
df_labels = c.transform(df, labels=True)
```

## 2. WOE & IV Analysis

While the `Combiner` handles binning, the `WOEEncoder` handles Weight of Evidence (WOE) and Information Value (IV) calculation.
*Note: The visualization module automatically calculates IV for you.*

If you want to calculate WOE/IV programmatically:

```python
from newt.features.analysis.woe_calculator import WOEEncoder

# Use the binned labels from Combiner
encoder = WOEEncoder()
encoder.fit(df_labels['feature_name'], df[target])

# Get IV
print(f"IV: {encoder.iv_}")

# Get Summary Stats (Good/Bad dist, WOE, IV contrib)
print(encoder.summary_)

# Transform to WOE values
df_woe = encoder.transform(df_labels['feature_name'])
```

## 3. Binning Visualization (可视化)

Use `plot_binning` to visualize the quality of your bins. It generates a combo chart (Histogram + Line) using Plotly.

```python
from newt.visualization import plot_binning

# Plot a specific feature
# Requires the fitted Combiner 'c'
fig = plot_binning(
    combiner=c,
    data=df,
    feature='age',
    target=target,
    bar_mode='total_dist', # Show percentage distribution
    decimals=0,            # Round bin edges to integer
    show_iv=True           # Show IV in title
)

fig.show()
```
**Charts:**
- **Bar**: Sample distribution (Total count or Percentage).
- **Line**: Bad Rate (Target Rate) per bin.
- **Title**: Includes the calculated IV.

## 4. Manual Adjustment (分箱调整)

Sometimes automated binning isn't perfect (e.g., non-monotonic bad rates, business logic constraints). You can manually adjust the splits.

### Step 1: Export Rules
Export the learned splits from the Combiner.
```python
rules = c.export()
# Output example:
# {
#    'age': [25.0, 40.0, 55.0],
#    'income': [50000.0, 100000.0]
# }
```

### Step 2: Modify Rules
Edit the list of split points.
- **Merge bins**: Remove a split point.
- **Split bins**: Add a split point.

```python
# Example: Merge the first two bins of 'age' by removing '25.0'
rules['age'] = [40.0, 55.0]
```

### Step 3: Load & Re-visualize
Load the modified rules back into the Combiner (or a new one) and check the result.

```python
# Update existing combiner
c.load(rules)

# Or create a new one
c_adjusted = Combiner()
c_adjusted.load(rules)

# Verify with visualization
fig = plot_binning(c_adjusted, df, 'age', target)
fig.show()
```

## 5. Deployment / Pipeline

Once satisfied with the binning, you can save the rules (e.g., as JSON) and use `c.load(rules)` in your production pipeline to transform new data.

```python
import json

# Save
with open('binning_rules.json', 'w') as f:
    json.dump(c.export(), f)

# Load in production
with open('binning_rules.json', 'r') as f:
    rules = json.load(f)

production_combiner = Combiner()
production_combiner.load(rules)

# Transform new data
df_new_binned = production_combiner.transform(df_new)
```
