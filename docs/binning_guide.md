# Binning and Visualization Guide

This guide describes how to use the `newt` library's binning module and visualization tools.

## 1. Feature Binning

The `Combiner` class provides a unified interface for binning features using various algorithms. It supports both supervised and unsupervised binning methods.

### Import
```python
from newt.features.binning import Combiner
```

### Initialization
```python
c = Combiner()
```

### Methods

#### `fit`
Fits the binning model to data.
```python
c.fit(
    X: pd.DataFrame, 
    y: Optional[Union[pd.Series, str]] = None, 
    method: str = "chi", 
    n_bins: int = 5,
    min_samples: Union[int, float] = None,
    exclude: List[str] = None
)
```
- **method**:
    - `'chi'`: ChiMerge (Supervised, default).
    - `'dt'`: Decision Tree (Supervised).
    - `'kmean'`: K-Means Clustering (Unsupervised).
    - `'quantile'`: Equal Frequency (Unsupervised).
    - `'step'`: Equal Width (Unsupervised).
- **n_bins**: Target number of bins.
- **min_samples**: Minimum samples per leaf (for `'dt'` method).

#### `transform`
Applies binning rules to new data.
```python
# Return bin codes (0, 1, 2...)
df_codes = c.transform(df, labels=False)

# Return bin intervals as strings
df_labels = c.transform(df, labels=True)
```

#### `export` & `load`
Export rules to a dictionary or load them.
```python
# Export
rules = c.export()
# rules = {'feature_name': [split1, split2, ...]}

# Load
c_new = Combiner()
c_new.load(rules)
```

### Example
```python
import pandas as pd
from newt.features.binning import Combiner

# Load data
df = pd.read_csv('data.csv')

# Fit
c = Combiner()
c.fit(df, y='target', method='chi', n_bins=5)

# Transform
df_binned = c.transform(df, labels=True)
```

---

## 2. Visualization

The `plot_binning` function visualizes the binning results, showing the distribution of samples and the target rate (bad rate) for each bin. It also displays the Information Value (IV).

### Import
```python
from newt.visualization import plot_binning
```

### Usage
```python
plot_binning(
    combiner: Combiner,
    data: pd.DataFrame,
    feature: str,
    target: str,
    labels: bool = True,
    show_iv: bool = True,
    decimals: int = 2,
    bar_mode: str = 'total'
)
```

- **combiner**: A fitted `Combiner` object.
- **data**: DataFrame containing feature and target.
- **feature**: Name of the feature to plot.
- **target**: Name of the target column.
- **decimals**: Number of decimals to round bin intervals.
- **bar_mode**:
    - `'total'`: Show total count per bin.
    - `'bad'`: Show bad count per bin.
    - `'total_dist'`: Show percentage of total samples.
    - `'bad_dist'`: Show percentage of bad samples.

### Output
Returns a Plotly `Figure` object which can be displayed in a notebook or saved to HTML.

### Example
```python
# Assuming 'c' is a fitted Combiner
fig = plot_binning(
    combiner=c,
    data=df,
    feature='age',
    target='default_flag',
    decimals=0,
    bar_mode='total_dist'
)

fig.show()
```
![Example Plot](https://via.placeholder.com/800x400?text=Bar+Line+Combo+Chart)
*(Chart includes Bar for distribution and Line for bad rate, with IV in title)*
