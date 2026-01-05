
import pytest
import pandas as pd
import numpy as np
from src.newt.features.binning.unsupervised import EqualWidthBinner, EqualFrequencyBinner
from src.newt.features.binning.supervised import DecisionTreeBinner, ChiMergeBinner

@pytest.fixture
def binning_data():
    np.random.seed(42)
    n = 200
    X = np.sort(np.random.rand(n)) # Sorted X for easy checking
    # Target: Monotonic sigmoid with some noise
    prob = 1 / (1 + np.exp(-(X - 0.5) * 10))
    # Make it slightly non-monotonic locally by flipping some labels
    target = (np.random.rand(n) < prob).astype(int)
    # Inject noise in middle
    target[80:90] = 1 - target[80:90] 
    
    return pd.Series(X), pd.Series(target)

def test_equal_width(binning_data):
    X, _ = binning_data
    binner = EqualWidthBinner(n_bins=5)
    binner.fit(X)
    
    assert binner.is_fitted_
    assert len(binner.splits_) <= 5 # Internal splits < n_bins
    # Check bounds roughly
    assert 0.0 < binner.splits_[0] < 1.0
    
    binned = binner.transform(X)
    assert len(binned.unique()) <= 5

def test_equal_frequency(binning_data):
    X, _ = binning_data
    binner = EqualFrequencyBinner(n_bins=5)
    binner.fit(X)
    
    binned = binner.transform(X)
    counts = binned.value_counts()
    # Should be roughly equal (allow variation due to duplicates handling or n)
    assert counts.std() < 20 # reasonable bound for n=200, mean=40

def test_decision_tree(binning_data):
    X, y = binning_data
    binner = DecisionTreeBinner(n_bins=5)
    binner.fit(X, y)
    
    assert binner.is_fitted_
    assert len(binner.splits_) + 1 <= 5
    
    binned = binner.transform(X)
    # Check IV (should be decent)
    # (Just basic functionality check)

def test_chimerge(binning_data):
    X, y = binning_data
    binner = ChiMergeBinner(n_bins=5)
    binner.fit(X, y)
    
    assert binner.is_fitted_
    assert len(binner.splits_) + 1 <= 5

def test_monotonicity_adjustment(binning_data):
    X, y = binning_data
    # 1. Fit without monotonic
    binner = EqualFrequencyBinner(n_bins=10, force_monotonic=False)
    binner.fit(X, y)
    
    splits_non_mono = binner.splits_
    
    # 2. Fit with monotonic
    binner_mono = EqualFrequencyBinner(n_bins=10, force_monotonic=True)
    binner_mono.fit(X, y)
    
    splits_mono = binner_mono.splits_
    
    # Expect mono splits to be subset of original or different, 
    # but produce monotonic event rates.
    
    # Helper to check monotonicity
    def is_monotonic(binner, X, y):
        binned = binner.transform(X)
        df = pd.DataFrame({'bin': binned, 'target': y})
        rates = df.groupby('bin', observed=True)['target'].mean().values
        # Drop nans if any
        rates = rates[~np.isnan(rates)]
        if len(rates) < 2: return True
        increasing = np.all(np.diff(rates) >= -1e-6) # float tolerance
        decreasing = np.all(np.diff(rates) <= 1e-6)
        return increasing or decreasing
        
    assert is_monotonic(binner_mono, X, y)
    
    # Check if number of bins reduced (since we injected noise)
    # Original splits should result in non-monotonicity due to noise at index 80-90
    # So adjustment should have merged bins.
    # Note: 10 bins on 200 samples = 20 per bin.
    # Noise at 80-90 is half a bin or full bin.
    assert len(splits_mono) <= len(splits_non_mono)

def test_manual_splits():
    X = pd.Series(np.arange(100))
    binner = EqualWidthBinner()
    binner.set_splits([20, 50, 80])
    
    binned = binner.transform(X)
    assert binned.nunique() == 4 # (-inf, 20], (20, 50], (50, 80], (80, inf]
    # Check boundaries
    assert binned.iloc[0] == pd.Interval(-np.inf, 20.0, closed='right')
