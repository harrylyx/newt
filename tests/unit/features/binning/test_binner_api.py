import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from newt.features.binning import Binner


@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        'age': [20, 30, 40, 50, 60] * 20,
        'income': [50000, 60000, 70000, 80000, 90000] * 20,
    })
    y = pd.Series([0, 0, 1, 1, 1] * 20)
    return df, y


def test_binner_getitem_returns_proxy(sample_data):
    X, y = sample_data
    binner = Binner()
    binner.fit(X, y, method='step', n_bins=5)
    
    # Test dictionary access
    result = binner['age']
    assert result.__class__.__name__ == 'BinningResult'
    assert hasattr(result, 'stats')
    assert hasattr(result, 'plot')
    assert hasattr(result, 'woe_map')
    
    # Check stats content
    assert isinstance(result.stats, pd.DataFrame)
    assert 'bin' in result.stats.columns
    assert 'woe' in result.stats.columns


def test_binner_global_reports(sample_data):
    X, y = sample_data
    binner = Binner()
    binner.fit(X, y, method='step', n_bins=5)
    
    # Test .stats() returns Dict[str, pd.DataFrame]
    all_stats = binner.stats()
    assert isinstance(all_stats, dict)
    assert 'age' in all_stats
    assert 'income' in all_stats
    assert isinstance(all_stats['age'], pd.DataFrame)
    
    # Test .woe_map()
    all_woes = binner.woe_map()
    assert isinstance(all_woes, dict)
    assert 'age' in all_woes


@patch('newt.visualization.binning_viz.plot_binning_result')
def test_binner_proxy_plot(mock_plot, sample_data):
    X, y = sample_data
    binner = Binner()
    binner.fit(X, y, method='step', n_bins=5)
    
    # Call plot on proxy with default args
    binner['age'].plot()
    assert mock_plot.called
    
    # Call plot with custom args
    binner['age'].plot(x='total_prop', y='bad_rate')
    # Verify kwargs passed to viz function
    call_kwargs = mock_plot.call_args[1]
    assert call_kwargs['x_col'] == 'total_prop'
    assert call_kwargs['y_col'] == 'bad_rate'


@patch('newt.visualization.binning_viz.plot_binning_result')
def test_binner_stats_plot(mock_plot, sample_data):
    X, y = sample_data
    binner = Binner()
    binner.fit(X, y, method='step', n_bins=5)
    
    # Call global stats_plot
    # We expect it to print/display tables and call plot for each feature
    binner.stats_plot()
    
    # Verify plot was called for each feature (2 features)
    assert mock_plot.call_count == 2

