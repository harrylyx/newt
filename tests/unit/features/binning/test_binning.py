import numpy as np
import pandas as pd
import pytest

import newt.features.binning.binner as binner_module
import newt.features.binning.supervised as supervised_module
from newt.features.binning import Binner
from newt.features.binning.supervised import ChiMergeBinner, DecisionTreeBinner
from newt.features.binning.unsupervised import EqualFrequencyBinner, EqualWidthBinner


@pytest.fixture
def binning_data():
    np.random.seed(42)
    n = 200
    X = np.sort(np.random.rand(n))  # Sorted X for easy checking
    # Target: Monotonic sigmoid with some noise
    prob = 1 / (1 + np.exp(-(X - 0.5) * 10))
    # Make it slightly non-monotonic locally by flipping some labels
    target = (np.random.rand(n) < prob).astype(int)
    # Inject noise in middle
    target[80:90] = 1 - target[80:90]

    return pd.Series(X), pd.Series(target)


def _require_chimerge_rust():
    rust_module = supervised_module._load_rust_engine()
    if rust_module is None or not hasattr(rust_module, "calculate_chi_merge_numpy"):
        pytest.skip("Rust ChiMerge engine is not available in this environment")


def _require_batch_chimerge_rust():
    rust_module = supervised_module._load_rust_engine()
    if rust_module is None or not hasattr(
        rust_module, "calculate_batch_chi_merge_numpy"
    ):
        pytest.skip("Rust batch ChiMerge engine is not available in this environment")


def test_equal_width(binning_data):
    X, _ = binning_data
    binner = EqualWidthBinner(n_bins=5)
    binner.fit(X)

    assert binner.is_fitted_
    assert len(binner.splits_) <= 5  # Internal splits < n_bins
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
    assert counts.std() < 20  # reasonable bound for n=200, mean=40


def test_decision_tree(binning_data):
    X, y = binning_data
    binner = DecisionTreeBinner(n_bins=5)
    binner.fit(X, y)

    assert binner.is_fitted_
    assert len(binner.splits_) + 1 <= 5

    assert not binner.transform(X).empty
    # Check IV (should be decent)
    # (Just basic functionality check)


def test_chimerge(binning_data):
    X, y = binning_data
    binner = ChiMergeBinner(n_bins=5)
    binner.fit(X, y)

    assert binner.is_fitted_
    assert len(binner.splits_) + 1 <= 5


def test_chimerge_rust_matches_python_splits(binning_data, monkeypatch):
    _require_chimerge_rust()
    X, y = binning_data

    rust_binner = ChiMergeBinner(n_bins=5)
    rust_binner.fit(X, y)

    monkeypatch.setattr(supervised_module, "_load_rust_engine", lambda: None)
    python_binner = ChiMergeBinner(n_bins=5)
    python_binner.fit(X, y)

    assert rust_binner.splits_ == pytest.approx(python_binner.splits_, abs=1e-12)


def test_chimerge_rust_matches_python_bin_assignments(binning_data, monkeypatch):
    _require_chimerge_rust()
    X, y = binning_data

    rust_binner = ChiMergeBinner(n_bins=5)
    rust_binner.fit(X, y)

    monkeypatch.setattr(supervised_module, "_load_rust_engine", lambda: None)
    python_binner = ChiMergeBinner(n_bins=5)
    python_binner.fit(X, y)

    np.testing.assert_array_equal(
        rust_binner.transform(X).cat.codes.to_numpy(),
        python_binner.transform(X).cat.codes.to_numpy(),
    )


def test_binner_chi_rust_matches_python_public_entrypoint(binning_data, monkeypatch):
    _require_chimerge_rust()
    X, y = binning_data
    frame = pd.DataFrame({"score": X, "score_sq": X**2})

    rust_binner = Binner()
    rust_binner.fit(frame, y, method="chi", n_bins=5, show_progress=False)

    monkeypatch.setattr(supervised_module, "_load_rust_engine", lambda: None)
    monkeypatch.setattr(binner_module, "_load_rust_engine", lambda: None)
    python_binner = Binner()
    python_binner.fit(frame, y, method="chi", n_bins=5, show_progress=False)

    for feature in frame.columns:
        assert rust_binner.rules_[feature] == pytest.approx(
            python_binner.rules_[feature], abs=1e-12
        )

    pd.testing.assert_frame_equal(
        rust_binner.transform(frame, show_progress=False),
        python_binner.transform(frame, show_progress=False),
    )


def test_binner_chi_rust_matches_python_with_monotonic_true(binning_data, monkeypatch):
    _require_chimerge_rust()
    X, y = binning_data
    frame = pd.DataFrame({"score": X, "score_sq": X**2})

    rust_binner = Binner()
    rust_binner.fit(
        frame,
        y,
        method="chi",
        n_bins=5,
        monotonic=True,
        show_progress=False,
    )

    monkeypatch.setattr(supervised_module, "_load_rust_engine", lambda: None)
    monkeypatch.setattr(binner_module, "_load_rust_engine", lambda: None)
    python_binner = Binner()
    python_binner.fit(
        frame,
        y,
        method="chi",
        n_bins=5,
        monotonic=True,
        show_progress=False,
    )

    for feature in frame.columns:
        assert rust_binner.rules_[feature] == pytest.approx(
            python_binner.rules_[feature], abs=1e-12
        )

    pd.testing.assert_frame_equal(
        rust_binner.transform(frame, show_progress=False),
        python_binner.transform(frame, show_progress=False),
    )


def test_binner_chi_rust_matches_python_with_missing_constant_and_repeated(
    binning_data, monkeypatch
):
    _require_chimerge_rust()
    X, y = binning_data
    frame = pd.DataFrame(
        {
            "score": X,
            "score_nan": X.mask(np.arange(len(X)) % 11 == 0),
            "score_repeat": np.round(X, 2),
            "score_const": 1.0,
        }
    )

    rust_binner = Binner()
    rust_binner.fit(frame, y, method="chi", n_bins=5, show_progress=False)

    monkeypatch.setattr(supervised_module, "_load_rust_engine", lambda: None)
    monkeypatch.setattr(binner_module, "_load_rust_engine", lambda: None)
    python_binner = Binner()
    python_binner.fit(frame, y, method="chi", n_bins=5, show_progress=False)

    for feature in frame.columns:
        assert rust_binner.rules_[feature] == pytest.approx(
            python_binner.rules_[feature], abs=1e-12
        )

    pd.testing.assert_frame_equal(
        rust_binner.transform(frame, show_progress=False),
        python_binner.transform(frame, show_progress=False),
    )


def test_binner_chi_uses_batch_rust_api_for_multi_feature_fit(
    binning_data, monkeypatch
):
    _require_batch_chimerge_rust()
    rust_module = supervised_module._load_rust_engine()
    X, y = binning_data
    frame = pd.DataFrame({"score": X, "score_sq": X**2})

    class RustProxy:
        def __init__(self, module):
            self._module = module
            self.batch_calls = 0
            self.single_calls = 0

        def calculate_batch_chi_merge_numpy(self, *args, **kwargs):
            self.batch_calls += 1
            return self._module.calculate_batch_chi_merge_numpy(*args, **kwargs)

        def calculate_chi_merge_numpy(self, *args, **kwargs):
            self.single_calls += 1
            return self._module.calculate_chi_merge_numpy(*args, **kwargs)

    proxy = RustProxy(rust_module)
    monkeypatch.setattr(binner_module, "_load_rust_engine", lambda: proxy)
    monkeypatch.setattr(supervised_module, "_load_rust_engine", lambda: proxy)

    binner = Binner()
    binner.fit(frame, y, method="chi", n_bins=5, show_progress=False)

    assert proxy.batch_calls == 1
    assert proxy.single_calls == 0


def test_monotonicity_adjustment(binning_data):
    X, y = binning_data
    # 1. Fit without monotonic
    binner = EqualFrequencyBinner(n_bins=10, monotonic=False)
    binner.fit(X, y)

    splits_non_mono = binner.splits_

    # 2. Fit with monotonic
    binner_mono = EqualFrequencyBinner(n_bins=10, monotonic=True)
    binner_mono.fit(X, y)

    splits_mono = binner_mono.splits_

    # Expect mono splits to be subset of original or different,
    # but produce monotonic event rates.

    # Helper to check monotonicity
    def is_monotonic(binner, X, y):
        binned = binner.transform(X)
        df = pd.DataFrame({"bin": binned, "target": y})
        rates = df.groupby("bin", observed=True)["target"].mean().values
        # Drop nans if any
        rates = rates[~np.isnan(rates)]
        if len(rates) < 2:
            return True
        increasing = np.all(np.diff(rates) >= -1e-6)  # float tolerance
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
    assert binned.nunique() == 4  # (-inf, 20], (20, 50], (50, 80], (80, inf]
    # Check boundaries
    assert binned.iloc[0] == pd.Interval(-np.inf, 20.0, closed="right")
