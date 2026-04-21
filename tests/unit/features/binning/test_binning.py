import numpy as np
import pandas as pd
import pytest

import newt.features.binning.binner as binner_module
import newt.features.binning.supervised as supervised_module
from newt.features.binning import Binner
from newt.features.binning.base import BaseBinner
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


def _high_cardinality_chi_data(seed: int = 5, n: int = 100):
    rng = np.random.default_rng(seed)
    X = pd.Series(rng.integers(0, 30, size=n).astype(float))
    prob = 1 / (1 + np.exp(-((X - 15.0) / 2.0)))
    y = pd.Series((rng.random(n) < prob).astype(int))
    return X, y


def _min_samples_regression_data(seed: int = 1, n: int = 500):
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.random(n))
    # Build a jagged target curve that previously produced final bins
    # below min_samples after split-point conversion.
    prob = (
        0.05
        + 0.55 * ((x > 0.15) & (x < 0.25)).astype(float)
        + 0.35 * ((x > 0.45) & (x < 0.55)).astype(float)
        + 0.45 * ((x > 0.75) & (x < 0.82)).astype(float)
    )
    prob = np.clip(prob, 0.01, 0.99)
    y = pd.Series((rng.random(n) < prob).astype(int))
    return x, y


def test_chimerge_respects_n_bins_hard_cap():
    X, y = _high_cardinality_chi_data()
    binner = ChiMergeBinner(n_bins=5)
    binner.fit(X, y)

    assert len(binner.splits_) + 1 <= 5


def test_binner_chi_rejects_missing_target_values():
    frame = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([0, 1, np.nan, 0, 1])

    with pytest.raises(ValueError, match="missing value"):
        Binner().fit(frame, y, method="chi", n_bins=3, show_progress=False)


def test_binner_chi_rejects_non_binary_target_values():
    frame = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([0, 1, 2, 0, 1])

    with pytest.raises(ValueError, match="non-binary"):
        Binner().fit(frame, y, method="chi", n_bins=3, show_progress=False)


def test_chimerge_enforces_min_samples_threshold():
    X, y = _high_cardinality_chi_data()
    frame = pd.DataFrame({"score": X})
    min_samples = 0.30
    expected_min_count = int(np.ceil(min_samples * len(frame)))

    binner = Binner()
    binner.fit(
        frame,
        y,
        method="chi",
        n_bins=8,
        min_samples=min_samples,
        show_progress=False,
    )

    stats = binner["score"].stats
    assert stats["total"].min() >= expected_min_count


def test_chimerge_min_samples_regression_with_monotonic():
    X, y = _min_samples_regression_data()
    frame = pd.DataFrame({"score": X})
    min_samples = 0.15
    expected_min_count = int(np.ceil(min_samples * len(frame)))

    binner = Binner()
    binner.fit(
        frame,
        y,
        method="chi",
        n_bins=10,
        min_samples=min_samples,
        monotonic=True,
        show_progress=False,
    )

    stats = binner["score"].stats
    assert stats["total"].min() >= expected_min_count


@pytest.mark.parametrize(
    "method",
    ["chi", "dt", "quantile", "step", "kmean", "opt"],
)
def test_binner_min_samples_hard_constraint_across_methods(method):
    X, y = _high_cardinality_chi_data(seed=7, n=300)
    frame = pd.DataFrame({"score": X})
    min_samples = 0.10
    expected_min_count = int(np.ceil(min_samples * len(frame)))

    binner = Binner()
    try:
        binner.fit(
            frame,
            y,
            method=method,
            n_bins=8,
            min_samples=min_samples,
            monotonic=True,
            show_progress=False,
        )
    except ImportError:
        if method == "opt":
            pytest.skip("optbinning is not available in this environment")
        raise

    stats = binner["score"].stats
    assert stats["total"].min() >= expected_min_count


def test_binner_min_samples_int_above_non_missing_raises():
    frame = pd.DataFrame({"score": [1.0, 2.0, np.nan, 4.0, 5.0]})
    y = pd.Series([0, 1, 0, 1, 0])

    with pytest.raises(ValueError, match="exceeds non-missing sample count"):
        Binner().fit(
            frame,
            y,
            method="chi",
            n_bins=3,
            min_samples=5,
            show_progress=False,
        )


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


def test_binner_chi_rust_monotonic_with_missing_constant_and_repeated(
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
    rust_binner.fit(
        frame, y, method="chi", n_bins=5, monotonic=True, show_progress=False
    )

    monkeypatch.setattr(supervised_module, "_load_rust_engine", lambda: None)
    monkeypatch.setattr(binner_module, "_load_rust_engine", lambda: None)
    python_binner = Binner()
    python_binner.fit(
        frame, y, method="chi", n_bins=5, monotonic=True, show_progress=False
    )

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


def test_chimerge_rust_exposes_monotonic_apis():
    _require_batch_chimerge_rust()
    rust_module = supervised_module._load_rust_engine()

    assert hasattr(rust_module, "adjust_chi_merge_monotonic_numpy")
    assert hasattr(rust_module, "adjust_batch_chi_merge_monotonic_numpy")


def test_chimerge_single_uses_rust_monotonic_api(binning_data, monkeypatch):
    _require_chimerge_rust()
    rust_module = supervised_module._load_rust_engine()
    X, y = binning_data

    class RustProxy:
        def __init__(self, module):
            self._module = module
            self.single_monotonic_calls = 0

        def calculate_chi_merge_numpy(self, *args, **kwargs):
            return self._module.calculate_chi_merge_numpy(*args, **kwargs)

        def adjust_chi_merge_monotonic_numpy(self, feature, target, splits, monotonic):
            self.single_monotonic_calls += 1
            return sorted(list(set(splits)))

    proxy = RustProxy(rust_module)
    monkeypatch.setattr(supervised_module, "_load_rust_engine", lambda: proxy)

    binner = ChiMergeBinner(n_bins=5, monotonic=True)
    binner.fit(X, y)

    assert proxy.single_monotonic_calls == 1


def test_binner_chi_uses_batch_rust_monotonic_api_for_multi_feature_fit(
    binning_data, monkeypatch
):
    _require_batch_chimerge_rust()
    rust_module = supervised_module._load_rust_engine()
    X, y = binning_data
    frame = pd.DataFrame({"score": X, "score_sq": X**2})

    class RustProxy:
        def __init__(self, module):
            self._module = module
            self.batch_monotonic_calls = 0

        def calculate_batch_chi_merge_numpy(self, *args, **kwargs):
            return self._module.calculate_batch_chi_merge_numpy(*args, **kwargs)

        def calculate_chi_merge_numpy(self, *args, **kwargs):
            return self._module.calculate_chi_merge_numpy(*args, **kwargs)

        def adjust_batch_chi_merge_monotonic_numpy(
            self, features, target, splits_list, monotonic
        ):
            self.batch_monotonic_calls += 1
            return splits_list, [True] * len(splits_list)

    proxy = RustProxy(rust_module)
    monkeypatch.setattr(binner_module, "_load_rust_engine", lambda: proxy)
    monkeypatch.setattr(supervised_module, "_load_rust_engine", lambda: proxy)

    binner = Binner()
    binner.fit(frame, y, method="chi", n_bins=5, monotonic=True, show_progress=False)

    assert proxy.batch_monotonic_calls == 1


def test_binner_chi_batch_rust_monotonic_failure_falls_back_per_column(
    binning_data, monkeypatch
):
    _require_batch_chimerge_rust()
    rust_module = supervised_module._load_rust_engine()
    X, y = binning_data
    frame = pd.DataFrame({"score": X, "score_sq": X**2})

    class RustProxy:
        def __init__(self, module):
            self._module = module

        def calculate_batch_chi_merge_numpy(self, *args, **kwargs):
            return self._module.calculate_batch_chi_merge_numpy(*args, **kwargs)

        def calculate_chi_merge_numpy(self, *args, **kwargs):
            return self._module.calculate_chi_merge_numpy(*args, **kwargs)

        def adjust_batch_chi_merge_monotonic_numpy(
            self, features, target, splits_list, monotonic
        ):
            if len(splits_list) < 2:
                return splits_list, [False] * len(splits_list)
            return [splits_list[0], sorted(list(set(splits_list[1])))], [False, True]

    fallback_calls = []
    original_adjust = BaseBinner._adjust_monotonicity

    def spy_adjust(self, X, y, splits):
        fallback_calls.append(self)
        return original_adjust(self, X, y, splits)

    proxy = RustProxy(rust_module)
    monkeypatch.setattr(binner_module, "_load_rust_engine", lambda: proxy)
    monkeypatch.setattr(supervised_module, "_load_rust_engine", lambda: proxy)
    monkeypatch.setattr(BaseBinner, "_adjust_monotonicity", spy_adjust)

    binner = Binner()
    binner.fit(frame, y, method="chi", n_bins=5, monotonic=True, show_progress=False)

    assert len(fallback_calls) >= 1


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
