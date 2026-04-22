from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import newt.features.analysis.iv_calculator as iv_calculator_module
from newt.features.analysis.iv_calculator import calculate_iv
from newt.features.analysis.iv_math import prepare_feature_for_iv


def _toad_iv(df: pd.DataFrame, feature: str) -> float:
    toad = pytest.importorskip("toad")
    raw = toad.quality(df[[feature, "target"]], target="target", iv_only=True)
    if isinstance(raw, pd.DataFrame):
        if "iv" in raw.columns:
            iv_series = raw["iv"]
        else:
            iv_series = raw.squeeze(axis=1)
    else:
        iv_series = raw
    return float(iv_series.get(feature, np.nan))


def _build_small_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [0.1, 0.2, 0.2, 0.5, 0.9, 1.1, 1.5, np.nan, 2.0, 2.0, 2.2, 2.8],
            "cat": [
                "a",
                "b",
                "b",
                "b",
                "c",
                "c",
                "d",
                None,
                "d",
                "d",
                "e",
                "e",
            ],
            "target": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        }
    )


@pytest.mark.parametrize("feature", ["num", "cat"])
@pytest.mark.parametrize("engine", ["python", "rust"])
def test_calculate_iv_matches_toad_on_prepared_data(feature: str, engine: str):
    df = _build_small_dataset()
    prepared = prepare_feature_for_iv(df[feature], buckets=5)
    iv_df = pd.DataFrame({feature: prepared, "target": df["target"]})
    expected = _toad_iv(iv_df, feature=feature)

    if engine == "rust":
        try:
            result = calculate_iv(
                df,
                target="target",
                feature=feature,
                buckets=5,
                engine="rust",
            )
        except ImportError:
            pytest.skip("Rust IV extension is not available in this environment")
    else:
        result = calculate_iv(
            df,
            target="target",
            feature=feature,
            buckets=5,
            engine="python",
        )

    assert result["iv"] == pytest.approx(expected, abs=1e-9)


def test_calculate_iv_rust_matches_python():
    df = _build_small_dataset()

    try:
        rust_numeric = calculate_iv(
            df,
            target="target",
            feature="num",
            buckets=5,
            engine="rust",
        )
        rust_categorical = calculate_iv(
            df,
            target="target",
            feature="cat",
            buckets=5,
            engine="rust",
        )
    except ImportError:
        pytest.skip("Rust IV extension is not available in this environment")

    py_numeric = calculate_iv(
        df,
        target="target",
        feature="num",
        buckets=5,
        engine="python",
    )
    py_categorical = calculate_iv(
        df,
        target="target",
        feature="cat",
        buckets=5,
        engine="python",
    )

    assert rust_numeric["iv"] == pytest.approx(py_numeric["iv"], abs=1e-9)
    assert rust_categorical["iv"] == pytest.approx(py_categorical["iv"], abs=1e-9)


def test_calculate_iv_default_auto_matches_python_when_native_unavailable(monkeypatch):
    df = _build_small_dataset()
    monkeypatch.setattr(iv_calculator_module, "load_native_module", lambda: None)

    default_result = calculate_iv(df, target="target", feature="cat", buckets=5)
    python_result = calculate_iv(
        df,
        target="target",
        feature="cat",
        buckets=5,
        engine="python",
    )
    assert default_result["iv"] == pytest.approx(python_result["iv"], abs=1e-9)


def test_userinfo_24_iv_regression_matches_toad():
    data_path = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "data"
        / "test_data"
        / "all_data.pq"
    )
    if not data_path.exists():
        pytest.skip("Benchmark dataset is not available")

    df = pd.read_parquet(data_path, columns=["userinfo_24", "target"])
    df = df.loc[df["target"].isin([0, 1])].copy()
    df["target"] = df["target"].astype(int)

    prepared = prepare_feature_for_iv(df["userinfo_24"], buckets=10)
    iv_df = pd.DataFrame({"userinfo_24": prepared, "target": df["target"]})
    expected = _toad_iv(iv_df, feature="userinfo_24")

    try:
        rust_result = calculate_iv(
            df,
            target="target",
            feature="userinfo_24",
            buckets=10,
            engine="rust",
        )
    except ImportError:
        pytest.skip("Rust IV extension is not available in this environment")

    py_result = calculate_iv(
        df,
        target="target",
        feature="userinfo_24",
        buckets=10,
        engine="python",
    )

    assert rust_result["iv"] == pytest.approx(expected, abs=1e-9)
    assert py_result["iv"] == pytest.approx(expected, abs=1e-9)
