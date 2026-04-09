import sys
import types

import pandas as pd
import pytest

from benchmarks.metric_vs_toad import (
    METADATA_COLUMNS,
    apply_reference_bins,
    build_newt_psi_breakpoints,
    compute_toad_results,
    get_feature_columns,
    get_numeric_feature_columns,
    make_serializable,
    parse_args,
    prepare_feature_for_iv,
    render_markdown,
)


def test_get_feature_columns_excludes_metadata():
    frame = pd.DataFrame(
        {
            "idx": [1, 2],
            "target": [0, 1],
            "score_p": [0.1, 0.9],
            "feature_a": [10, 20],
            "feature_b": ["x", "y"],
        }
    )

    features = get_feature_columns(frame)

    assert "feature_a" in features
    assert "feature_b" in features
    for column in METADATA_COLUMNS.intersection(frame.columns):
        assert column not in features


def test_get_numeric_feature_columns_filters_non_numeric():
    frame = pd.DataFrame(
        {
            "target": [0, 1],
            "feature_a": [10.0, 20.0],
            "feature_b": ["x", "y"],
            "feature_c": [1, 2],
        }
    )

    numeric = get_numeric_feature_columns(frame)

    assert numeric == ["feature_a", "feature_c"]


def test_prepare_feature_for_iv_bins_numeric_and_marks_missing():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, None], name="feature")

    prepared = prepare_feature_for_iv(series, buckets=2)

    assert prepared.iloc[-1] == "Missing"
    assert prepared.nunique() == 3


def test_build_newt_psi_breakpoints_and_apply_reference_bins_keep_missing_bucket():
    breakpoints = build_newt_psi_breakpoints([1.0, 2.0, 3.0, None], buckets=2)
    binned = apply_reference_bins([1.0, 3.0, None], breakpoints)

    assert breakpoints[0] == float("-inf")
    assert breakpoints[-1] == float("inf")
    assert list(binned.astype(str))[-1] == "Missing"


def test_render_markdown_mentions_direct_benchmark_environment():
    report = {
        "metadata": {
            "generated_at": "2026-04-04T12:00:00",
            "data_path": "examples/data/test_data/all_data.pq",
            "dataset": {
                "binary_rows": 10,
                "split_rows": {"train": 5, "test": 3, "oot": 2},
                "feature_count": 4,
                "numeric_feature_count": 3,
            },
            "newt_runtime": {"python": "3.12.12", "platform": "macOS"},
            "toad_runtime": {
                "status": "ok",
                "python": "3.10.19",
                "executable": (
                    "/Users/cabbage/Project/newt/.venv-benchmark-3.10/bin/python"
                ),
                "toad_version": "0.1.5",
            },
            "toad_status": "ok",
        },
        "correctness": {
            "auc": {"rows": [], "summary": {"mean_abs_diff": 0.0, "max_abs_diff": 0.0}},
            "ks": {"rows": [], "summary": {"mean_abs_diff": 0.0, "max_abs_diff": 0.0}},
            "psi_scores": {
                "rows": [],
                "summary": {"mean_abs_diff": 0.0, "max_abs_diff": 0.0},
            },
            "iv": {
                "rows": [],
                "summary": {"mean_abs_diff": 0.0, "max_abs_diff": 0.0},
                "top_diffs": [],
            },
            "psi_features": {
                "rows": [],
                "summary": {"mean_abs_diff": 0.0, "max_abs_diff": 0.0},
                "top_diffs": [],
            },
        },
        "efficiency": {"paired": [], "iv": []},
    }

    markdown = render_markdown(report)

    assert "Toad Python: `3.10.19`" in markdown
    assert "benchmark requested `3.12`" not in markdown


def test_parse_args_uses_one_benchmark_environment_only():
    args = parse_args([])

    assert args.repeat == 5
    assert args.warmup == 1
    assert args.iv_buckets == 10
    assert args.psi_buckets == 10
    assert not hasattr(args, "toad_version")
    assert not hasattr(args, "toad_python")
    assert not hasattr(args, "toad_fallback_python")
    assert not hasattr(args, "worker")
    assert not hasattr(args, "result_json")


def test_compute_toad_results_uses_current_environment(monkeypatch):
    fake_toad = types.ModuleType("toad")
    fake_toad.__version__ = "0.1.5"
    fake_toad.metrics = types.SimpleNamespace(
        AUC=lambda values, labels: float(values.mean() + labels.mean()),
        KS=lambda values, labels: float(values.mean() - labels.mean()),
        PSI=lambda actual, expected: float(len(actual) + len(expected)),
    )

    def fake_quality(frame, target, iv_only):
        feature_names = [column for column in frame.columns if column != target]
        return pd.DataFrame(
            {"iv": [0.11 * (index + 1) for index, _ in enumerate(feature_names)]},
            index=feature_names,
        )

    fake_toad.quality = fake_quality
    monkeypatch.setitem(sys.modules, "toad", fake_toad)

    binary = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "flag_train": [True, True, False, False],
            "flag_test": [False, False, True, False],
            "flag_oot": [False, False, False, True],
            "score_p": [0.1, 0.9, 0.2, 0.8],
            "score_old_p": [0.2, 0.8, 0.3, 0.7],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": ["a", "b", "a", "b"],
        }
    )
    splits = {
        "train": binary.loc[binary["flag_train"]].copy(),
        "test": binary.loc[binary["flag_test"]].copy(),
        "oot": binary.loc[binary["flag_oot"]].copy(),
    }

    result = compute_toad_results(
        binary=binary,
        splits=splits,
        feature_columns=["feature_a", "feature_b"],
        numeric_features=["feature_a"],
        repeat=1,
        warmup=0,
        iv_buckets=2,
        psi_buckets=2,
    )

    meta = result["metadata"]["toad_runtime"]

    assert meta["status"] == "ok"
    assert meta["python"] == sys.version.split()[0]
    assert meta["executable"] == sys.executable
    assert meta["toad_version"] == "0.1.5"
    assert "requested_python" not in meta
    assert "fallback_used" not in meta
    assert result["correctness"]["iv"][0]["value"] == pytest.approx(0.11)


def test_make_serializable_keeps_boolean_values():
    assert make_serializable(True) is True
