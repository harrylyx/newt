import pandas as pd

from newt.benchmarking.metric_vs_toad import (
    METADATA_COLUMNS,
    apply_reference_bins,
    build_newt_psi_breakpoints,
    get_feature_columns,
    get_numeric_feature_columns,
    make_serializable,
    prepare_feature_for_iv,
    render_markdown,
    resolve_python_interpreter,
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


def test_render_markdown_mentions_toad_fallback():
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
                "requested_python": "3.12",
                "python": "3.10.19",
                "used_python": "/tmp/python3.10",
                "toad_version": "0.1.5",
                "fallback_used": True,
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

    assert "fell back to a lower Python version" in markdown


def test_make_serializable_keeps_boolean_values():
    assert make_serializable(True) is True


def test_resolve_python_interpreter_prefers_uv_version_lookup(monkeypatch):
    uv_python = "/tmp/uv-python3.10"
    broken_shim = "/tmp/pyenv-python3.10"

    def fake_which(name):
        mapping = {
            "uv": "/usr/bin/uv",
            "python3.10": broken_shim,
        }
        return mapping.get(name)

    class Completed:
        def __init__(self, returncode, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, capture_output, text, check):
        if command == ["/usr/bin/uv", "python", "find", "3.10"]:
            return Completed(0, "{}\n".format(uv_python))
        if command == [uv_python, "-c", "import sys; print(sys.executable)"]:
            return Completed(0, "{}\n".format(uv_python))
        raise AssertionError("unexpected command: {}".format(command))

    monkeypatch.setattr("newt.benchmarking.metric_vs_toad.shutil.which", fake_which)
    monkeypatch.setattr("newt.benchmarking.metric_vs_toad.subprocess.run", fake_run)

    assert resolve_python_interpreter("3.10") == uv_python


def test_resolve_python_interpreter_falls_back_when_uv_python_is_broken(monkeypatch):
    broken_uv_python = "/tmp/uv-python3.10"
    direct_python = "/tmp/direct-python3.10"

    def fake_which(name):
        mapping = {
            "uv": "/usr/bin/uv",
            "python3.10": direct_python,
        }
        return mapping.get(name)

    class Completed:
        def __init__(self, returncode, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, capture_output, text, check):
        if command == ["/usr/bin/uv", "python", "find", "3.10"]:
            return Completed(0, "{}\n".format(broken_uv_python))
        if command == [broken_uv_python, "-c", "import sys; print(sys.executable)"]:
            return Completed(1, "", "broken")
        if command == [direct_python, "-c", "import sys; print(sys.executable)"]:
            return Completed(0, "{}\n".format(direct_python))
        raise AssertionError("unexpected command: {}".format(command))

    monkeypatch.setattr("newt.benchmarking.metric_vs_toad.shutil.which", fake_which)
    monkeypatch.setattr("newt.benchmarking.metric_vs_toad.subprocess.run", fake_run)

    assert resolve_python_interpreter("3.10") == direct_python
