import json
from pathlib import Path

import pytest

from newt.config import BINNING, LOGGING, load_conf


def _snapshot_config_values():
    return {
        "BINNING": {
            name: getattr(BINNING, name) for name in BINNING.__dataclass_fields__
        },
        "LOGGING": {
            name: getattr(LOGGING, name) for name in LOGGING.__dataclass_fields__
        },
    }


def _restore_config_values(snapshot):
    for key, value in snapshot["BINNING"].items():
        object.__setattr__(BINNING, key, value)
    for key, value in snapshot["LOGGING"].items():
        object.__setattr__(LOGGING, key, value)


def test_load_conf_applies_json_overrides(tmp_path: Path):
    snapshot = _snapshot_config_values()
    try:
        config_path = tmp_path / "newt_conf.json"
        config_path.write_text(
            json.dumps(
                {
                    "binning": {"DEFAULT_BUCKETS": 13},
                    "logging": {"DEFAULT_LOG_LEVEL": "INFO"},
                }
            ),
            encoding="utf-8",
        )

        loaded = load_conf(str(config_path))

        assert loaded["BINNING"]["DEFAULT_BUCKETS"] == 13
        assert loaded["LOGGING"]["DEFAULT_LOG_LEVEL"] == "INFO"
        assert BINNING.DEFAULT_BUCKETS == 13
        assert LOGGING.DEFAULT_LOG_LEVEL == "INFO"
    finally:
        _restore_config_values(snapshot)


def test_load_conf_rejects_unknown_section(tmp_path: Path):
    config_path = tmp_path / "bad_section.json"
    config_path.write_text(
        json.dumps({"UNKNOWN_SECTION": {"DEFAULT_BUCKETS": 11}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown config section"):
        load_conf(str(config_path))


def test_load_conf_rejects_unknown_key(tmp_path: Path):
    config_path = tmp_path / "bad_key.json"
    config_path.write_text(
        json.dumps({"BINNING": {"NOT_EXIST_KEY": 11}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown config key"):
        load_conf(str(config_path))
