"""
Configuration constants for the newt package.

Provides centralized default values for binning, filtering, and modeling.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, Mapping


@dataclass(frozen=True)
class BinningConfig:
    """分箱相关默认配置"""

    DEFAULT_N_BINS: Final[int] = 5
    DEFAULT_BUCKETS: Final[int] = 10
    DEFAULT_EPSILON: Final[float] = 1e-8
    MIN_SAMPLES_LEAF: Final[int] = 10


@dataclass(frozen=True)
class FilteringConfig:
    """过滤相关默认配置"""

    DEFAULT_IV_THRESHOLD: Final[float] = 0.02
    DEFAULT_MISSING_THRESHOLD: Final[float] = 0.9
    DEFAULT_CORR_THRESHOLD: Final[float] = 0.8
    DEFAULT_PSI_THRESHOLD: Final[float] = 0.25
    DEFAULT_VIF_THRESHOLD: Final[float] = 10.0


@dataclass(frozen=True)
class ModelingConfig:
    """建模相关默认配置"""

    DEFAULT_P_ENTER: Final[float] = 0.05
    DEFAULT_P_REMOVE: Final[float] = 0.10
    DEFAULT_CLASSIFICATION_THRESHOLD: Final[float] = 0.5


@dataclass(frozen=True)
class ScorecardConfig:
    """评分卡相关默认配置"""

    DEFAULT_PDO: Final[int] = 20
    DEFAULT_BASE_SCORE: Final[int] = 600
    DEFAULT_BASE_ODDS: Final[float] = 1.0


@dataclass(frozen=True)
class LoggingConfig:
    """日志相关默认配置"""

    DEFAULT_LOG_LEVEL: Final[str] = "DEBUG"
    DEFAULT_LOG_FORMAT: Final[
        str
    ] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


# Singleton instances for easy access
BINNING = BinningConfig()
FILTERING = FilteringConfig()
MODELING = ModelingConfig()
SCORECARD = ScorecardConfig()
LOGGING = LoggingConfig()


_CONFIG_SECTION_NAMES = {
    "binning": "BINNING",
    "filtering": "FILTERING",
    "modeling": "MODELING",
    "scorecard": "SCORECARD",
    "logging": "LOGGING",
}


def load_conf(conf_path: str) -> Dict[str, Dict[str, Any]]:
    """Load and apply external configuration from a file.

    Supported file types: ``.json``, ``.toml``, ``.yaml``, ``.yml``.

    The configuration file should use section keys matching Newt config groups,
    for example ``BINNING`` or ``binning``.

    Example payload::

        {
            "binning": {"DEFAULT_BUCKETS": 15},
            "logging": {"DEFAULT_LOG_LEVEL": "INFO"}
        }

    Args:
        conf_path: Path to config file.

    Returns:
        Canonicalized overrides applied to runtime config objects.
    """
    payload = _read_config_payload(conf_path)
    overrides = _normalize_config_overrides(payload)
    _apply_config_overrides(overrides)
    return overrides


def _read_config_payload(conf_path: str) -> Dict[str, Any]:
    path = Path(conf_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        payload = _load_toml(path)
    elif suffix in {".yaml", ".yml"}:
        payload = _load_yaml(path)
    else:
        raise ValueError(
            "Unsupported config file extension. Supported: .json, .toml, .yaml, .yml"
        )

    if not isinstance(payload, dict):
        raise ValueError("Config file root must be a mapping object.")
    return payload


def _normalize_config_overrides(
    payload: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    targets = _config_targets()
    normalized: Dict[str, Dict[str, Any]] = {}
    for raw_section, section_values in payload.items():
        section_name = _normalize_section_name(raw_section)
        if section_name is None:
            raise ValueError(f"Unknown config section: {raw_section}")
        if not isinstance(section_values, dict):
            raise ValueError(
                f"Config section '{raw_section}' must be a mapping of key-value pairs."
            )
        target = targets[section_name]
        allowed_keys = set(target.__dataclass_fields__.keys())
        section_output = normalized.setdefault(section_name, {})
        for raw_key, value in section_values.items():
            key = str(raw_key).strip()
            if key not in allowed_keys:
                raise ValueError(f"Unknown config key: {section_name}.{key}")
            section_output[key] = value
    return normalized


def _apply_config_overrides(overrides: Mapping[str, Mapping[str, Any]]) -> None:
    targets = _config_targets()
    for section_name, section_values in overrides.items():
        target = targets[section_name]
        for key, value in section_values.items():
            object.__setattr__(target, key, value)


def _normalize_section_name(section_name: Any) -> str:
    text = str(section_name).strip()
    upper = text.upper()
    if upper in _config_targets():
        return upper
    return _CONFIG_SECTION_NAMES.get(text.lower())


def _config_targets() -> Dict[str, Any]:
    return {
        "BINNING": BINNING,
        "FILTERING": FILTERING,
        "MODELING": MODELING,
        "SCORECARD": SCORECARD,
        "LOGGING": LOGGING,
    }


def _load_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise ImportError(
                "TOML config requires tomllib (Python 3.11+) or tomli."
            ) from exc
    with path.open("rb") as file:
        return tomllib.load(file)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ImportError("YAML config requires pyyaml package.") from exc
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    return payload if payload is not None else {}
