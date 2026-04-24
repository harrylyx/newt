from types import SimpleNamespace

import pytest

from newt._engine import ensure_native_functions, resolve_engine


def test_resolve_engine_auto_falls_back_to_python_when_native_missing():
    resolved = resolve_engine("auto", loader=lambda: None)

    assert resolved == "python"


def test_resolve_engine_rust_requires_native_module():
    with pytest.raises(ImportError, match="Rust engine requested"):
        resolve_engine("rust", loader=lambda: None)


def test_resolve_engine_rejects_invalid_engine():
    with pytest.raises(ValueError, match="engine must be one of"):
        resolve_engine("invalid", loader=lambda: None)


def test_ensure_native_functions_reports_missing_capabilities():
    module = SimpleNamespace(existing=lambda: None)

    with pytest.raises(RuntimeError, match="missing native functions: missing"):
        ensure_native_functions(module, ["existing", "missing"])
