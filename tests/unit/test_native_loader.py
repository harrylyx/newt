"""Tests for the shared native extension loader."""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from newt import _native


def test_load_native_extension_imports_new_module_name_only():
    """The shared loader should import the new extension module name."""
    expected_module = SimpleNamespace(calculate_batch_iv=True)

    def fake_import(name, *args, **kwargs):
        if name != "newt._newt_native":
            raise AssertionError(f"unexpected import attempt: {name}")
        return expected_module

    with patch.object(importlib, "import_module", side_effect=fake_import):
        loaded_module = _native.load_native_module()

    assert loaded_module is expected_module


def test_require_native_extension_error_mentions_new_build_path():
    """The shared loader should raise a clear error for the new module path."""

    def fake_import(name, *args, **kwargs):
        raise ImportError(f"mocked missing: {name}")

    with patch.object(importlib, "import_module", side_effect=fake_import):
        with pytest.raises(ImportError) as exc_info:
            _native.require_native_module()

    error_text = str(exc_info.value)
    assert "newt._newt_native" in error_text
    assert "_newt_native" in error_text
    assert "rust/newt_native/Cargo.toml" in error_text
    assert "_newt_iv_rust" not in error_text
