"""Tests for Rust extension packaging and loading behavior."""

import importlib
from importlib import import_module
from unittest.mock import patch

import pytest


def test_batch_iv_uses_package_internal_rust_module():
    """The Rust extension should be importable as newt._newt_native."""
    module = import_module("newt._newt_native")
    assert hasattr(module, "calculate_batch_iv")
    assert hasattr(module, "calculate_binary_metrics_batch_numpy")
    assert hasattr(module, "calculate_feature_psi_pairs_numpy")


def test_load_rust_extension_does_not_trigger_local_build_when_missing():
    """When the compiled extension is absent, _load_rust_extension must raise
    ImportError instead of attempting a hidden local build via maturin."""
    from newt.features.analysis import batch_iv as biv

    original_import = importlib.import_module

    def selective_import(name, *args, **kwargs):
        if name in ("newt_native", "newt._newt_native", "_newt_native"):
            raise ImportError(f"mocked missing: {name}")
        return original_import(name, *args, **kwargs)

    with patch.object(importlib, "import_module", side_effect=selective_import):
        with pytest.raises(ImportError, match="_newt_native"):
            biv._load_rust_extension()

    # Verify no subprocess/build functions exist in the module
    assert not hasattr(biv, "_build_rust_extension")
    assert not hasattr(biv, "_resolve_maturin_executable")
