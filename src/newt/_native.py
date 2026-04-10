"""Shared loader for the compiled native extension."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

NATIVE_MODULE_NAME = "newt._newt_native"
NATIVE_BUILD_MANIFEST = "rust/newt_native/Cargo.toml"


def load_native_module() -> Optional[ModuleType]:
    """Return the compiled native module when available."""
    try:
        return importlib.import_module(NATIVE_MODULE_NAME)
    except ImportError:
        return None


def require_native_module() -> ModuleType:
    """Return the compiled native module or raise a clear ImportError."""
    module = load_native_module()
    if module is not None:
        return module

    raise ImportError(
        "The compiled native extension (newt._newt_native or _newt_native) "
        "is not available. Install Newt from an official wheel that includes "
        "the prebuilt native extension, or build from source with "
        f"'maturin develop --manifest-path {NATIVE_BUILD_MANIFEST}'."
    )


__all__ = [
    "NATIVE_BUILD_MANIFEST",
    "NATIVE_MODULE_NAME",
    "load_native_module",
    "require_native_module",
]
