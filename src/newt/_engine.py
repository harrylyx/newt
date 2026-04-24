"""Shared helpers for resolving Python vs native execution engines."""

from __future__ import annotations

from types import ModuleType
from typing import Callable, Optional, Sequence

from newt._native import load_native_module

VALID_ENGINES = frozenset(["auto", "rust", "python"])
NativeLoader = Callable[[], Optional[ModuleType]]


def validate_engine(engine: str) -> str:
    """Validate an engine option and return it unchanged."""
    if engine not in VALID_ENGINES:
        raise ValueError(
            f"engine must be one of {sorted(VALID_ENGINES)}, got: {engine}"
        )
    return engine


def resolve_engine(
    engine: str,
    required_functions: Sequence[str] = (),
    component: str = "Rust engine",
    loader: NativeLoader = load_native_module,
) -> str:
    """Resolve a user-facing engine option to ``"rust"`` or ``"python"``.

    Args:
        engine: ``"auto"``, ``"rust"``, or ``"python"``.
        required_functions: Native functions required by this call path.
        component: Human-readable component name for error messages.
        loader: Native module loader, injectable for tests.

    Returns:
        Concrete runtime engine: ``"rust"`` or ``"python"``.
    """
    validate_engine(engine)
    if engine == "python":
        return "python"

    module = loader()
    if module is None:
        if engine == "rust":
            raise ImportError(
                "Rust engine requested but native extension is unavailable. "
                "Use engine='auto' or engine='python', or install a wheel with "
                "the native extension."
            )
        return "python"

    if required_functions:
        try:
            ensure_native_functions(module, required_functions, component=component)
        except RuntimeError:
            if engine == "rust":
                raise
            return "python"

    return "rust"


def ensure_native_functions(
    module: object,
    required_functions: Sequence[str],
    component: str = "Rust engine",
) -> None:
    """Raise when a native module misses functions required by a call path."""
    missing = [
        function_name
        for function_name in required_functions
        if not callable(getattr(module, function_name, None))
    ]
    if missing:
        raise RuntimeError(
            f"{component} missing native functions: {', '.join(sorted(missing))}"
        )


__all__ = [
    "VALID_ENGINES",
    "ensure_native_functions",
    "resolve_engine",
    "validate_engine",
]
