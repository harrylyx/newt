"""Wheel installation and smoke tests.

These tests build and install a wheel into a temporary virtual environment
to verify the end-user experience: the Rust extension must be available
immediately after ``pip install`` without local compilation.

Uses ``uv`` to create venvs and install wheels for reliability.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = ROOT / "dist"


def _find_wheel() -> Path:
    """Return the most recently built wheel under ``dist/``."""
    wheels = sorted(DIST_DIR.glob("newt-*.whl"), key=os.path.getmtime, reverse=True)
    if not wheels:
        pytest.skip("No wheel found under dist/ – run 'uv build' first.")
    return wheels[0]


def _uv() -> str:
    """Resolve the uv executable."""
    uv = shutil.which("uv")
    if uv:
        return uv
    pytest.skip("uv not found on PATH")
    return ""  # unreachable


def _create_venv(tmp_path: Path) -> Path:
    """Create a minimal virtual environment with uv and return its Python path."""
    venv_path = tmp_path / "test_venv"
    uv = _uv()
    subprocess.check_call(
        [
            uv,
            "venv",
            str(venv_path),
            "--python",
            sys.executable,
        ],
        timeout=60,
    )
    if sys.platform == "win32":
        python = venv_path / "Scripts" / "python.exe"
    else:
        python = venv_path / "bin" / "python"
    assert python.exists(), f"venv Python not found at {python}"
    return python


@pytest.mark.slow
def test_installed_wheel_can_import_rust_extension(tmp_path: Path):
    """Install the wheel into a clean venv and import the Rust extension."""
    wheel = _find_wheel()
    python = _create_venv(tmp_path)
    uv = _uv()

    # Install wheel using uv pip
    subprocess.check_call(
        [uv, "pip", "install", "--python", str(python), str(wheel)],
        timeout=120,
    )

    # Verify the Rust extension is importable
    result = subprocess.run(
        [
            str(python),
            "-c",
            textwrap.dedent(
                """\
                import importlib
                import sys

                try:
                    import newt._newt_native as ext
                except ImportError as e:
                    print(f"FAILED: Native import error: {e}", file=sys.stderr)
                    sys.exit(1)

                expected_attrs = [
                    "calculate_batch_iv",
                    "calculate_categorical_iv",
                    "calculate_binary_metrics_batch_numpy",
                    "calculate_feature_psi_pairs_numpy",
                ]

                for attr in expected_attrs:
                    if not hasattr(ext, attr):
                        msg = f"FAILED: Attribute '{attr}' missing"
                        print(msg, file=sys.stderr)
                        sys.exit(2)

                try:
                    importlib.import_module("newt._newt_iv_rust")
                    msg = "FAILED: Old path 'newt._newt_iv_rust' still importable"
                    print(msg, file=sys.stderr)
                    sys.exit(3)
                except ImportError:
                    pass

                print("RUST_IMPORT_OK")
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        "Import failed.\n"
        "stdout:\n" + result.stdout + "\n"
        "stderr:\n" + result.stderr
    )
    assert "RUST_IMPORT_OK" in result.stdout


@pytest.mark.slow
def test_installed_wheel_can_run_rust_batch_iv(tmp_path: Path):
    """Install the wheel and execute a Rust-backed IV calculation."""
    wheel = _find_wheel()
    python = _create_venv(tmp_path)
    uv = _uv()

    subprocess.check_call(
        [uv, "pip", "install", "--python", str(python), str(wheel)],
        timeout=120,
    )

    snippet = textwrap.dedent(
        """\
        import pandas as pd
        from newt.features.analysis.batch_iv import calculate_batch_iv

        data = pd.DataFrame({
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
            "x2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        })
        y = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])

        result = calculate_batch_iv(data, y, engine="rust", bins=4)
        assert len(result) == 2, f"Expected 2 rows, got {len(result)}"
        assert all(result["iv"] >= 0), "IV values must be non-negative"
        print("RUST_BATCH_IV_OK")
        """
    )

    result = subprocess.run(
        [str(python), "-c", snippet],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        "Rust IV failed.\n"
        "stdout:\n" + result.stdout + "\n"
        "stderr:\n" + result.stderr
    )
    assert "RUST_BATCH_IV_OK" in result.stdout
