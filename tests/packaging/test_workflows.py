"""Tests for GitHub Actions workflow configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_build_wheels_workflow_exists():
    """The build-wheels workflow file must exist."""
    assert (WORKFLOWS / "build-wheels.yml").exists()


def test_release_workflow_references_wheel_artifacts():
    """The release workflow must reference built wheel artifacts."""
    content = (WORKFLOWS / "release.yml").read_text()
    # Must use the build-wheels job or download wheel artifacts
    assert "build-wheels" in content or "download-artifact" in content


def test_build_wheels_workflow_references_cibuildwheel():
    """The build workflow must use cibuildwheel."""
    content = (WORKFLOWS / "build-wheels.yml").read_text()
    assert "cibuildwheel" in content


def test_build_wheels_workflow_covers_all_platforms():
    """The build workflow must target macOS, Windows, and Linux."""
    content = (WORKFLOWS / "build-wheels.yml").read_text()
    assert "macos" in content.lower()
    assert "windows" in content.lower()
    assert "ubuntu" in content.lower() or "linux" in content.lower()


def test_ci_workflow_does_not_build_wheels():
    """CI workflow should stay lean and not run cibuildwheel."""
    content = (WORKFLOWS / "ci.yml").read_text()
    assert "cibuildwheel" not in content


def test_build_wheels_workflow_has_smoke_test():
    """The build workflow must include a post-install smoke test."""
    content = (WORKFLOWS / "build-wheels.yml").read_text()
    assert "newt._newt_iv_rust" in content or "smoke" in content.lower()
