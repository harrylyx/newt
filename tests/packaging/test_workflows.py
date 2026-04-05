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


def test_build_wheels_workflow_uses_native_linux_arm_runner():
    """Linux aarch64 builds should run on a native ARM runner."""
    content = (WORKFLOWS / "build-wheels.yml").read_text()
    assert "ubuntu-24.04-arm" in content
    assert "docker/setup-qemu-action" not in content


def test_ci_workflow_does_not_build_wheels():
    """CI workflow should stay lean and not run cibuildwheel."""
    content = (WORKFLOWS / "ci.yml").read_text()
    assert "cibuildwheel" not in content


def test_build_wheels_workflow_has_smoke_test():
    """The build workflow must include a post-install smoke test."""
    content = (WORKFLOWS / "build-wheels.yml").read_text()
    assert "newt._newt_iv_rust" in content or "smoke" in content.lower()


def test_wheel_matrix_excludes_cp313():
    """Wheel matrix should align with supported Python 3.8.5-3.12."""
    build = (WORKFLOWS / "build-wheels.yml").read_text()
    release = (WORKFLOWS / "release.yml").read_text()
    assert "cp313" not in build
    assert "cp313" not in release


def test_wheel_matrix_skips_musllinux():
    """Current wheel matrix should skip musllinux jobs."""
    build = (WORKFLOWS / "build-wheels.yml").read_text()
    release = (WORKFLOWS / "release.yml").read_text()
    assert "*-musllinux_*" in build
    assert "*-musllinux_*" in release


def test_manylinux_images_use_policy_aliases():
    """Use stable manylinux policy aliases, not date-pinned image tags."""
    build = (WORKFLOWS / "build-wheels.yml").read_text()
    release = (WORKFLOWS / "release.yml").read_text()
    assert 'CIBW_MANYLINUX_X86_64_IMAGE: "manylinux_2_28"' in build
    assert 'CIBW_MANYLINUX_AARCH64_IMAGE: "manylinux_2_28"' in build
    assert 'CIBW_MANYLINUX_X86_64_IMAGE: "manylinux_2_28"' in release
    assert 'CIBW_MANYLINUX_AARCH64_IMAGE: "manylinux_2_28"' in release
    assert "quay.io/pypa/manylinux_2_28_" not in build
    assert "quay.io/pypa/manylinux_2_28_" not in release


def test_release_workflow_uses_native_linux_arm_runner():
    """Release builds should use the same native Linux runner split."""
    content = (WORKFLOWS / "release.yml").read_text()
    assert "ubuntu-24.04-arm" in content
    assert "docker/setup-qemu-action" not in content


def test_sdist_build_uses_project_root_context():
    """sdist should be built from pyproject context, not Cargo manifest path."""
    build = (WORKFLOWS / "build-wheels.yml").read_text()
    release = (WORKFLOWS / "release.yml").read_text()
    assert "maturin sdist --out dist" in build
    assert "maturin sdist --out dist" in release
