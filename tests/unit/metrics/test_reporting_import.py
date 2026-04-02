import subprocess
import sys


def test_reporting_module_imports_in_fresh_python_process():
    result = subprocess.run(
        [sys.executable, "-c", "import newt.metrics.reporting"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
