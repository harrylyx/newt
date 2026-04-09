import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.metric_vs_toad import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
