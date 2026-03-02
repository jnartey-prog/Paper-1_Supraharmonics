"""CLI execution example."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    from supraharmonic_aggregation.cli import main as cli_main

    return cli_main(["--quickstart", "--output-dir", "manuscript/artifacts"])


if __name__ == "__main__":
    raise SystemExit(main())
