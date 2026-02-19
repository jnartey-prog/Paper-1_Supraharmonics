"""Quickstart example for supraharmonic aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import supraharmonic_aggregation as sha


def main() -> None:
    config = sha.default_config()
    run = sha.analyze(config)
    sha.generate_artifacts(run, output_dir=config.output_dir)
    print(f"run_id={run.run_id}")


if __name__ == "__main__":
    main()
