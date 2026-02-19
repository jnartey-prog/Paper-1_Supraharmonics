"""Run manifest export utilities."""

from __future__ import annotations

import json
from pathlib import Path

from ..models import RunBundle


def export_run_manifest(run: RunBundle, path: str) -> str:
    """Write run manifest to disk and return target path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    return str(target)
