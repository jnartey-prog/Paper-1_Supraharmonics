from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.acceptance
def test_governance_files_exist() -> None:
    required = [
        Path("DECISIONS.md"),
        Path("tasks.yaml"),
        Path("tasks.seed.yaml"),
        Path("proposal.normalized.json"),
    ]
    for path in required:
        assert path.exists(), f"Missing governance artifact: {path}"
