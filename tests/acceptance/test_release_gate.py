from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.acceptance
def test_release_gate_core_artifacts_present() -> None:
    must_exist = [
        Path("agents.yaml"),
        Path("specs/A1.md"),
        Path("specs/A2.md"),
        Path("specs/A3.md"),
        Path("specs/A4.md"),
        Path("specs/A5.md"),
        Path("pyproject.toml"),
    ]
    for path in must_exist:
        assert path.exists(), f"Missing release gate artifact: {path}"
