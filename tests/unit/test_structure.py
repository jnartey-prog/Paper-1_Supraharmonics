from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_project_structure_files_exist() -> None:
    required = [
        Path("pyproject.toml"),
        Path("src/supraharmonic_aggregation/__init__.py"),
        Path("src/supraharmonic_aggregation/cli.py"),
        Path("specs/A1.md"),
    ]
    for path in required:
        assert path.exists(), f"Missing required file: {path}"
