from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_pyproject_contains_project_name() -> None:
    content = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "supraharmonic-aggregation"' in content
