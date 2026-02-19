from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_ci_workflow_exists() -> None:
    workflow = Path(".github/workflows/ci.yml")
    assert workflow.exists()
