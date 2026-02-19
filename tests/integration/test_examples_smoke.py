from __future__ import annotations

import runpy
from pathlib import Path

import pytest


@pytest.mark.integration
def test_quickstart_example_runs() -> None:
    target = Path("examples/quickstart.py")
    assert target.exists()
    runpy.run_path(str(target), run_name="__main__")
