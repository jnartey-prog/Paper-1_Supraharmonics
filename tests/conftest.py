"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from supraharmonic_aggregation.config import AnalysisConfig  # noqa: E402


@pytest.fixture()
def baseline_config(tmp_path: Path) -> AnalysisConfig:
    """Return a deterministic baseline config for tests."""
    return AnalysisConfig(
        frequencies_khz=[2.0, 10.0, 30.0],
        density=10.0,
        region_radius_m=300.0,
        coherence=0.1,
        monte_carlo_samples=24,
        seed=11,
        log_dir=str(tmp_path / "logs"),
        output_dir=str(tmp_path / "artifacts"),
    )
