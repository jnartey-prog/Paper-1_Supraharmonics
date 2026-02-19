"""Scaling-law evaluation utilities."""

from __future__ import annotations

from ..config import AnalysisConfig
from ..models import StatisticsFrame
from .analytical import compute_analytical_statistics


def evaluate_scaling_laws(config: AnalysisConfig, densities: list[float], coherence: float) -> StatisticsFrame:
    """Evaluate RMS scaling trends over candidate densities."""
    rows: StatisticsFrame = []
    for density in densities:
        cfg = AnalysisConfig.from_dict({**config.to_dict(), "density": density, "coherence": coherence})
        stats = compute_analytical_statistics(cfg)
        rms_mean = sum(float(row["rms_abs_v"]) for row in stats) / len(stats)
        rows.append(
            {
                "density": density,
                "coherence": coherence,
                "mean_rms_abs_v": rms_mean,
                "sqrt_density": density ** 0.5,
            }
        )
    return rows
