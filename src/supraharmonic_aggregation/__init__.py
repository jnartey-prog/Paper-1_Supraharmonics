"""Public package interface for supraharmonic aggregation workflows."""

from ._version import __version__
from .api import analyze, default_config, generate_artifacts, run_pipeline
from .config import AnalysisConfig
from .models import (
    BenchmarkComparison,
    IntegrabilityReport,
    MonteCarloResult,
    RunBundle,
    TailMetrics,
)

__all__ = [
    "__version__",
    "AnalysisConfig",
    "TailMetrics",
    "IntegrabilityReport",
    "MonteCarloResult",
    "BenchmarkComparison",
    "RunBundle",
    "default_config",
    "analyze",
    "generate_artifacts",
    "run_pipeline",
]
