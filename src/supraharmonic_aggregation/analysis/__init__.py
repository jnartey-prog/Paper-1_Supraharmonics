"""Analytical and validation routines."""

from .analytical import compute_analytical_statistics
from .scaling import evaluate_scaling_laws
from .tail import compute_tail_metrics
from .validation import check_integrability_conditions

__all__ = [
    "compute_analytical_statistics",
    "evaluate_scaling_laws",
    "compute_tail_metrics",
    "check_integrability_conditions",
]
