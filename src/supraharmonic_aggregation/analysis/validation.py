"""Model validity checks."""

from __future__ import annotations

from ..config import AnalysisConfig
from ..models import IntegrabilityReport


def check_integrability_conditions(config: AnalysisConfig) -> IntegrabilityReport:
    """Validate boundedness conditions from configured attenuation parameters."""
    config.validate()
    finite_ok = config.region_radius_m > 0 and config.density > 0
    asymptotic_ok = config.kernel_alpha > 0 and config.admittance_s >= 0
    details = (
        "Finite-domain conditions satisfied."
        if finite_ok
        else "Finite-domain conditions violated."
    )
    if asymptotic_ok:
        details += " Asymptotic attenuation condition satisfied."
    else:
        details += " Asymptotic attenuation condition violated."
    return IntegrabilityReport(
        finite_domain_ok=finite_ok,
        asymptotic_domain_ok=asymptotic_ok,
        details=details,
    )
