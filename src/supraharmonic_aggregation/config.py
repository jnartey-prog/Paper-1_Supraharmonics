"""Configuration schema and loading utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AnalysisConfig:
    """Top-level analysis configuration."""

    frequencies_khz: list[float] = field(default_factory=lambda: [2.0, 10.0, 30.0, 75.0, 150.0])
    density: float = 12.0
    region_radius_m: float = 500.0
    coherence: float = 0.0
    base_current_a: float = 1.0
    admittance_s: float = 0.01
    kernel_alpha: float = 0.8
    resonance_scale: float = 0.05
    threshold: float = 1.0
    monte_carlo_samples: int = 128
    seed: int = 7
    log_dir: str = "logs"
    output_dir: str = "manuscript/artifacts"

    def validate(self) -> None:
        """Validate this configuration and raise ValueError on invalid values."""
        if not self.frequencies_khz:
            raise ValueError("frequencies_khz must not be empty.")
        if any(freq <= 0 for freq in self.frequencies_khz):
            raise ValueError("frequencies_khz must be positive.")
        if self.density <= 0:
            raise ValueError("density must be positive.")
        if self.region_radius_m <= 0:
            raise ValueError("region_radius_m must be positive.")
        if not (0.0 <= self.coherence <= 1.0):
            raise ValueError("coherence must be between 0 and 1.")
        if self.monte_carlo_samples <= 0:
            raise ValueError("monte_carlo_samples must be positive.")
        if self.kernel_alpha <= 0:
            raise ValueError("kernel_alpha must be positive.")

    def to_dict(self) -> dict[str, object]:
        """Return a dict representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AnalysisConfig":
        """Build configuration from dictionary keys."""
        known = {field_name for field_name in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in payload.items() if k in known}
        config = cls(**kwargs)  # type: ignore[arg-type]
        config.validate()
        return config


def default_config() -> AnalysisConfig:
    """Return the package default analysis configuration."""
    config = AnalysisConfig()
    config.validate()
    return config


def load_config(path: str | None) -> AnalysisConfig:
    """Load configuration from JSON file path or defaults if path is None."""
    if not path:
        return default_config()
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration JSON must be an object.")
    return AnalysisConfig.from_dict(payload)


def save_config(config: AnalysisConfig, path: str) -> str:
    """Write a JSON config file to disk and return its path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return str(out)
