"""Synthetic dataset generation for framework validation."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from ..analysis.analytical import compute_analytical_statistics
from ..analysis.tail import adaptive_threshold, compute_tail_metrics
from ..benchmark.compare import compare_with_feeder_benchmark
from ..config import AnalysisConfig
from ..core.aggregator import SupraharmonicAggregator
from ..core.kernel import ExponentialKernel
from ..core.marks import amplitudes, generate_source_population
from ..models import StatisticsFrame


SyntheticObservation = dict[str, float | int | str]


@dataclass(slots=True)
class SyntheticDataset:
    """Synthetic observations and derived validation frames."""

    observations: list[SyntheticObservation]
    per_frequency_samples: dict[str, list[float]]
    statistics_frame: StatisticsFrame
    analytical_frame: StatisticsFrame
    validation_frame: StatisticsFrame


def _fieldnames(rows: list[dict[str, float | int | str]]) -> list[str]:
    if not rows:
        return []
    names = list(rows[0].keys())
    seen = set(names)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def _build_statistics(
    frequencies_khz: list[float],
    per_frequency_samples: dict[str, list[float]],
    threshold: float | None,
    threshold_rms_multiplier: float,
) -> StatisticsFrame:
    rows: StatisticsFrame = []
    for frequency in frequencies_khz:
        key = str(frequency)
        values = per_frequency_samples[key]
        mean_abs_v = sum(values) / len(values) if values else 0.0
        var_v = sum((value - mean_abs_v) ** 2 for value in values) / len(values) if values else 0.0
        rms_abs_v = (mean_abs_v**2 + var_v) ** 0.5
        tail_threshold = adaptive_threshold(
            floor_threshold=threshold or 0.0,
            rms_abs_v=rms_abs_v,
            multiplier=threshold_rms_multiplier,
        )
        tail = compute_tail_metrics(values, threshold=tail_threshold)
        rows.append(
            {
                "frequency_khz": frequency,
                "mean_abs_v": mean_abs_v,
                "var_v": var_v,
                "rms_abs_v": rms_abs_v,
                "p90_abs_v": tail.percentiles.get(90, 0.0),
                "p95_abs_v": tail.percentiles.get(95, 0.0),
                "p99_abs_v": tail.percentiles.get(99, 0.0),
                "exceedance_probability": tail.exceedance_probability or 0.0,
                "exceedance_threshold_v": tail_threshold,
                "sample_size": tail.sample_size,
            }
        )
    return rows


class SyntheticDataGenerator:
    """Generate synthetic observations aligned to package physics/statistics."""

    def __init__(self, config: AnalysisConfig, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed

    def generate(
        self,
        n_samples: int | None = None,
        include_complex: bool = True,
        frequencies_khz: list[float] | None = None,
        include_measurement_noise: bool = True,
    ) -> SyntheticDataset:
        """Generate a reproducible synthetic dataset and validation tables."""
        self.config.validate()
        sample_count = self.config.monte_carlo_samples if n_samples is None else n_samples
        if sample_count <= 0:
            raise ValueError("n_samples must be positive.")

        rng = random.Random(self.seed)
        kernel = ExponentialKernel(
            alpha=self.config.kernel_alpha, resonance_scale=self.config.resonance_scale
        )
        aggregator = SupraharmonicAggregator(kernel)
        frequencies = list(frequencies_khz or self.config.frequencies_khz)
        per_frequency_samples: dict[str, list[float]] = {str(freq): [] for freq in frequencies}
        observations: list[SyntheticObservation] = []

        for sample_id in range(sample_count):
            population = generate_source_population(
                density=self.config.density,
                region_radius_m=self.config.region_radius_m,
                coherence=self.config.coherence,
                base_current_a=self.config.base_current_a,
                admittance_s=self.config.admittance_s,
                rng=rng,
            )
            source_count = len(population)
            source_amplitudes = amplitudes(population)
            mean_source_amplitude = (
                sum(source_amplitudes) / len(source_amplitudes) if source_amplitudes else 0.0
            )

            for frequency in frequencies:
                complex_voltage = aggregator.aggregate_complex_voltage(frequency, population)
                latent_abs_v = abs(complex_voltage)
                observed_abs_v = latent_abs_v
                if include_measurement_noise and self.config.measurement_noise_cv > 0:
                    observed_abs_v = latent_abs_v * (1.0 + self.config.measurement_bias)
                    observed_abs_v += rng.gauss(
                        0.0, self.config.measurement_noise_cv * max(latent_abs_v, 1e-6)
                    )
                    observed_abs_v = max(observed_abs_v, 0.0)
                per_frequency_samples[str(frequency)].append(observed_abs_v)
                row: SyntheticObservation = {
                    "sample_id": sample_id,
                    "frequency_khz": frequency,
                    "source_count": source_count,
                    "mean_source_amplitude_a": mean_source_amplitude,
                    "abs_v": observed_abs_v,
                    "latent_abs_v": latent_abs_v,
                }
                if include_complex:
                    row["real_v"] = complex_voltage.real
                    row["imag_v"] = complex_voltage.imag
                observations.append(row)

        analysis_config = self.config
        if frequencies != self.config.frequencies_khz:
            analysis_config = AnalysisConfig.from_dict(
                {
                    **self.config.to_dict(),
                    "frequencies_khz": frequencies,
                }
            )
        statistics_frame = _build_statistics(
            frequencies,
            per_frequency_samples,
            threshold=self.config.threshold,
            threshold_rms_multiplier=self.config.threshold_rms_multiplier,
        )
        analytical_frame = compute_analytical_statistics(analysis_config)
        validation_frame = compare_with_feeder_benchmark(
            analytical=analytical_frame,
            simulated=statistics_frame,
        ).rows

        return SyntheticDataset(
            observations=observations,
            per_frequency_samples=per_frequency_samples,
            statistics_frame=statistics_frame,
            analytical_frame=analytical_frame,
            validation_frame=validation_frame,
        )

    def save_latest(
        self,
        dataset: SyntheticDataset,
        output_dir: str = "synthetic_data",
        prefix: str = "synthetic_latest",
    ) -> dict[str, str]:
        """Persist dataset CSVs, removing older files with the same prefix."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        managed_files = {
            "observations": out_dir / f"{prefix}_observations.csv",
            "per_frequency_samples": out_dir / f"{prefix}_per_frequency_samples.csv",
            "statistics": out_dir / f"{prefix}_statistics.csv",
            "analytical": out_dir / f"{prefix}_analytical.csv",
            "validation": out_dir / f"{prefix}_validation.csv",
        }

        for stale in out_dir.glob("*"):
            if stale.is_file() and stale.suffix.lower() in {".csv", ".md"}:
                stale.unlink(missing_ok=True)

        per_frequency_rows: list[dict[str, float | int | str]] = []
        for frequency, values in dataset.per_frequency_samples.items():
            for sample_id, value in enumerate(values):
                per_frequency_rows.append(
                    {
                        "frequency_khz": frequency,
                        "sample_id": sample_id,
                        "abs_v": value,
                    }
                )

        _write_csv(managed_files["observations"], dataset.observations)
        _write_csv(managed_files["per_frequency_samples"], per_frequency_rows)
        _write_csv(managed_files["statistics"], dataset.statistics_frame)
        _write_csv(managed_files["analytical"], dataset.analytical_frame)
        _write_csv(managed_files["validation"], dataset.validation_frame)

        return {name: str(path) for name, path in managed_files.items()}

    def generate_and_save_latest(
        self,
        n_samples: int | None = None,
        include_complex: bool = True,
        output_dir: str = "synthetic_data",
        prefix: str = "synthetic_latest",
        review_ready: bool = True,
    ) -> tuple[SyntheticDataset, dict[str, str]]:
        """Generate dataset and save only the latest synthetic CSV outputs."""
        sample_count = n_samples
        frequencies: list[float] | None = None
        if review_ready:
            sample_count = (
                max(self.config.review_ready_min_samples, self.config.monte_carlo_samples)
                if n_samples is None
                else n_samples
            )
            start = min(self.config.frequencies_khz)
            stop = max(self.config.frequencies_khz)
            step = self.config.review_ready_frequency_step_khz
            n_steps = int(round((stop - start) / step))
            frequencies = [round(start + idx * step, 6) for idx in range(n_steps + 1)]
            if frequencies[-1] < stop:
                frequencies.append(stop)
        dataset = self.generate(
            n_samples=sample_count,
            include_complex=include_complex,
            frequencies_khz=frequencies,
            include_measurement_noise=True,
        )
        paths = self.save_latest(dataset=dataset, output_dir=output_dir, prefix=prefix)
        return dataset, paths
