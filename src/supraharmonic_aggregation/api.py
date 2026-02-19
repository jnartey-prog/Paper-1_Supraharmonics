"""User-facing API for analysis and pipeline execution."""

from __future__ import annotations

import uuid
from pathlib import Path

from .analysis.analytical import compute_analytical_statistics
from .benchmark.compare import compare_with_feeder_benchmark
from .config import AnalysisConfig, default_config as _default_config, load_config
from .governance.manifest import export_run_manifest
from .logging_config import create_logger
from .models import RunBundle
from .simulation.monte_carlo import MonteCarloRunner
from .artifacts.generator import ArtifactGenerator
from .artifacts.manifest import build_artifact_manifest


def default_config() -> AnalysisConfig:
    """Return the package default configuration."""
    return _default_config()


def analyze(config: AnalysisConfig) -> RunBundle:
    """Run analytical + simulation + benchmark comparison workflow."""
    config.validate()
    run_id = str(uuid.uuid4())
    logger = create_logger(run_id=run_id, log_dir=config.log_dir)
    logger.log_event(
        stage="analysis",
        event="start",
        status="ok",
        duration_ms=0,
        component="api.analyze",
        message="Analysis workflow started.",
    )

    analytical = compute_analytical_statistics(config)
    mc = MonteCarloRunner(config=config, seed=config.seed).run(config.monte_carlo_samples)
    benchmark = compare_with_feeder_benchmark(analytical=analytical, simulated=mc.statistics_frame)

    run = RunBundle(
        run_id=run_id,
        config=config.to_dict(),
        analytical=analytical,
        monte_carlo=mc,
        benchmark=benchmark,
        log_path=str(logger.run_log_path),
        stats_log_path=str(logger.stats_log_path),
    )
    first_stats = mc.statistics_frame[0] if mc.statistics_frame else {}
    logger.log_statistics(
        {
            "frequency_khz": first_stats.get("frequency_khz", 0.0),
            "mean_abs_v": first_stats.get("mean_abs_v", 0.0),
            "var_v": first_stats.get("var_v", 0.0),
            "rms_abs_v": first_stats.get("rms_abs_v", 0.0),
            "p95_abs_v": first_stats.get("p95_abs_v", 0.0),
            "sample_size": first_stats.get("sample_size", 0),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }
    )
    logger.log_event(
        stage="analysis",
        event="complete",
        status="ok",
        duration_ms=0,
        component="api.analyze",
        message="Analysis workflow completed.",
    )
    return run


def generate_artifacts(run: RunBundle, output_dir: str | None = None) -> list[str]:
    """Generate manuscript tables and figures for the given run."""
    out = output_dir or str(run.config.get("output_dir", "manuscript/artifacts"))
    generator = ArtifactGenerator(output_dir=out)
    table_paths = generator.generate_tables(run)
    figure_paths = generator.generate_figures(run)
    paths = table_paths + figure_paths
    run.artifact_paths = paths
    return paths


def run_pipeline(config_path: str | None = None, output_dir: str = "manuscript/artifacts") -> RunBundle:
    """Run end-to-end workflow using config file or defaults."""
    config = load_config(config_path)
    config.output_dir = output_dir
    run = analyze(config)
    artifact_paths = generate_artifacts(run, output_dir=output_dir)
    manifest_data = build_artifact_manifest(artifact_paths)
    manifest_data["run_id"] = run.run_id
    manifest_path = Path(output_dir) / "run_manifest.json"
    run.run_manifest_path = export_run_manifest(run, str(manifest_path))
    return run
