# Supraharmonic Aggregation

Research-oriented Python package for transformer-level supraharmonic aggregation analysis. The repository contains the package source in `src/`, the automated test suite in `tests/`, and generated manuscript artifacts in `manuscript/artifacts/`.

## What It Does

The package provides:

- analytical statistics for supraharmonic aggregation studies
- Monte Carlo simulation for per-frequency response estimates
- benchmark comparison utilities
- artifact generation for manuscript tables and figures
- a CLI pipeline for end-to-end runs

## Repository Layout

- `src/supraharmonic_aggregation/`: package implementation
- `tests/`: unit, integration, pipeline, and acceptance tests
- `manuscript/artifacts/`: generated figures, tables, and manifests
- `pyproject.toml`: packaging and tool configuration

## Installation

```bash
pip install -e .
```

For development tooling:

```bash
pip install -e .[dev]
```

If your installer does not support dependency groups from `pyproject.toml`, install the development tools manually.

## Command-Line Usage

Write a default configuration file:

```bash
supraharmonic-pipeline --write-default-config config.json
```

Run the pipeline with defaults and write outputs to `manuscript/artifacts`:

```bash
supraharmonic-pipeline --quickstart
```

Run the pipeline with an explicit configuration and output directory:

```bash
supraharmonic-pipeline --config config.json --output-dir manuscript/artifacts
```

The CLI prints a JSON summary including the run id, artifact count, manifest path, and log paths.

## Python API

```python
from supraharmonic_aggregation.api import default_config, run_pipeline

config = default_config()
run = run_pipeline(output_dir="manuscript/artifacts")
print(run.run_id)
print(run.run_manifest_path)
```

For lower-level use, the public API also exposes `analyze()` and `generate_artifacts()`.

## Configuration

The default analysis configuration includes controls for:

- frequency points in kHz
- source density and region size
- phase coherence
- current and admittance parameters
- kernel and resonance settings
- threshold and RMS screening parameters
- Monte Carlo sample count and seed
- logging and artifact output directories

Configurations are stored as JSON and validated before execution.

## Testing

Run the test suite with:

```bash
pytest
```

The configured test groups are `unit`, `integration`, `pipeline`, and `acceptance`.

## Notes

- `project.readme` in `pyproject.toml` points to this file.
- The current repository snapshot intentionally tracks only `README.md`, `pyproject.toml`, `src/`, `tests/`, and `manuscript/`.
