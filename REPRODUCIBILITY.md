# Reproducibility Guide

This document defines the exact workflow to reproduce the paper artifacts from this repository.

## 1) Reference the archival snapshot

For manuscript citation, always reference:
- the Git tag used for submission (for example `v1.0.0-paper`), and
- the exact commit SHA behind that tag.

Do not cite a moving branch such as `main`.

## 2) Environment requirements

- OS: Linux, macOS, or Windows
- Python: 3.10.x (paper baseline)
- Tooling: `uv` (dependency manager and runner)

Project metadata allows Python >=3.10, but paper reproduction should use Python 3.10.x for maximal consistency.

## 3) Clean setup (fresh clone)

```bash
git clone <repo-url>
cd <repo-folder>
uv python install 3.10
uv sync --frozen --group dev --python 3.10
```

`--frozen` ensures the environment is resolved strictly from `uv.lock`.

## 4) Reproduce paper artifacts

```bash
uv run supraharmonic-pipeline --quickstart --output-dir manuscript/artifacts
```

Expected generated files include:
- `manuscript/artifacts/table_1_scenario_matrix.csv`
- `manuscript/artifacts/table_6_analytical_vs_feeder_metrics.csv`
- `manuscript/artifacts/figure_1_transfer_impedance_vs_distance.png`
- `manuscript/artifacts/figure_8_allowable_density_screening_chart.png`

## 5) Verify integrity

Run the test suite used in CI:

```bash
uv run pytest tests/ --cov=supraharmonic_aggregation --cov-report=xml
```

Optional code-quality checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/supraharmonic_aggregation
```

## 6) Determinism notes

- Simulations are seeded via `AnalysisConfig.seed` (default `7`).
- Dependency versions are pinned in `uv.lock`.
- CI validates the locked environment and build on each push and pull request.

## 7) Reviewer-facing citation block

Include this in the manuscript supplementary reproducibility statement:

1. Repository URL
2. Paper release tag
3. Commit SHA
4. DOI (Zenodo archive of the tagged release)
5. Date accessed

