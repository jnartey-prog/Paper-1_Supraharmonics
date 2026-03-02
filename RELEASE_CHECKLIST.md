# Paper Release Checklist

Use this checklist before creating the manuscript-cited release.

## Repository state

- [ ] Working tree is clean (`git status` shows no tracked modifications).
- [ ] No untracked generated artifacts that should be excluded.
- [ ] `uv.lock` is committed and up to date.

## Validation

- [ ] `uv sync --frozen --group dev --python 3.10`
- [ ] `uv run ruff check .`
- [ ] `uv run ruff format --check .`
- [ ] `uv run mypy src/supraharmonic_aggregation`
- [ ] `uv run pytest tests/ --cov=supraharmonic_aggregation --cov-report=xml`
- [ ] `uv build`

## Artifact reproduction

- [ ] `uv run supraharmonic-pipeline --quickstart --output-dir manuscript/artifacts`
- [ ] Required tables and figures exist under `manuscript/artifacts`.
- [ ] Repro instructions in `REPRODUCIBILITY.md` match the current commands.

## Archival and citation

- [ ] Create annotated tag (for example `v1.0.0-paper`).
- [ ] Push tag to GitHub.
- [ ] Create GitHub release from the tag.
- [ ] Archive the release via Zenodo and obtain DOI.
- [ ] Add DOI and release tag to `README.md`.
- [ ] Record the exact commit SHA in manuscript supplementary materials.

