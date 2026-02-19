"""Artifact manifest helpers."""

from __future__ import annotations

from pathlib import Path


def build_artifact_manifest(paths: list[str]) -> dict[str, list[str]]:
    """Build a lightweight manifest split by extension type."""
    tables: list[str] = []
    figures: list[str] = []
    for path in paths:
        suffix = Path(path).suffix.lower()
        if suffix in {".csv", ".tex"}:
            tables.append(path)
        elif suffix in {".png", ".pdf", ".svg"}:
            figures.append(path)
    return {"tables": tables, "figures": figures}
