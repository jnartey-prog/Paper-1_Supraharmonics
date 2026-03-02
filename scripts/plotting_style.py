"""Shared Matplotlib styling and figure saving for manuscript-ready outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

MANUSCRIPT_DPI = 600
EXPORT_FORMATS = ("png", "pdf", "svg")


def apply_manuscript_style() -> None:
    """Apply conservative defaults that render well in manuscripts."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.8,
            # IEEE/Elsevier-friendly defaults: embed TrueType fonts (avoid Type 3).
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Prefer Times-like serif; fall back safely.
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = MANUSCRIPT_DPI, **kwargs: Any) -> None:
    """Save a figure with settings suitable for submission-quality raster output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not getattr(fig, "_skip_tight_layout", False):
        fig.tight_layout()
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        **kwargs,
    )
    plt.close(fig)


def save_figure_bundle(fig: plt.Figure, path: Path, *, dpi: int = MANUSCRIPT_DPI) -> None:
    """Save both raster and vector versions for submission workflows.

    `path` may include any suffix; its stem is used for all exported formats.
    """
    base = path.with_suffix("")
    for fmt in EXPORT_FORMATS:
        out_path = base.with_suffix(f".{fmt}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not getattr(fig, "_skip_tight_layout", False):
            fig.tight_layout()
        if fmt.lower() in {"png", "tif", "tiff", "jpg", "jpeg"}:
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        else:
            # Vector formats (PDF/EPS/SVG): DPI is irrelevant; preserve tight bounds.
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
