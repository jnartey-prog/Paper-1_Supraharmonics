"""Command-line interface for interactive pipeline execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import default_config, run_pipeline
from .config import save_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supraharmonic-pipeline",
        description="Guided pipeline for supraharmonic aggregation analysis.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="manuscript/artifacts",
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Generate default config file and run with defaults.",
    )
    parser.add_argument(
        "--write-default-config",
        type=str,
        default=None,
        help="Write default configuration JSON to this path and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run CLI pipeline and return process status code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.write_default_config:
        path = save_config(default_config(), args.write_default_config)
        print(f"Wrote default config: {path}")
        return 0

    config_path = args.config
    if args.quickstart and not config_path:
        path = Path(args.output_dir) / "default_config.json"
        save_config(default_config(), str(path))
        config_path = str(path)

    try:
        run = run_pipeline(config_path=config_path, output_dir=args.output_dir)
    except Exception as exc:  # pragma: no cover - exercised in CLI failure paths
        print(f"Pipeline failed: {exc}")
        return 1

    summary = {
        "run_id": run.run_id,
        "artifacts_generated": len(run.artifact_paths),
        "manifest_path": run.run_manifest_path,
        "log_path": run.log_path,
        "stats_log_path": run.stats_log_path,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
