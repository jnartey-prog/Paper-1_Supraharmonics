from __future__ import annotations

from pathlib import Path

import pytest

from supraharmonic_aggregation.cli import main


@pytest.mark.pipeline
def test_cli_pipeline_quickstart(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    status = main(["--quickstart", "--output-dir", str(output_dir)])
    assert status == 0
    assert (output_dir / "run_manifest.json").exists()


@pytest.mark.pipeline
def test_cli_pipeline_writes_logs(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    status = main(["--quickstart", "--output-dir", str(output_dir)])
    assert status == 0
    logs_root = output_dir / "logs"
    # Quickstart config writes logs under default path if not overridden.
    # Accept either global default logs path or local output logs path.
    assert Path("logs").exists() or logs_root.exists()
