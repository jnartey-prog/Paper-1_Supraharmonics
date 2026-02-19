"""Structured log writing helpers."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(slots=True)
class StructuredLogger:
    """Persist runtime and statistics logs in JSON/JSONL format."""

    run_id: str
    log_dir: Path
    run_dir: Path = field(init=False)
    stats_dir: Path = field(init=False)
    run_log_path: Path = field(init=False)
    stats_log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.run_dir = self.log_dir / "runs"
        self.stats_dir = self.log_dir / "stats"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.run_dir / f"{self.run_id}.jsonl"
        self.stats_log_path = self.stats_dir / f"{self.run_id}.json"

    def log_event(
        self,
        stage: str,
        event: str,
        status: str,
        duration_ms: int,
        component: str,
        message: str,
        trace_id: str | None = None,
    ) -> None:
        """Append one runtime event record."""
        payload = {
            "run_id": self.run_id,
            "timestamp_utc": _utc_now(),
            "stage": stage,
            "event": event,
            "status": status,
            "duration_ms": duration_ms,
            "component": component,
            "message": message,
            "trace_id": trace_id or str(uuid.uuid4()),
        }
        with self.run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def log_statistics(self, payload: dict[str, Any]) -> None:
        """Write statistics payload for this run."""
        envelope = {"run_id": self.run_id, "timestamp_utc": _utc_now(), "statistics": payload}
        self.stats_log_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")


def create_logger(run_id: str, log_dir: str) -> StructuredLogger:
    """Create a configured structured logger."""
    return StructuredLogger(run_id=run_id, log_dir=Path(log_dir))
