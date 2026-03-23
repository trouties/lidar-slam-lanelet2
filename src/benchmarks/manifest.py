"""Benchmark manifest: append-only JSON log of benchmark runs."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmarks.git_info import get_git_sha
from src.cache.layered_cache import compute_hash


class BenchmarkManifest:
    """Append-only JSON log stored at ``benchmarks/benchmark_manifest.json``.

    Each entry records a benchmark run with git SHA, config hash, timestamp,
    and pointers to result artifacts.
    """

    DEFAULT_PATH = Path("benchmarks/benchmark_manifest.json")

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict[str, Any]]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return []

    def _save(self, entries: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")

    def append(
        self,
        task: str,
        config: dict,
        sequences: list[str],
        artifacts: list[str],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Append a new benchmark run record.

        Args:
            task: SUP task id, e.g. ``"SUP-01"``.
            config: Full config dict (hashed for reproducibility).
            sequences: List of sequence ids used.
            artifacts: List of artifact file paths produced.
            metrics: Summary metrics dict.

        Returns:
            The created record (for inspection).
        """
        record = {
            "run_id": uuid.uuid4().hex[:12],
            "task": task,
            "git_sha": get_git_sha(),
            "config_hash": compute_hash(config),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequences": sequences,
            "artifacts": artifacts,
            "metrics": metrics,
        }
        entries = self._load()
        entries.append(record)
        self._save(entries)
        return record
