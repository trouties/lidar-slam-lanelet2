"""Per-stage timing utilities."""

from __future__ import annotations

import time

import numpy as np


class StageTimer:
    """Context manager that collects per-invocation wall-clock times.

    Usage::

        timer = StageTimer("Stage2")
        for frame in frames:
            with timer:
                process(frame)
        print(timer.summary())

    Can also be used as a single-shot timer::

        with StageTimer("Stage3") as t:
            do_work()
        print(t.summary())
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._laps: list[float] = []
        self._start: float | None = None

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self._start is not None:
            self._laps.append(time.perf_counter() - self._start)
            self._start = None

    @property
    def total_s(self) -> float:
        return sum(self._laps)

    def summary(self) -> dict[str, float]:
        """Return timing statistics in milliseconds."""
        if not self._laps:
            return {"n": 0, "p50": 0, "p95": 0, "max": 0, "mean": 0, "total_ms": 0}
        arr = np.array(self._laps) * 1000  # s → ms
        return {
            "n": int(len(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "total_ms": float(arr.sum()),
        }
