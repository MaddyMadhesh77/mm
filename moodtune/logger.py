"""CSV logging for perception and decision-layer evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Optional

from .config import MOODS

FIELDNAMES = (
    "timestamp",
    "happy",
    "sad",
    "angry",
    "neutral",
    "surprised",
    "raw_argmax",
    "smoothed_mood",
    "stable_mood",
    "switch_event",
    "inference_ms",
    "fps",
    "smoothing_enabled",
    "ground_truth",
    "cpu_percent",
)


class CsvLogger:
    """Append one normalized row for each analyzed frame."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def log(
        self,
        timestamp: float,
        probabilities: Mapping[str, float],
        smoothed_mood: str,
        stable_mood: str,
        switch_event: bool,
        inference_ms: Optional[float],
        fps: float,
        smoothing_enabled: bool,
        ground_truth: Optional[str] = None,
        cpu_percent: Optional[float] = None,
    ) -> None:
        values = {mood: float(probabilities.get(mood, 0.0)) for mood in MOODS}
        raw_argmax = max(MOODS, key=values.__getitem__)
        row = {
            "timestamp": timestamp,
            **values,
            "raw_argmax": raw_argmax,
            "smoothed_mood": smoothed_mood,
            "stable_mood": stable_mood,
            "switch_event": int(switch_event),
            "inference_ms": "" if inference_ms is None else inference_ms,
            "fps": fps,
            "smoothing_enabled": int(smoothing_enabled),
            "ground_truth": ground_truth or "",
            "cpu_percent": "" if cpu_percent is None else cpu_percent,
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CsvLogger":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()