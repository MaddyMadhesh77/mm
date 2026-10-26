"""Throttled camera-to-emotion pipeline for Phase 2."""

from __future__ import annotations

import time
from typing import Optional

from .capture import CameraStream
from .config import INFERENCE_FPS
from .emotion import EmotionDetector


class PerceptionLoop:
    """Analyze the latest camera frame at a bounded rate."""

    def __init__(self, camera: CameraStream, detector: EmotionDetector, inference_fps: float = INFERENCE_FPS) -> None:
        self.camera = camera
        self.detector = detector
        self.interval = 1.0 / inference_fps
        self._last_inference = 0.0
        self.last_probabilities: Optional[dict[str, float]] = None
        self.last_inference_ms: Optional[float] = None

    def update(self, now: Optional[float] = None) -> Optional[dict[str, float]]:
        now = time.monotonic() if now is None else now
        if now - self._last_inference < self.interval:
            return self.last_probabilities
        frame = self.camera.read()
        if frame is None:
            return self.last_probabilities
        started = time.perf_counter()
        self.last_probabilities = self.detector.detect(frame)
        self.last_inference_ms = (time.perf_counter() - started) * 1000
        self._last_inference = now
        return self.last_probabilities