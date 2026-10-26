"""Shared background coordinator for the Streamlit dashboard."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

from .audio import AudioEngine
from .capture import CameraStream
from .decision import DecisionEngine
from .emotion import EmotionDetector
from .logger import CsvLogger
from .perception import PerceptionLoop
from .visual import theme_for


@dataclass
class DashboardState:
    frame: object = None
    mood: str = "neutral"
    confidence: float = 0.0
    smoothed_mood: str = "neutral"
    raw_mood: Optional[str] = None
    reason: str = "waiting"
    dwell_remaining: float = 0.0
    threshold: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    inference_ms: Optional[float] = None
    fps: float = 0.0
    cpu_percent: Optional[float] = None
    track: Optional[str] = None
    warning: Optional[str] = None
    error: Optional[str] = None
    no_face: bool = True
    paused: bool = False
    log_path: Optional[str] = None


class DashboardRuntime:
    """Run perception, decision, and audio work outside Streamlit reruns."""

    def __init__(self) -> None:
        self.state = DashboardState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.camera: Optional[CameraStream] = None
        self.audio: Optional[AudioEngine] = None
        self.decision = DecisionEngine()
        self.perception: Optional[PerceptionLoop] = None
        self.logger: Optional[CsvLogger] = None
        self.mode = "mirror"
        self.paused = False
        self.ground_truth: Optional[str] = None
        self._last_logged_inference: Optional[float] = None
        self._process = None

    def start(self) -> "DashboardRuntime":
        if self._thread and self._thread.is_alive():
            return self
        try:
            self.camera = CameraStream().start()
            self.perception = PerceptionLoop(self.camera, EmotionDetector())
            self.audio = AudioEngine().start()
            try:
                import psutil

                self._process = psutil.Process()
                self._process.cpu_percent(None)
            except ImportError:
                self._process = None
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="moodtune-dashboard", daemon=True)
            self._thread.start()
        except Exception as error:
            self.stop()
            self._set_error(str(error))
        return self

    def configure(self, window: float, threshold: float, dwell: float, smoothing_enabled: bool = True) -> None:
        self.decision.window_seconds = window
        self.decision.confidence_threshold = threshold
        self.decision.dwell_seconds = dwell
        self.decision.smoothing_enabled = smoothing_enabled

    def set_logging(self, path: Optional[str]) -> None:
        """Start logging to ``path``, or stop logging when ``path`` is None."""
        with self._lock:
            if self.logger:
                self.logger.close()
                self.logger = None
            if path:
                self.logger = CsvLogger(path)
            self.state.log_path = path

    def set_manual_mood(self, mood: Optional[str]) -> None:
        if mood != self.decision.manual_mood:
            self.decision.set_manual_override(mood, time.monotonic())

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if self.audio:
            self.audio.set_mode(mode)

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        if not self.audio:
            return
        if paused and not self.audio.paused:
            self.audio.pause()
        elif not paused and self.audio.paused:
            self.audio.resume()

    def set_volume(self, volume: float) -> None:
        if self.audio:
            self.audio.set_volume(volume)

    def snapshot(self) -> DashboardState:
        with self._lock:
            return DashboardState(**self.state.__dict__)

    def _run(self) -> None:
        assert self.perception is not None
        previous = time.monotonic()
        fps = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            elapsed = now - previous
            previous = now
            if elapsed:
                fps = 0.9 * fps + 0.1 / elapsed

            if self.paused:
                probabilities = None
                inferred = False
            else:
                before = self.perception._last_inference
                probabilities = self.perception.update(now)
                inferred = self.perception._last_inference != before

            # Only new inferences feed the moving window; manual override applies every tick.
            if inferred or self.decision.manual_mood is not None:
                mood, confidence, changed = self.decision.update(probabilities, now)
            else:
                mood, confidence, changed = self.decision.stable_mood, self.state.confidence, False

            if self.audio and not self.paused:
                if self.audio.current_mood != mood:
                    self.audio.play_mood(mood, self.mode)
                else:
                    self.audio.update()

            cpu = self._process.cpu_percent(None) if self._process and inferred else self.state.cpu_percent
            frame = self.camera.read() if self.camera else None
            if frame is not None:
                frame = self._annotate(frame, mood, probabilities is None and not self.paused)

            with self._lock:
                self.state.frame = frame
                self.state.mood = mood
                self.state.confidence = confidence
                self.state.smoothed_mood = self.decision.last_smoothed_mood
                self.state.raw_mood = max(probabilities, key=probabilities.get) if probabilities else None
                self.state.reason = "paused" if self.paused else self.decision.last_reason
                self.state.dwell_remaining = self.decision.dwell_remaining
                self.state.threshold = self.decision.confidence_threshold
                self.state.probabilities = probabilities or {}
                self.state.inference_ms = self.perception.last_inference_ms
                self.state.fps = fps
                self.state.cpu_percent = cpu
                self.state.track = Path(self.audio.current_track).name if self.audio and self.audio.current_track else None
                self.state.warning = self.audio.warning if self.audio else None
                self.state.no_face = probabilities is None
                self.state.paused = self.paused
                self.state.error = self.camera.error if self.camera else None
                if inferred and probabilities is not None and self.logger:
                    self.logger.log(
                        now,
                        probabilities,
                        self.decision.last_smoothed_mood,
                        mood,
                        changed,
                        self.perception.last_inference_ms,
                        fps,
                        self.decision.smoothing_enabled,
                        ground_truth=self.ground_truth,
                        cpu_percent=cpu,
                    )
            time.sleep(0.02)

    def _annotate(self, frame, mood: str, no_face: bool):
        """Draw the face box and stable mood; frames are never stored."""
        box = self.perception.detector.last_box if self.perception else None
        if box and not no_face:
            x, y, w, h = box
            hex_colour = theme_for(mood, self.mode)["end"].lstrip("#")
            r, g, b = (int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (b, g, r), 2)
            cv2.putText(frame, mood.upper(), (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (b, g, r), 2)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _set_error(self, message: str) -> None:
        with self._lock:
            self.state.error = message

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.camera:
            self.camera.stop()
        if self.audio:
            self.audio.stop()
        if self.logger:
            self.logger.close()
            self.logger = None
