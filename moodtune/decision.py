"""Temporal smoothing and stable-mood decisions."""

from __future__ import annotations

from collections import deque
from typing import Mapping, Optional

from .config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_DWELL_SECONDS, DEFAULT_WINDOW_SECONDS, MOODS


class DecisionEngine:
    """Turn noisy per-frame probabilities into a stable mood label."""

    def __init__(
        self,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        dwell_seconds: float = DEFAULT_DWELL_SECONDS,
        initial_mood: str = "neutral",
        smoothing_enabled: bool = True,
    ) -> None:
        if window_seconds <= 0 or dwell_seconds < 0:
            raise ValueError("window_seconds must be positive and dwell_seconds non-negative")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if initial_mood not in MOODS:
            raise ValueError(f"Unknown initial mood: {initial_mood}")
        self.window_seconds = window_seconds
        self.confidence_threshold = confidence_threshold
        self.dwell_seconds = dwell_seconds
        self.stable_mood = initial_mood
        self._last_change_time: Optional[float] = None
        self._samples: deque[tuple[float, dict[str, float]]] = deque()
        self.manual_mood: Optional[str] = None
        self.smoothing_enabled = smoothing_enabled
        self.last_smoothed_mood = initial_mood
        # Why the last update did or did not switch; shown in the dashboard.
        self.last_reason = "waiting"
        self.dwell_remaining = 0.0

    def update(self, probabilities: Optional[Mapping[str, float]], timestamp: float) -> tuple[str, float, bool]:
        """Return ``(stable_mood, confidence, changed)`` for one observation."""
        if self.manual_mood is not None:
            self.last_reason = "manual"
            return self._apply_manual(timestamp)
        if probabilities is None:
            self.last_reason = "no_face"
            return self.stable_mood, 0.0, False

        sample = {mood: max(0.0, float(probabilities.get(mood, 0.0))) for mood in MOODS}
        self._samples.append((timestamp, sample))
        if not self.smoothing_enabled:
            self._samples.clear()
            self._samples.append((timestamp, sample))
        cutoff = timestamp - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        averages = {
            mood: sum(sample[mood] for _, sample in self._samples) / len(self._samples)
            for mood in MOODS
        }
        candidate = max(MOODS, key=averages.__getitem__)
        confidence = averages[candidate]
        self.last_smoothed_mood = candidate
        self.dwell_remaining = 0.0
        if candidate == self.stable_mood:
            self.last_reason = "stable"
            return self.stable_mood, confidence, False
        if confidence < self.confidence_threshold:
            self.last_reason = "low_confidence"
            return self.stable_mood, confidence, False
        if self._last_change_time is not None and timestamp - self._last_change_time < self.dwell_seconds:
            self.last_reason = "dwell"
            self.dwell_remaining = self.dwell_seconds - (timestamp - self._last_change_time)
            return self.stable_mood, confidence, False

        self.last_reason = "switched"
        self.stable_mood = candidate
        self._last_change_time = timestamp
        return self.stable_mood, confidence, True

    def set_manual_override(self, mood: Optional[str], timestamp: float = 0.0) -> tuple[str, float, bool]:
        """Set or clear the override; setting it applies immediately."""
        if mood is not None and mood not in MOODS:
            raise ValueError(f"Unknown mood: {mood}")
        self.manual_mood = mood
        if mood is None:
            return self.stable_mood, 0.0, False
        return self._apply_manual(timestamp)

    def _apply_manual(self, timestamp: float) -> tuple[str, float, bool]:
        assert self.manual_mood is not None
        changed = self.stable_mood != self.manual_mood
        self.stable_mood = self.manual_mood
        if changed:
            self._last_change_time = timestamp
        return self.stable_mood, 1.0, changed


MoodDecision = DecisionEngine