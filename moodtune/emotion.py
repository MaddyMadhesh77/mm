"""FER adapter and seven-emotion to five-mood mapping."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from .config import MOODS

EMOTION_KEYS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def create_detector():
    """Create FER with the faster OpenCV detector configuration."""
    try:
        from fer import FER
    except ImportError:  # fer >= 25 only exposes the class from fer.fer
        from fer.fer import FER

    return FER(mtcnn=False)


def map_emotions(emotions: Mapping[str, float]) -> dict[str, float]:
    """Fold FER's seven probabilities into the five application moods."""
    values = {key: float(emotions.get(key, 0.0)) for key in EMOTION_KEYS}
    return {
        "happy": values["happy"],
        "sad": values["sad"],
        "neutral": values["neutral"],
        "angry": values["angry"] + values["disgust"],
        "surprised": values["surprise"] + values["fear"],
    }


class EmotionDetector:
    """Select the largest detected face and return its five mood scores."""

    def __init__(self, detector: Any = None) -> None:
        self.detector = detector or create_detector()
        self.last_box: Optional[tuple[int, int, int, int]] = None

    def detect(self, frame: np.ndarray) -> Optional[dict[str, float]]:
        detections = self.detector.detect_emotions(frame)
        if not detections:
            self.last_box = None
            return None
        largest = max(detections, key=lambda detection: self._area(detection.get("box")))
        box = largest.get("box")
        self.last_box = tuple(int(value) for value in box[:4]) if box is not None and len(box) >= 4 else None
        return map_emotions(largest.get("emotions", {}))

    @staticmethod
    def _area(box: Any) -> int:
        if box is None or len(box) < 4:  # fer >= 25 returns numpy arrays
            return 0
        return max(0, int(box[2])) * max(0, int(box[3]))


def top_mood(probabilities: Mapping[str, float]) -> tuple[str, float]:
    """Return the highest-probability mood and confidence."""
    mood = max(MOODS, key=lambda name: probabilities.get(name, 0.0))
    return mood, float(probabilities.get(mood, 0.0))