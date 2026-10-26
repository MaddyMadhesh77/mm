"""Backward-compatible import for the package emotion implementation."""

from moodtune.emotion import EmotionDetector, create_detector, map_emotions, top_mood

__all__ = ["EmotionDetector", "create_detector", "map_emotions", "top_mood"]
