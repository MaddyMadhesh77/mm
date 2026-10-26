"""Central configuration for the MoodTune prototype."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MUSIC_ROOT = PROJECT_ROOT / "music"

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
INFERENCE_FPS = 8

MOODS = ("happy", "sad", "angry", "neutral", "surprised")
DEFAULT_WINDOW_SECONDS = 2.5
# FER rarely averages above 0.5 for subtle moods (sad/angry) on real webcams.
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_DWELL_SECONDS = 5.0
