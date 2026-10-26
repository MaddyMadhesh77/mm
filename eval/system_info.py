"""Print the values needed for report Table 4.1 (experimental parameters).

    python -m eval.system_info
"""

import platform

from moodtune import config
from moodtune.audio import AUDIO_SUFFIXES


def track_counts() -> dict[str, int]:
    counts = {}
    for mood in config.MOODS:
        directory = config.MUSIC_ROOT / mood
        counts[mood] = sum(1 for p in directory.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES) if directory.is_dir() else 0
        regulate = directory / "regulate"
        if regulate.is_dir():
            counts[f"{mood}/regulate"] = sum(1 for p in regulate.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    return counts


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            cpu = next(line.split(":", 1)[1].strip() for line in file if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    try:
        import psutil

        ram = f"{psutil.virtual_memory().total / 2**30:.0f} GB RAM"
    except ImportError:
        ram = "RAM unknown"
    return f"{cpu}, {ram}, {platform.system()} {platform.release()}"


def main() -> None:
    counts = track_counts()
    print("| Parameter | Value |\n|---|---|")
    print(f"| Programming language | Python {platform.python_version()} |")
    print("| Emotion model | FER library (Keras CNN trained on FER-2013), OpenCV Haar face detector |")
    print("| Target moods | Happy, Sad, Angry, Neutral, Surprised |")
    print(f"| Camera | Webcam, {config.FRAME_WIDTH} x {config.FRAME_HEIGHT} |")
    print(f"| Emotion analysis rate | {config.INFERENCE_FPS} frames per second |")
    print(f"| Smoothing window | {config.DEFAULT_WINDOW_SECONDS} s |")
    print(f"| Confidence threshold | {config.DEFAULT_CONFIDENCE_THRESHOLD} |")
    print(f"| Minimum dwell time | {config.DEFAULT_DWELL_SECONDS:.0f} s |")
    print("| Cross-fade duration | 2 s |")
    print(f"| Music library | {min(counts.values())}-{max(counts.values())} royalty-free tracks per mood folder ({sum(counts.values())} total) |")
    print(f"| Hardware | {hardware()}, no GPU |")
    print("| Test participants | [N] volunteers, with consent |")


if __name__ == "__main__":
    main()
