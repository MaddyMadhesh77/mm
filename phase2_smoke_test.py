"""Run the Phase 2 camera smoke test.

This opens the webcam, displays the preview, and overlays the top mood and
inference FPS. Press ``q`` to quit. It requires FER and a working camera.
"""

import time

import cv2

from moodtune.capture import CameraStream
from moodtune.config import FRAME_HEIGHT, FRAME_WIDTH
from moodtune.emotion import EmotionDetector, top_mood
from moodtune.perception import PerceptionLoop


def main() -> int:
    camera = CameraStream(width=FRAME_WIDTH, height=FRAME_HEIGHT)
    try:
        camera.start()
        pipeline = PerceptionLoop(camera, EmotionDetector())
        previous = time.monotonic()
        fps = 0.0
        while True:
            now = time.monotonic()
            elapsed = now - previous
            previous = now
            if elapsed:
                fps = 0.9 * fps + 0.1 / elapsed
            probabilities = pipeline.update(now)
            frame = camera.read()
            if frame is None:
                continue
            label = "No face detected"
            if probabilities:
                mood, confidence = top_mood(probabilities)
                label = f"{mood}: {confidence:.2f}"
            cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            cv2.imshow("MoodTune Phase 2", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())