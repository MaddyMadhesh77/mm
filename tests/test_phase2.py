import numpy as np

from moodtune.emotion import EmotionDetector, map_emotions
from moodtune.perception import PerceptionLoop


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def detect_emotions(self, _frame):
        self.calls += 1
        return [
            {"box": [0, 0, 2, 2], "emotions": {"happy": 0.9}},
            {
                "box": [0, 0, 10, 10],
                "emotions": {"fear": 0.4, "surprise": 0.2, "disgust": 0.3, "angry": 0.1},
            },
        ]


class FakeCamera:
    def read(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)


def test_emotion_mapping_folds_fer_classes():
    mapped = map_emotions({"fear": 0.4, "surprise": 0.2, "disgust": 0.3, "angry": 0.1})

    assert abs(mapped["surprised"] - 0.6) < 1e-9
    assert abs(mapped["angry"] - 0.4) < 1e-9


def test_detector_selects_largest_face():
    result = EmotionDetector(FakeDetector()).detect(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result is not None
    assert abs(result["surprised"] - 0.6) < 1e-9


def test_perception_loop_throttles_inference():
    detector = FakeDetector()
    loop = PerceptionLoop(FakeCamera(), EmotionDetector(detector), inference_fps=2)

    loop.update(now=1.0)
    loop.update(now=1.1)

    assert detector.calls == 1


def test_numpy_boxes_from_newer_fer_are_supported():
    class ArrayBoxDetector:
        def detect_emotions(self, _frame):
            return [
                {"box": np.array([0, 0, 2, 2]), "emotions": {"sad": 0.9}},
                {"box": np.array([5, 5, 20, 20]), "emotions": {"happy": 0.8}},
            ]

    detector = EmotionDetector(ArrayBoxDetector())
    result = detector.detect(np.zeros((40, 40, 3), dtype=np.uint8))

    assert result["happy"] == 0.8
    assert detector.last_box == (5, 5, 20, 20)


if __name__ == "__main__":
    test_emotion_mapping_folds_fer_classes()
    test_detector_selects_largest_face()
    test_perception_loop_throttles_inference()
    print("Phase 2 behavioral tests passed.")