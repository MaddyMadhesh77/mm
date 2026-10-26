from moodtune.decision import DecisionEngine


def probabilities(mood, confidence=0.9):
    values = {name: 0.0 for name in ("happy", "sad", "angry", "neutral", "surprised")}
    values[mood] = confidence
    return values


def test_steady_neutral_stays_stable():
    engine = DecisionEngine(window_seconds=2, dwell_seconds=0)
    for timestamp in range(5):
        mood, _, changed = engine.update(probabilities("neutral"), timestamp)
        assert mood == "neutral"
        assert not changed


def test_one_frame_spike_does_not_switch():
    engine = DecisionEngine(window_seconds=2, confidence_threshold=0.5, dwell_seconds=0)
    engine.update(probabilities("neutral"), 0)
    mood, _, changed = engine.update(probabilities("happy"), 1)

    assert mood == "neutral"
    assert not changed


def test_low_confidence_does_not_switch():
    engine = DecisionEngine(window_seconds=1, confidence_threshold=0.5, dwell_seconds=0)
    mood, confidence, changed = engine.update(probabilities("happy", confidence=0.4), 0)

    assert mood == "neutral"
    assert confidence == 0.4
    assert not changed


def test_genuine_change_requires_dwell_time():
    engine = DecisionEngine(window_seconds=1, confidence_threshold=0.5, dwell_seconds=15)
    engine.update(probabilities("happy"), 0)
    mood, _, changed = engine.update(probabilities("sad"), 5)
    assert mood == "happy"
    assert not changed

    mood, _, changed = engine.update(probabilities("sad"), 16)
    assert mood == "sad"
    assert changed


def test_no_face_does_not_add_a_sample():
    engine = DecisionEngine(window_seconds=2, dwell_seconds=0)
    engine.update(probabilities("happy"), 0)
    mood, confidence, changed = engine.update(None, 1)

    assert mood == "happy"
    assert confidence == 0.0
    assert not changed


def test_manual_override_bypasses_smoothing_and_dwell():
    engine = DecisionEngine(dwell_seconds=100)
    mood, confidence, changed = engine.set_manual_override("angry", timestamp=1)

    assert (mood, confidence, changed) == ("angry", 1.0, True)
    mood, _, changed = engine.update(probabilities("happy"), 2)
    assert mood == "angry"
    assert not changed


def test_reasons_explain_why_mood_is_held():
    engine = DecisionEngine(window_seconds=1, confidence_threshold=0.5, dwell_seconds=10)
    engine.update(probabilities("happy", 0.4), 0)
    assert engine.last_reason == "low_confidence"
    engine.update(probabilities("happy"), 2)
    assert engine.last_reason == "switched"
    engine.update(probabilities("sad"), 5)
    assert engine.last_reason == "dwell"
    assert engine.dwell_remaining == 7
    engine.update(None, 6)
    assert engine.last_reason == "no_face"


def test_default_threshold_accepts_typical_webcam_sad_scores():
    engine = DecisionEngine(window_seconds=2.5, dwell_seconds=0)
    values = {"happy": 0.1, "sad": 0.42, "angry": 0.12, "neutral": 0.3, "surprised": 0.06}
    for step in range(10):
        mood, _, _ = engine.update(values, step * 0.125)
    assert mood == "sad"


if __name__ == "__main__":
    test_steady_neutral_stays_stable()
    test_one_frame_spike_does_not_switch()
    test_low_confidence_does_not_switch()
    test_genuine_change_requires_dwell_time()
    test_no_face_does_not_add_a_sample()
    test_manual_override_bypasses_smoothing_and_dwell()
    print("Phase 3 behavioral tests passed.")