"""Automated versions of report Table 4.2 functional cases (decision + audio + visual).

T1 and T9 need a real camera and are checked manually (see docs/TESTING.md).
"""

from moodtune.audio import AudioEngine
from moodtune.decision import DecisionEngine
from moodtune.visual import css_for_mood, theme_for

from tests.test_phase4 import FakeMixer, FakeSound, make_library


def probs(mood, confidence=0.9):
    values = {name: (1 - confidence) / 4 for name in ("happy", "sad", "angry", "neutral", "surprised")}
    values[mood] = confidence
    return values


def pipeline(tmp_path, dwell=15):
    make_library(tmp_path)
    mixer = FakeMixer()
    audio = AudioEngine(tmp_path, mixer=mixer, sound_loader=FakeSound).start()
    decision = DecisionEngine(window_seconds=2, confidence_threshold=0.5, dwell_seconds=dwell)
    audio.play_mood(decision.stable_mood)
    return decision, audio, mixer


def step(decision, audio, probabilities, t):
    mood, _, changed = decision.update(probabilities, t)
    if audio.current_mood != mood:
        audio.play_mood(mood)
    return mood, changed


def test_t2_no_face_keeps_mood_and_music(tmp_path):
    decision, audio, _ = pipeline(tmp_path)
    track = audio.current_track
    for t in range(20):
        assert step(decision, audio, None, t) == ("neutral", False)
    assert audio.current_track == track


def test_t3_neutral_to_smile_switches_after_smoothing_and_dwell(tmp_path):
    decision, audio, _ = pipeline(tmp_path, dwell=0)
    t = 0.0
    while decision.stable_mood != "happy" and t < 10:
        step(decision, audio, probs("happy"), t)
        t += 0.125
    assert decision.stable_mood == "happy"
    assert audio.current_track.name == "happy.mp3"
    assert theme_for("happy")["start"] == "#ffe066"  # warm yellow


def test_t4_single_frame_blink_does_not_change_mood(tmp_path):
    decision, audio, _ = pipeline(tmp_path, dwell=0)
    for i in range(16):
        step(decision, audio, probs("neutral"), i * 0.125)
    mood, changed = step(decision, audio, probs("angry", 1.0), 2.0)
    assert (mood, changed) == ("neutral", False)


def test_t5_mood_change_crossfades(tmp_path):
    decision, audio, mixer = pipeline(tmp_path, dwell=0)
    old = audio._active_channel
    decision.set_manual_override("happy", 1)
    step(decision, audio, None, 1)
    assert ("fadeout", 2000) in mixer.channels[old].events
    assert mixer.channels[1 - old].events[-1][2] == 2000  # fade-in duration


def test_t6_low_confidence_retains_previous_mood(tmp_path):
    decision, audio, _ = pipeline(tmp_path, dwell=0)
    for i in range(30):
        mood, changed = step(decision, audio, probs("sad", 0.35), i * 0.125)
        assert (mood, changed) == ("neutral", False)


def test_t7_manual_override_until_automatic_resumes(tmp_path):
    decision, audio, _ = pipeline(tmp_path, dwell=0)
    decision.set_manual_override("happy", 0)
    for i in range(20):
        assert step(decision, audio, probs("sad"), i * 0.125)[0] == "happy"
    decision.set_manual_override(None)
    t = 3.0
    while decision.stable_mood != "sad" and t < 10:
        step(decision, audio, probs("sad"), t)
        t += 0.125
    assert decision.stable_mood == "sad"


def test_t8_missing_folder_warns_and_falls_back_to_neutral(tmp_path):
    decision, audio, _ = pipeline(tmp_path)
    path = audio.play_mood("angry")  # no angry folder in the fake library
    assert path.name == "neutral.mp3"
    assert "neutral" in audio.warning


def test_mode_switch_crossfades_to_regulate_library(tmp_path):
    _, audio, _ = pipeline(tmp_path)
    audio.play_mood("sad")  # no sad/ mirror folder -> neutral fallback
    path = audio.set_mode("regulate")
    assert path.name == "calm.mp3"
    assert audio.set_mode("regulate") is None  # no-op when unchanged


def test_regulate_mode_uses_teal_for_angry():
    assert theme_for("angry", "regulate")["start"] == "#2a9d8f"
    assert theme_for("angry")["start"] != theme_for("angry", "regulate")["start"]
    assert "#2a9d8f" in css_for_mood("angry", "regulate")
