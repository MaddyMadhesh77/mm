from pathlib import Path
from tempfile import TemporaryDirectory

from moodtune.audio import AudioEngine


class FakeSound:
    def __init__(self, path):
        self.path = path


class FakeChannel:
    def __init__(self):
        self.events = []
        self.busy = False

    def fadeout(self, milliseconds):
        self.events.append(("fadeout", milliseconds))

    def play(self, sound, fade_ms=0):
        self.events.append(("play", sound.path, fade_ms))
        self.busy = True

    def get_busy(self):
        return self.busy

    def pause(self):
        self.events.append(("pause",))

    def unpause(self):
        self.events.append(("unpause",))

    def stop(self):
        self.events.append(("stop",))

    def set_volume(self, volume):
        self.events.append(("volume", volume))


class FakeMixer:
    def __init__(self):
        self.channels = [FakeChannel(), FakeChannel()]

    def Channel(self, index):
        return self.channels[index]


def make_library(root: Path):
    (root / "neutral").mkdir(parents=True)
    (root / "happy").mkdir()
    (root / "sad" / "regulate").mkdir(parents=True)
    (root / "neutral" / "neutral.mp3").touch()
    (root / "happy" / "happy.mp3").touch()
    (root / "sad" / "regulate" / "calm.mp3").touch()


def test_crossfade_and_regulate_selection(tmp_path):
    make_library(tmp_path)
    mixer = FakeMixer()
    engine = AudioEngine(tmp_path, mixer=mixer, sound_loader=FakeSound)
    engine.start()

    selected = engine.play_mood("sad", "regulate")

    assert selected == tmp_path / "sad" / "regulate" / "calm.mp3"
    assert mixer.channels[0].events[-1] == ("fadeout", 2000)
    assert mixer.channels[1].events[-1] == ("play", str(selected), 2000)


def test_missing_mood_falls_back_to_neutral(tmp_path):
    make_library(tmp_path)
    mixer = FakeMixer()
    engine = AudioEngine(tmp_path, mixer=mixer, sound_loader=FakeSound)
    engine.start()

    selected = engine.play_mood("angry")

    assert selected == tmp_path / "neutral" / "neutral.mp3"
    assert engine.warning is not None


def test_controls_and_finished_track_queue(tmp_path):
    make_library(tmp_path)
    mixer = FakeMixer()
    engine = AudioEngine(tmp_path, mixer=mixer, sound_loader=FakeSound)
    engine.start()
    engine.play_mood("happy")
    engine.set_volume(0.4)
    engine.pause()
    engine.resume()
    mixer.channels[engine._active_channel].busy = False

    assert engine.update() == tmp_path / "happy" / "happy.mp3"
    assert engine.volume == 0.4


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        root = Path(directory)
        test_crossfade_and_regulate_selection(root)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        test_missing_mood_falls_back_to_neutral(root)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        test_controls_and_finished_track_queue(root)
    print("Phase 4 behavioral tests passed.")