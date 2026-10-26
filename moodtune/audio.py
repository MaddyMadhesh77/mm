"""Two-channel pygame audio engine for mood-based cross-fading."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

from .config import MOODS, MUSIC_ROOT

AUDIO_SUFFIXES = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
REGULATE_MOODS = {"sad", "angry", "surprised"}


class AudioEngine:
    """Play mood tracks using alternating mixer channels."""

    def __init__(
        self,
        music_root: Path = MUSIC_ROOT,
        mixer: Any = None,
        sound_loader: Any = None,
        rng: Optional[random.Random] = None,
        fade_ms: int = 2000,
    ) -> None:
        self.music_root = Path(music_root)
        self._mixer = mixer
        self._sound_loader = sound_loader
        self._rng = rng or random.Random()
        self.fade_ms = fade_ms
        self._channels: list[Any] = []
        self._active_channel = 0
        self._current_path: Optional[Path] = None
        self.current_mood: Optional[str] = None
        self.current_mode = "mirror"
        self.volume = 1.0
        self.paused = False
        self.warning: Optional[str] = None

    def start(self) -> "AudioEngine":
        """Initialize pygame's mixer and reserve the two cross-fade channels."""
        if self._mixer is None:
            import pygame

            pygame.mixer.init()
            self._mixer = pygame.mixer
        if self._sound_loader is None:
            self._sound_loader = self._mixer.Sound
        if not self._channels:
            self._channels = [self._mixer.Channel(0), self._mixer.Channel(1)]
            self.set_volume(self.volume)
        return self

    def play_mood(self, mood: str, mode: str = "mirror") -> Optional[Path]:
        """Cross-fade to a random track for ``mood`` and ``mode``."""
        self._require_started()
        if mood not in MOODS:
            raise ValueError(f"Unknown mood: {mood}")
        if mode not in {"mirror", "regulate"}:
            raise ValueError("mode must be mirror or regulate")

        tracks = self._tracks_for(mood, mode)
        if not tracks:
            self.warning = f"No tracks found for {mood}/{mode}; using neutral/mirror."
            tracks = self._tracks_for("neutral", "mirror")
        if not tracks:
            self.warning = "No audio tracks are available in the music library."
            return None

        path = self._choose_track(tracks)
        new_index = 1 - self._active_channel
        old_channel = self._channels[self._active_channel]
        new_channel = self._channels[new_index]
        sound = self._sound_loader(str(path))
        old_channel.fadeout(self.fade_ms)
        new_channel.play(sound, fade_ms=self.fade_ms)
        self._active_channel = new_index
        self._current_path = path
        self.current_mood = mood
        self.current_mode = mode
        self.paused = False
        return path

    def update(self) -> Optional[Path]:
        """Queue another track when the active channel has finished."""
        if not self._channels or self.current_mood is None or self.paused:
            return None
        if self._channels[self._active_channel].get_busy():
            return None
        return self.play_mood(self.current_mood, self.current_mode)

    def set_mode(self, mode: str) -> Optional[Path]:
        """Switch Mirror/Regulate and cross-fade if the music folder changes."""
        if mode not in {"mirror", "regulate"}:
            raise ValueError("mode must be mirror or regulate")
        previous = self.current_mode
        self.current_mode = mode
        if mode == previous or self.current_mood is None or not self._channels:
            return None
        if self._tracks_for(self.current_mood, mode) == self._tracks_for(self.current_mood, previous):
            return None
        return self.play_mood(self.current_mood, mode)

    @property
    def current_track(self) -> Optional[Path]:
        return self._current_path

    def pause(self) -> None:
        self._require_started()
        for channel in self._channels:
            channel.pause()
        self.paused = True

    def resume(self) -> None:
        self._require_started()
        for channel in self._channels:
            channel.unpause()
        self.paused = False

    def set_volume(self, volume: float) -> None:
        if not 0 <= volume <= 1:
            raise ValueError("volume must be between 0 and 1")
        self.volume = volume
        for channel in self._channels:
            channel.set_volume(volume)

    def stop(self) -> None:
        for channel in self._channels:
            channel.stop()
        self.current_mood = None
        self._current_path = None
        self.paused = False

    def _tracks_for(self, mood: str, mode: str) -> list[Path]:
        directory = self.music_root / mood
        if mode == "regulate" and mood in REGULATE_MOODS:
            directory = directory / "regulate"
        if not directory.is_dir():
            return []
        return sorted(path for path in directory.iterdir() if path.suffix.lower() in AUDIO_SUFFIXES)

    def _choose_track(self, tracks: list[Path]) -> Path:
        choices = [track for track in tracks if track != self._current_path]
        return self._rng.choice(choices or tracks)

    def _require_started(self) -> None:
        if not self._channels:
            raise RuntimeError("AudioEngine.start() must be called first")