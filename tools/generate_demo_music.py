"""Synthesize original, royalty-free demo tracks for every mood folder.

These placeholder tracks let the full pipeline run before real tracks from
Pixabay / Free Music Archive are added. Each style follows report Appendix B.

    python -m tools.generate_demo_music            # 5 tracks per folder, 40 s each
    python -m tools.generate_demo_music --count 2 --seconds 20
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
import wave
import zlib
from pathlib import Path

import numpy as np

from moodtune.config import MUSIC_ROOT

SR = 44100
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]

# folder -> style parameters
STYLES = {
    "happy": dict(name="Upbeat pop", bpm=(118, 128), scale=MAJOR, prog=[[0, 4, 5, 3], [0, 5, 3, 4]],
                  lead="pluck", chord="pluck", bass="pluck", drums="pop", root=(60, 67)),
    "sad": dict(name="Calm acoustic", bpm=(66, 76), scale=MINOR, prog=[[0, 5, 2, 6], [0, 3, 5, 4]],
                lead=None, chord="arp", bass="pluck", drums=None, root=(55, 60)),
    "sad/regulate": dict(name="Gentle hopeful acoustic", bpm=(78, 88), scale=MAJOR, prog=[[0, 4, 5, 3], [3, 4, 0, 5]],
                         lead="pluck", chord="arp", bass="pluck", drums=None, root=(57, 64)),
    "angry": dict(name="Intense rock/electronic", bpm=(138, 150), scale=MINOR, prog=[[0, 0, 5, 6], [0, 6, 5, 6]],
                  lead="saw", chord="saw", bass="saw", drums="rock", root=(40, 45)),
    "angry/regulate": dict(name="Slow calming ambient", bpm=(56, 64), scale=MAJOR, prog=[[0, 3, 0, 4], [0, 5, 3, 4]],
                           lead=None, chord="pad", bass="pad", drums=None, root=(48, 55)),
    "neutral": dict(name="Relaxed lo-fi", bpm=(80, 90), scale=MAJOR, prog=[[1, 4, 0, 5], [3, 2, 1, 4]],
                    lead="epiano", chord="epiano", bass="pluck", drums="lofi", root=(57, 62), sevenths=True),
    "surprised": dict(name="Energetic playful electronic", bpm=(124, 132), scale=MAJOR, prog=[[0, 5, 3, 4], [5, 3, 0, 4]],
                      lead="square", chord="square_arp", bass="square", drums="edm", root=(60, 65)),
    "surprised/regulate": dict(name="Light bright instrumental", bpm=(96, 104), scale=MAJOR, prog=[[0, 3, 4, 0], [0, 5, 3, 4]],
                               lead="bell", chord="arp", bass="pluck", drums=None, root=(64, 69)),
}


def freq(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def envelope(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    env = np.exp(-t / decay) if decay else np.ones(n)
    a = max(1, int(attack * SR))
    env[:a] *= np.linspace(0, 1, min(a, n))
    release = min(n, int(0.02 * SR))
    env[-release:] *= np.linspace(1, 0, release)
    return env


def tone(midi: float, seconds: float, kind: str) -> np.ndarray:
    n = max(1, int(seconds * SR))
    t = np.arange(n) / SR
    f = freq(midi)
    if kind in ("pluck", "arp"):
        sig = sum(np.sin(2 * np.pi * f * h * t) / h ** 1.3 for h in range(1, 7))
        return sig * envelope(n, 0.003, 0.35)
    if kind == "pad":
        sig = sum(np.sin(2 * np.pi * f * d * t) for d in (0.996, 1.0, 1.004))
        sig += 0.3 * np.sin(2 * np.pi * f * 2 * t)
        return sig / 3 * envelope(n, min(1.5, seconds / 3), 0) * np.linspace(1, 0.6, n)
    if kind == "saw":
        sig = sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, 12))
        return np.tanh(2.5 * sig) * envelope(n, 0.005, 0.6)
    if kind in ("square", "square_arp"):
        sig = sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, 12, 2))
        return sig * envelope(n, 0.003, 0.18)
    if kind == "bell":
        sig = sum(a * np.sin(2 * np.pi * f * r * t) for r, a in ((1, 1), (2.76, 0.4), (5.4, 0.2)))
        return sig * envelope(n, 0.002, 0.7)
    if kind == "epiano":
        trem = 1 + 0.15 * np.sin(2 * np.pi * 5 * t)
        sig = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t)
        return sig * trem * envelope(n, 0.01, 0.9)
    raise ValueError(kind)


def drum(kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "kick":
        n = int(0.35 * SR)
        t = np.arange(n) / SR
        sweep = 45 + 90 * np.exp(-t * 30)
        return np.sin(2 * np.pi * np.cumsum(sweep) / SR) * np.exp(-t * 9)
    if kind == "snare":
        n = int(0.2 * SR)
        t = np.arange(n) / SR
        return (0.7 * rng.uniform(-1, 1, n) + 0.3 * np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 22)
    if kind == "hat":
        n = int(0.06 * SR)
        noise = np.diff(rng.uniform(-1, 1, n + 1))
        return noise * np.exp(-np.arange(n) / SR * 70) * 0.5
    raise ValueError(kind)


def add(buffer: np.ndarray, start: float, signal: np.ndarray, gain: float) -> None:
    i = int(start * SR)
    if i >= len(buffer):
        return
    j = min(len(buffer), i + len(signal))
    buffer[i:j] += gain * signal[: j - i]


def chord_notes(style: dict, root: int, degree: int) -> list[int]:
    scale = style["scale"]
    count = 4 if style.get("sevenths") else 3
    notes = []
    for k in range(count):
        step = degree + 2 * k
        notes.append(root + scale[step % 7] + 12 * (step // 7))
    return notes


def render(style: dict, seconds: float, rng: np.random.Generator) -> np.ndarray:
    bpm = rng.integers(*style["bpm"])
    beat = 60.0 / bpm
    root = int(rng.integers(*style["root"]))
    prog = style["prog"][int(rng.integers(len(style["prog"])))]
    bars = max(4, int(seconds / (4 * beat)))
    out = np.zeros(int(bars * 4 * beat * SR) + SR * 2)
    scale = style["scale"]
    motif = rng.integers(0, 7, size=8)

    for bar in range(bars):
        t0 = bar * 4 * beat
        notes = chord_notes(style, root, prog[bar % len(prog)])
        chord = style["chord"]
        if chord in ("arp", "square_arp"):
            step = beat / (4 if chord == "square_arp" else 2)
            pattern = notes + notes[1:-1][::-1]
            for k in range(int(4 * beat / step)):
                kind = "square" if chord == "square_arp" else "pluck"
                add(out, t0 + k * step, tone(pattern[k % len(pattern)] + 12 * (chord == "square_arp"), step * 2, kind), 0.18)
        elif chord == "pad":
            for note in notes:
                add(out, t0, tone(note, 4 * beat, "pad"), 0.18)
        else:
            hits = (0, 1.5, 2, 3) if chord in ("pluck", "saw") else (0, 2.5)
            for h in hits:
                for note in notes:
                    add(out, t0 + h * beat, tone(note, beat * 1.5, chord), 0.11)

        bass = style["bass"]
        bass_note = notes[0] - 12
        for h in ((0, 2) if bass in ("pluck", "pad") else (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5)):
            add(out, t0 + h * beat, tone(bass_note, beat * (2 if bass == "pad" else 0.6), bass), 0.22)

        if style["lead"] and bar % 4 in (1, 2, 3):
            step = beat / 2
            for k, degree in enumerate(motif):
                if rng.random() < 0.25:
                    continue
                d = int(degree + prog[bar % len(prog)])
                midi = root + 12 + scale[d % 7] + 12 * (d // 7)
                add(out, t0 + k * step, tone(midi, step * 1.8, style["lead"]), 0.16)

        drums = style["drums"]
        if drums:
            for b in range(4):
                tb = t0 + b * beat
                if drums in ("pop", "edm", "rock") or b in (0, 2):
                    if drums != "lofi" or b in (0, 2):
                        add(out, tb, drum("kick", rng), 0.6)
                if b in (1, 3):
                    add(out, tb, drum("snare", rng), 0.35 if drums != "lofi" else 0.2)
                subdivisions = 4 if drums in ("edm", "rock") else 2
                for s in range(subdivisions):
                    add(out, tb + s * beat / subdivisions, drum("hat", rng), 0.25)
                if drums == "rock" and b in (1, 3):
                    add(out, tb + beat / 2, drum("kick", rng), 0.5)

    if style["drums"] == "lofi":
        out += rng.normal(0, 0.004, len(out))  # vinyl hiss

    out = np.convolve(out, np.ones(3) / 3, mode="same")  # gentle low-pass
    fade = int(2 * SR)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out / (np.max(np.abs(out)) + 1e-9) * 0.9


def write_mp3(samples: np.ndarray, path: Path) -> None:
    stereo = np.stack([samples, np.roll(samples, int(0.012 * SR))], axis=1)
    pcm = (stereo * 32767).astype(np.int16)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "track.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(SR)
            wav.writeframes(pcm.tobytes())
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
             "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", str(SR), "-b:a", "160k", str(path)],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument("--root", type=Path, default=MUSIC_ROOT)
    args = parser.parse_args()

    manifest = args.root / "SOURCES.csv"
    rows = []
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as file:
            rows = [row for row in csv.DictReader(file) if row["artist"] != "MoodTune generator"]

    for folder, style in STYLES.items():
        directory = args.root / folder
        directory.mkdir(parents=True, exist_ok=True)
        mood, _, variant = folder.partition("/")
        for index in range(1, args.count + 1):
            rng = np.random.default_rng(zlib.crc32(f"{folder}:{index}".encode()))
            filename = f"demo_{mood}{'_' + variant if variant else ''}_{index:02d}.mp3"
            write_mp3(render(style, args.seconds, rng), directory / filename)
            rows.append({
                "title": f"{style['name']} demo {index}", "artist": "MoodTune generator",
                "url": "tools/generate_demo_music.py", "license": "Original work (CC0)",
                "mood": mood, "mode": variant or "mirror", "filename": f"{folder}/{filename}",
            })
            print(f"wrote {folder}/{filename}")

    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["title", "artist", "url", "license", "mood", "mode", "filename"])
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
