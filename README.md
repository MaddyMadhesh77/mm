# MoodTune-AI

A real-time facial emotion-driven adaptive music player with dynamic visual ambience.
A webcam reads your expression, a FER-2013 CNN classifies it into one of five moods
(Happy, Sad, Angry, Neutral, Surprised), and the app cross-fades to matching music
while the background colour shifts to suit the mood.

## Setup

Python 3.10 or 3.11, a webcam, and FFmpeg (only needed to generate demo music).

```bash
python -m venv .venv            # or: uv venv --python 3.11
source .venv/bin/activate       # fish: source .venv/bin/activate.fish
pip install -r requirements.txt # uv venvs: uv pip install -r requirements.txt
```

Fill the music library. To generate original placeholder tracks (CC0):

```bash
python -m tools.generate_demo_music
python -m moodtune.music_library      # validates folders + SOURCES.csv
```

Replace them with real royalty-free tracks from Pixabay Music or the Free Music Archive
when you can (see `music/README.md`), and record every track in `music/SOURCES.csv`.

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501. Music plays through the computer running the app.

Sidebar controls: Automatic/Manual mood, Mirror/Regulate mode, pause, volume,
smoothing window, confidence threshold and dwell time, a raw (no smoothing) toggle, and
session logging for evaluation.

Quick camera check without the UI: `python phase2_smoke_test.py` (press `q` to quit).

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `moodtune/capture.py` | Threaded OpenCV webcam capture |
| `moodtune/emotion.py` | FER wrapper, largest face, 7 → 5 mood mapping |
| `moodtune/perception.py` | Throttles inference to `INFERENCE_FPS` |
| `moodtune/decision.py` | Moving window, confidence threshold, dwell time, manual override |
| `moodtune/audio.py` | Two-channel pygame cross-fade, Mirror/Regulate, Neutral fallback |
| `moodtune/visual.py` | Appendix B colour themes as animated CSS |
| `moodtune/dashboard.py` | Background loop tying everything together |
| `moodtune/logger.py` | Per-frame CSV logging for evaluation |
| `moodtune/config.py` | All tunable values |
| `eval/analyze.py` | Confusion matrix, latency, stability, parameter sweep, report tables |
| `eval/system_info.py` | Prints Table 4.1 values |
| `tools/generate_demo_music.py` | Synthesizes CC0 demo tracks |
| `tests/` | Unit and functional tests (`python -m pytest -q`) |

Testing and evaluation steps are in [docs/TESTING.md](docs/TESTING.md).

## Privacy

Frames are processed in memory only and are never saved or uploaded. Logs contain
only mood probabilities and timings.
