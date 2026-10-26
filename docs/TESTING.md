# Testing and evaluation guide

## 1. Automated tests

```bash
python -m pytest -q
```

`tests/test_functional.py` automates the Table 4.2 cases that do not need a camera
(T2–T8), plus Mirror/Regulate switching. The phase tests cover each module separately.

## 2. Manual functional tests (Table 4.2)

Run `streamlit run app.py`, perform each case, record Pass/Fail and take a screenshot
(save under `docs/screenshots/`, e.g. `T3.png`).

| Test | How to perform it | Expected | Result |
|---|---|---|---|
| T1 | Start app, face the camera | Live preview with face box and mood | |
| T2 | Leave the frame | "No face detected", music continues | |
| T3 | Neutral → hold a smile ~3 s (dwell set to 0 for the demo, or wait 15 s) | Upbeat track fades in, background turns warm yellow | |
| T4 | Blink / quick grimace while neutral | Mood does not change | |
| T5 | Change mood while music plays | ~2 s cross-fade, no abrupt cut | |
| T6 | Ambiguous half-expression | Confidence below threshold, mood kept | |
| T7 | Sidebar → Manual → pick a mood | Follows chosen mood until Automatic | |
| T8 | Temporarily rename `music/angry/` and trigger Angry | Warning shown, Neutral music plays | |
| T9 | Dim the lights / cover half the face | Keeps last mood, no crash | |

## 3. Evaluation sessions (Chapter 5)

For each volunteer (get consent first):

1. Sidebar → **Record session log** on. A CSV is written to `logs/`.
2. **Posed accuracy:** for each of the five moods, set *Requested expression* to that mood,
   ask the volunteer to show it for ~20 s. Reset to "— none —" between moods.
3. **Spontaneous:** set the requested expression to the mood the video clip is meant to cause.
4. **Stability:** volunteer holds a neutral face for 2 min with smoothing ON, then
   tick *Disable smoothing* and repeat for 2 min.
5. **Performance:** leave it running ~10 min (CPU % and inference time are logged).
6. Turn logging off, then give the 1–5 questionnaire below.

Analyze each log (or concatenate them):

```bash
python -m eval.analyze logs/session-YYYYMMDD-HHMMSS.csv --output eval-results
```

Outputs in `eval-results/`:

- `report_tables.md`: Tables 5.1 and 5.2 ready to paste
- `confusion_matrix.png`, `switches_per_minute.png`: figures
- `parameter_sweep.csv`: offline replay of window/threshold/dwell values; use it to justify the final settings
- `summary.json`: all raw numbers

Print Table 4.1 values (hardware, library size, settings): `python -m eval.system_info`.

## 4. Questionnaire (1 = strongly disagree, 5 = strongly agree)

1. The detected mood matched how I was expressing myself.
2. The music suited the detected mood.
3. The background colours suited the mood.
4. Mood changes felt smooth, not jarring.
5. I preferred Regulate mode over Mirror mode when angry or sad.
6. I would use a player like this.
