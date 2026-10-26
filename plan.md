Phase 0: Setup (Day 1–2)
Environment
Use Python 3.10 or 3.11. Very new Python versions often break fer or TensorFlow.
python -m venv .venv, then pip install fer tensorflow opencv-python pygame streamlit numpy pandas scikit-learn matplotlib.
Known problem: fer imports moviepy.editor, which was removed in moviepy 2.x. Run pip install "moviepy<2.0" or you'll get an ImportError.
Project layout (it matches the modules in report §4.1):

moodtune/
  app.py              # Streamlit dashboard
  capture.py          # OpenCV camera wrapper
  emotion.py          # FER wrapper + 7→5 mapping
  decision.py         # smoothing, threshold, dwell
  audio.py            # pygame cross-fade engine
  visual.py           # mood → CSS theme
  config.py           # all tunable values in one place
  logger.py           # CSV logging for evaluation
  music/{happy,sad,angry,neutral,surprised}/  (+ regulate/ variants)
  eval/               # analysis scripts, logs, plots
Smoke tests: check that cv2.VideoCapture(0) shows frames, FER().detect_emotions(frame) returns probabilities, and pygame plays one MP3.
Exit criterion: all three libraries work on your laptop.

Phase 1: Music library (Day 2–4, can run in parallel)
Download 5–8 royalty-free tracks per mood from Pixabay Music or the Free Music Archive. Follow Appendix B, plus the Regulate variants for Sad, Angry and Surprised.
Normalise loudness (Audacity's "Loudness Normalization" or ffmpeg -af loudnorm) so the cross-fades don't jump in volume.
Keep a music/SOURCES.csv with title, artist, URL, license and mood. This backs the "Music licensing" row in Table 5.3.
Fills in: "[N] royalty-free tracks per mood" (Table 4.1).

Phase 2: Perception layer (Week 1)
capture.py: open the camera at 640×480 and return frames. Run it in a background thread so the UI never waits for the camera.
emotion.py:
Load FER(mtcnn=False) once. The OpenCV Haar detector is faster on a CPU, and you can mention MTCNN as an option.
Pick the largest face from detect_emotions().
Fold the 7 classes into 5:

moods = {
  "Happy": p["happy"], "Sad": p["sad"], "Neutral": p["neutral"],
  "Angry": p["angry"] + p["disgust"],
  "Surprised": p["surprise"] + p["fear"],
}
Return None when no face is found. This covers test cases T2 and T9.
Throttle inference to about 5–10 fps: analyse every Nth frame, or run it on a timer in its own thread, while the preview runs at full rate.
Write a simple OpenCV test window that draws the box, the top mood and the fps.
Exit criterion: live per-frame mood labels in a test window, with fps measured.

Phase 3: Decision layer (Week 1–2)
decision.py, built as a class with update(probs, t) -> (stable_mood, confidence, changed):
Moving window: a deque of (timestamp, probs) covering the last 2–3 s. Average the probabilities and take the argmax.
Confidence threshold: if the averaged maximum is below the threshold (start at 0.5), keep the current mood.
Dwell time: no switch until about 15 s after the last change.
No face: don't add anything to the window, and keep the current mood.
Add a manual override flag that bypasses everything (T7).
Put all values in config.py so you can sweep them later.
Unit tests: feed it synthetic sequences such as steady neutral, a one-frame spike (T4), low confidence (T6) and a genuine change after the dwell time (T3). These tests don't need a camera.
Exit criterion: test-window labels stay stable and no longer flicker.

Phase 4: Output layer, audio (Week 2)
audio.py: for a true cross-fade, use two pygame.mixer.Channels with Sound objects instead of mixer.music, because mixer.music only plays one stream at a time.

old_ch.fadeout(2000)
new_ch.play(pygame.mixer.Sound(path), fade_ms=2000)
Pick a random track in the mood folder, never the one that just played. When a track ends, queue the next track from the same mood.
Mirror/Regulate mode: map each mood to a folder according to the mode.
Fallback: if a folder is empty or a file is missing, show a warning and use the Neutral library (T8).
Keep a volume control, and pause/resume.
Exit criterion: changing the mood in code cross-fades smoothly between folders.

Phase 5: Output layer, visuals and dashboard (Week 2–3)
visual.py: a mood → theme dictionary covering the gradient colours, the accent colour and the animation speed from Appendix B. It produces a CSS string with:
transition: background 1.5s ease for the smooth fade
an animated gradient using @keyframes and background-size: 400%: slow for Sad and Neutral, a quick shimmer for Surprised.
app.py (Streamlit): Streamlit reruns the whole script on every interaction, so:
Keep the camera thread, detector, decision engine and audio engine in @st.cache_resource so they're created only once.
Run the capture → infer → decide → audio loop in a background thread that writes to a shared state object.
Refresh the UI with @st.fragment(run_every=0.2): the preview image, mood label, confidence bar and now-playing track.
Sidebar: Auto/Manual mood, Mirror/Regulate, Pause, volume, and sliders for window, threshold and dwell (useful in the demo).
Show a "No face detected" banner when the face is missing.
Privacy: process frames in memory only and never save them.
Exit criterion: the full working demo. Happy → warm yellow plus upbeat music; Sad → soft blue plus acoustic.

Phase 6: Logging and functional testing (Week 3)
logger.py: one CSV row per analysed frame with timestamp, raw 5-mood probabilities, raw argmax, smoothed mood, stable mood, a switch event flag, inference time in ms and fps.
Add a toggle that turns smoothing on or off. You need it for the comparison in §5.1.
Run test cases T1–T9 and record pass/fail with a screenshot of each for the report.
Phase 7: Evaluation (Week 3–4)
Recruit about 5–10 volunteers and get their consent.

Experiment	Procedure	Metric → report location
Posed accuracy	Each mood held for 20 s, with the requested label written into the log	Confusion matrix and per-class accuracy (sklearn) → §5.1
Spontaneous	Volunteer watches short clips	Accuracy, compared with posed
Performance	10-minute session	Mean fps, CPU % (psutil), inference time
Latency	Time from the cue to change expression until the stable switch	Seconds, split into window, dwell and fade
Stability	Steady expression, smoothing off vs on	Unwanted switches per minute
Parameter sweep	Replay logged raw probabilities through decision.py offline	Pick the final window, threshold and dwell values
Usability	1–5 questionnaire (music fit, colour fit, Mirror vs Regulate preference)	Mean scores
The offline replay is the key trick here. Record raw probabilities once, then test any parameter values without asking volunteers to come back.

Write eval/analyze.py to generate the confusion-matrix heatmap, a switches/min bar chart (on vs off) and a latency table.

Phase 8: Report and demo (Week 4–5)
Fill in every bracketed placeholder:
Abstract
Table 4.1: fps, window, threshold, dwell, N tracks, laptop specs, N participants
§5.1 results
§6.1 conclusion
Add figures: an architecture diagram, dashboard screenshots for each mood, the confusion matrix, the stability chart and the T1–T9 table.
Number the tables in order. The report goes from Table 4.2 to 5.3, so add Tables 5.1 and 5.2 for the accuracy and performance results.
Record a 2–3 minute demo video as a backup in case the webcam fails live. Write a README covering setup and how to run the app.