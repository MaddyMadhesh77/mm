"""Streamlit entry point for MoodTune.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from moodtune.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DWELL_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    MOODS,
    PROJECT_ROOT,
)
from moodtune.dashboard import DashboardRuntime
from moodtune.visual import css_for_mood

MOOD_ICONS = {"happy": "😄", "sad": "😢", "angry": "😠", "neutral": "😐", "surprised": "😲"}
LOG_DIR = PROJECT_ROOT / "logs"

st.set_page_config(page_title="MoodTune-AI", page_icon="🎵", layout="wide")


@st.cache_resource
def get_runtime() -> DashboardRuntime:
    """Create the camera, model, decision and audio engines once per server."""
    return DashboardRuntime().start()


with st.spinner("Loading camera and emotion model…"):
    runtime = get_runtime()

# ---- Sidebar controls -------------------------------------------------------
st.sidebar.title("🎵 MoodTune-AI")

control = st.sidebar.radio("Mood control", ["Automatic", "Manual"], horizontal=True)
manual = None
if control == "Manual":
    manual = st.sidebar.selectbox("Manual mood", MOODS, format_func=lambda m: f"{MOOD_ICONS[m]} {m.title()}")
runtime.set_manual_mood(manual)

mode = st.sidebar.radio(
    "Music mode",
    ["mirror", "regulate"],
    format_func=lambda m: {"mirror": "Mirror (match mood)", "regulate": "Regulate (calm mood)"}[m],
)
runtime.set_mode(mode)

paused = st.sidebar.toggle("Pause", value=False)
runtime.set_paused(paused)
runtime.set_volume(st.sidebar.slider("Volume", 0.0, 1.0, 0.8, 0.05))

st.sidebar.subheader("Stability settings")
raw_mode = st.sidebar.checkbox("Disable smoothing (raw frame-by-frame)", help="For the smoothing on/off comparison.")
window = st.sidebar.slider("Smoothing window (s)", 0.5, 5.0, DEFAULT_WINDOW_SECONDS, 0.5, disabled=raw_mode)
threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05, disabled=raw_mode)
dwell = st.sidebar.slider("Minimum dwell time (s)", 0.0, 30.0, DEFAULT_DWELL_SECONDS, 1.0, disabled=raw_mode)
if raw_mode:
    runtime.configure(window, 0.0, 0.0, smoothing_enabled=False)
else:
    runtime.configure(window, threshold, dwell, smoothing_enabled=True)

st.sidebar.subheader("Evaluation")
record = st.sidebar.toggle("Record session log (CSV)")
if record and runtime.logger is None:
    LOG_DIR.mkdir(exist_ok=True)
    runtime.set_logging(str(LOG_DIR / f"session-{time.strftime('%Y%m%d-%H%M%S')}.csv"))
elif not record and runtime.logger is not None:
    runtime.set_logging(None)
truth = st.sidebar.selectbox(
    "Requested expression (ground truth)",
    [None, *MOODS],
    format_func=lambda m: "— none —" if m is None else m.title(),
    help="Set this as you ask the volunteer to change expression; it is written to the log.",
)
runtime.ground_truth = truth
if runtime.logger is not None:
    st.sidebar.caption(f"Logging to `{Path(runtime.state.log_path).relative_to(PROJECT_ROOT)}`")

st.sidebar.caption("🔒 Video is processed in memory only — no frames are stored or uploaded.")


# ---- Live view ----------------------------------------------------------------
@st.fragment(run_every=0.2)
def live_view() -> None:
    state = runtime.snapshot()
    st.markdown(css_for_mood(state.mood, mode), unsafe_allow_html=True)

    if state.error:
        st.error(f"Camera/model error: {state.error}")

    video_col, info_col = st.columns([3, 2])
    with video_col:
        if state.frame is not None:
            st.image(state.frame, channels="RGB", width="stretch")
        else:
            st.info("Waiting for camera…")
        if state.paused:
            st.warning("⏸ Paused")
        elif state.no_face:
            st.warning("No face detected — keeping current mood.")

    with info_col:
        override = " (manual)" if runtime.decision.manual_mood else ""
        st.markdown(
            f"""<div class="mood-panel">
            <p>Current mood{override}</p>
            <h2>{MOOD_ICONS[state.mood]} {state.mood.title()}</h2>
            <p>🎵 {state.track or 'No track playing'}</p>
            <p>Mode: {mode.title()}</p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.progress(
            min(1.0, max(0.0, state.confidence)),
            text=f"Smoothed confidence {state.confidence:.2f} (switch needs ≥ {state.threshold:.2f})",
        )
        detected = f"{MOOD_ICONS[state.raw_mood]} {state.raw_mood.title()}" if state.raw_mood else "—"
        st.markdown(f"**Detected now:** {detected} &nbsp;·&nbsp; **Playing:** {MOOD_ICONS[state.mood]} {state.mood.title()}")
        candidate = state.smoothed_mood.title()
        status = {
            "stable": ("info", f"Holding {state.mood.title()}: it is still the strongest mood over the window."),
            "switched": ("success", f"Switched to {state.mood.title()}."),
            "low_confidence": ("warning", f"{candidate} is leading but its average ({state.confidence:.2f}) is below the "
                               f"threshold ({state.threshold:.2f}). Exaggerate the expression or lower the threshold."),
            "dwell": ("warning", f"{candidate} detected: switching in {state.dwell_remaining:.0f} s (minimum dwell time)."),
            "no_face": ("warning", "No face: keeping the current mood."),
            "manual": ("info", "Manual mood override is on."),
            "paused": ("info", "Paused."),
            "waiting": ("info", "Waiting for the first analysed frame…"),
        }
        kind, message = status.get(state.reason, status["waiting"])
        getattr(st, kind)(message)
        if state.warning:
            st.warning(state.warning)

        if state.probabilities:
            st.caption("Per-frame mood probabilities")
            st.bar_chart({m.title(): state.probabilities.get(m, 0.0) for m in MOODS}, horizontal=True, height=200)

        m1, m2, m3 = st.columns(3)
        m1.metric("Loop FPS", f"{state.fps:.0f}")
        m2.metric("Inference", f"{state.inference_ms:.0f} ms" if state.inference_ms else "—")
        m3.metric("CPU", f"{state.cpu_percent:.0f}%" if state.cpu_percent is not None else "—")



st.title("MoodTune-AI")
st.caption("Real-time facial emotion-driven adaptive music player")
live_view()
