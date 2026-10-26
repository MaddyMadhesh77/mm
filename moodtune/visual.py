"""Mood-driven CSS themes for the dashboard (report Appendix B)."""

from __future__ import annotations

from .config import MOODS

THEMES = {
    # Warm yellow, bright and lively gradient.
    "happy": {"start": "#ffe066", "end": "#ffb703", "accent": "#9a5b00", "speed": "12s"},
    # Soft blue, slow and gentle gradient.
    "sad": {"start": "#a8c5e8", "end": "#5b7db1", "accent": "#1d3557", "speed": "34s"},
    # Deep red with a steady glow.
    "angry": {"start": "#9d0208", "end": "#d00000", "accent": "#370617", "speed": "20s"},
    # Soft grey-teal, minimal movement.
    "neutral": {"start": "#b8c9c6", "end": "#8fa9a6", "accent": "#2f4f4f", "speed": "40s"},
    # Vivid purple-magenta with a quick shimmer.
    "surprised": {"start": "#9d4edd", "end": "#f72585", "accent": "#3c096c", "speed": "8s"},
}

# Regulate mode replaces themes that should calm rather than mirror the mood.
REGULATE_THEMES = {
    # Cool teal instead of deep red.
    "angry": {"start": "#2a9d8f", "end": "#76c7c0", "accent": "#0b3c49", "speed": "30s"},
}


def theme_for(mood: str, mode: str = "mirror") -> dict[str, str]:
    """Return the configured theme, falling back to neutral."""
    if mode == "regulate" and mood in REGULATE_THEMES:
        return REGULATE_THEMES[mood]
    return THEMES.get(mood, THEMES["neutral"])


def css_for_mood(mood: str, mode: str = "mirror") -> str:
    """Build the animated page background and accent styles."""
    theme = theme_for(mood, mode)
    return f"""
    <style>
    :root {{
        --mood-start: {theme['start']};
        --mood-end: {theme['end']};
        --mood-accent: {theme['accent']};
    }}
    .stApp {{
        background: linear-gradient(120deg, var(--mood-start), var(--mood-end), var(--mood-start));
        background-size: 400% 400%;
        animation: mood-gradient {theme['speed']} ease infinite;
        transition: background 1.5s ease;
    }}
    .mood-panel {{
        background: color-mix(in srgb, white 82%, var(--mood-end));
        color: #111827;
        border-left: 6px solid var(--mood-accent);
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, .12);
    }}
    .mood-panel h2 {{ margin: 0; color: var(--mood-accent); }}
    .mood-panel p {{ margin: .3rem 0 0; }}
    @keyframes mood-gradient {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    </style>
    """


def validate_themes() -> list[str]:
    """Return missing-theme errors for lightweight tests."""
    return [mood for mood in MOODS if mood not in THEMES]
