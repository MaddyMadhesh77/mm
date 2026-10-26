from moodtune.visual import THEMES, css_for_mood, theme_for, validate_themes


def test_all_moods_have_themes():
    assert validate_themes() == []
    assert set(THEMES) == {"happy", "sad", "angry", "neutral", "surprised"}


def test_css_contains_transition_and_animation():
    css = css_for_mood("surprised")

    assert "transition: background 1.5s ease" in css
    assert "@keyframes mood-gradient" in css
    assert "8s" in css


def test_unknown_mood_uses_neutral_theme():
    assert theme_for("unknown") == THEMES["neutral"]


if __name__ == "__main__":
    test_all_moods_have_themes()
    test_css_contains_transition_and_animation()
    test_unknown_mood_uses_neutral_theme()
    print("Phase 5 visual tests passed.")