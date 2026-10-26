"""Validate the Phase 1 music library and its licensing manifest."""

import csv
from pathlib import Path

from .config import MOODS, MUSIC_ROOT

REQUIRED_COLUMNS = {"title", "artist", "url", "license", "mood", "mode", "filename"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
REGULATE_MOODS = {"sad", "angry", "surprised"}


def validate_library(root: Path = MUSIC_ROOT, minimum_tracks: int = 5) -> list[str]:
    """Return validation errors for the folders and source manifest."""
    errors: list[str] = []
    manifest_path = root / "SOURCES.csv"
    if not manifest_path.is_file():
        return [f"Missing manifest: {manifest_path}"]

    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        columns = set(reader.fieldnames or ())
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            errors.append(f"Manifest missing columns: {', '.join(sorted(missing_columns))}")
        rows = list(reader)

    for mood in MOODS:
        mood_dir = root / mood
        if not mood_dir.is_dir():
            errors.append(f"Missing mood directory: {mood_dir}")
            continue

        tracks = [path for path in mood_dir.iterdir() if path.suffix.lower() in AUDIO_SUFFIXES]
        if len(tracks) < minimum_tracks:
            errors.append(f"{mood}: found {len(tracks)} tracks, need at least {minimum_tracks}")

        if mood in REGULATE_MOODS and not (mood_dir / "regulate").is_dir():
            errors.append(f"Missing regulate directory: {mood_dir / 'regulate'}")

    for index, row in enumerate(rows, start=2):
        if not all(row.get(column, "").strip() for column in REQUIRED_COLUMNS):
            errors.append(f"Manifest row {index}: all metadata fields are required")
        if row.get("mood", "").strip().lower() not in MOODS:
            errors.append(f"Manifest row {index}: unknown mood")
        if row.get("mode", "").strip().lower() not in {"mirror", "regulate"}:
            errors.append(f"Manifest row {index}: mode must be mirror or regulate")

    return errors


if __name__ == "__main__":
    library_errors = validate_library()
    if library_errors:
        print("Music library is incomplete:")
        print("\n".join(f"- {error}" for error in library_errors))
        raise SystemExit(1)
    print("Music library is valid.")
