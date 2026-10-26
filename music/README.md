# Music library

Each mood folder should contain 5-8 normalized, royalty-free tracks:

- `happy/`
- `sad/`
- `angry/`
- `neutral/`
- `surprised/`

Regulate variants belong in `sad/regulate/`, `angry/regulate/`, and
`surprised/regulate/`. Use `SOURCES.csv` to record the title, artist, source
URL, license, mood, mode, and local filename for every track.

## Normalization

Normalize downloaded files to a consistent loudness before adding them to the
library. With FFmpeg, a typical command is:

```text
ffmpeg -i input.mp3 -af loudnorm=I=-16:TP=-1.5:LRA=11 output.mp3
```

Only add the normalized output to the mood folder. Keep the original source
file outside the runtime library if it is needed for provenance.