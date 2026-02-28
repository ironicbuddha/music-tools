# music-tools

Small utilities for working with music/audio files.

## Concatenate audio files

Use `concat_audio_files.py` to combine all audio files in a folder into one output file.

Example:

```bash
./concat_audio_files.py \
  --input-dir "/Volumes/Obama/Music Library/Progressive Trance/Complexity" \
  --output ./complexity_concatenated.mp3
```

If source files are not stream-compatible, run again with `--reencode`.
