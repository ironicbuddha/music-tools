#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".alac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


def _is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def _collect_audio_files(
    input_dir: Path, recursive: bool
) -> tuple[list[Path], list[Path]]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    all_files = [p for p in iterator if p.is_file()]
    audio_files = [p for p in all_files if _is_audio_file(p)]
    skipped_files = [p for p in all_files if not _is_audio_file(p)]
    audio_files.sort(key=lambda p: p.name.lower())
    skipped_files.sort(key=lambda p: p.name.lower())
    return audio_files, skipped_files


def _escape_ffconcat_path(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def _write_concat_list(audio_files: list[Path]) -> Path:
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix="ffconcat_",
        delete=False,
        encoding="utf-8",
    )
    with temp_file as handle:
        for file_path in audio_files:
            handle.write(f"file '{_escape_ffconcat_path(file_path)}'\n")
    return Path(temp_file.name)


def _run_ffmpeg_concat(
    concat_list_path: Path, output_path: Path, reencode: bool
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
    ]

    if reencode:
        cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]
    else:
        cmd += ["-c", "copy"]

    cmd += [str(output_path)]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate audio files in a folder into one output file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing audio files to concatenate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output audio file path.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directory recursively.",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode to MP3 192k if stream-copy concat fails or inputs differ.",
    )

    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}", file=sys.stderr)
        return 2

    audio_files, skipped_files = _collect_audio_files(input_dir, args.recursive)
    if not audio_files:
        print(f"No audio files found in: {input_dir}", file=sys.stderr)
        return 1

    concat_list_path = _write_concat_list(audio_files)
    try:
        _run_ffmpeg_concat(concat_list_path, output_path, args.reencode)
    except subprocess.CalledProcessError as exc:
        print("ffmpeg concat failed.", file=sys.stderr)
        print(
            "Retry with --reencode if files are not stream-compatible.", file=sys.stderr
        )
        return exc.returncode
    finally:
        os.unlink(concat_list_path)

    print(f"Wrote concatenated output: {output_path}")
    print(
        f"Included {len(audio_files)} audio files; skipped {len(skipped_files)} files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
