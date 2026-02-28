#!/usr/bin/env python3

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run_capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlLogger:
    def __init__(self, log_path: Path | None):
        self._path = log_path
        self._handle = None

    def open(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")

    def log(self, event: str, **fields: object) -> None:
        if self._handle is None:
            return
        payload = {"ts": _utc_now_iso(), "event": event, **fields}
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None


def _ffprobe_duration_seconds(input_path: Path) -> int:
    out = _run_capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(input_path),
        ]
    )
    duration = float(out)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid duration from ffprobe: {out!r}")
    return int(math.ceil(duration))


def _safe_output_relpath(input_path: Path) -> Path:
    parts = list(input_path.parts)
    if input_path.is_absolute() and parts:
        # Preserve path but avoid writing to an absolute location by stripping the leading "/".
        parts = parts[1:]

    safe_parts: list[str] = []
    for part in parts:
        if part in ("", "."):
            continue
        if part == "..":
            continue
        safe_parts.append(part)

    if not safe_parts:
        return Path(input_path.name)
    return Path(*safe_parts)


def _bytes_from_threshold(
    threshold_mb: float | None, threshold_mib: float | None
) -> int:
    if threshold_mib is not None:
        if threshold_mib <= 0:
            raise ValueError("--threshold-mib must be > 0")
        return int(threshold_mib * 1024 * 1024)
    if threshold_mb is None:
        raise ValueError("missing threshold")
    if threshold_mb is not None:
        if threshold_mb <= 0:
            raise ValueError("--threshold-mb must be > 0")
        return int(threshold_mb * 1_000_000)
    raise ValueError("missing threshold")


def _read_list_file(list_path: Path) -> list[str]:
    lines: list[str] = []
    for raw in list_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("audio files larger than"):
            continue
        if line.lower().startswith("found:"):
            continue
        lines.append(line)
    return lines


def _compute_max_bitrate_kbps(
    duration_seconds: int,
    threshold_bytes: int,
    headroom_bytes: int,
) -> int:
    usable_bytes = threshold_bytes - headroom_bytes
    if usable_bytes <= 0:
        return 0
    # bitrate_kbps = floor((bytes * 8 / seconds) / 1000)
    return int((usable_bytes * 8) // (duration_seconds * 1000))


def _progress_prefix(index: int, total: int) -> str:
    remaining = max(total - index, 0)
    return f"[{index}/{total} remaining={remaining}]"


def _encode_aac_m4a(
    input_path: Path,
    output_path: Path,
    bitrate_kbps: int,
    sample_rate: int,
    channels: int,
    dry_run: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
    ]
    cmd += ["-y"]
    cmd += [
        "-i",
        str(input_path),
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        f"{bitrate_kbps}k",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]

    if dry_run:
        print("DRY-RUN:", " ".join(shlex_quote(x) for x in cmd))
        return

    try:
        subprocess.run(cmd, check=True)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def shlex_quote(s: str) -> str:
    # Minimal shell-escaping for human-readable logging.
    if not s or any(c in s for c in " \t\n\"'\\$`!(){}[]*?;|&<>"):
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compress audio files from a list to fit under a size threshold, using the max "
            "average bitrate that stays under the threshold."
        )
    )
    parser.add_argument(
        "--list",
        default="large_audio_files_over_209MB.txt",
        type=Path,
        help="Path to the input list file (one audio file path per line).",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("output"),
        type=Path,
        help="Output directory (will preserve source path under this folder).",
    )
    parser.add_argument(
        "--threshold-mb",
        type=float,
        default=200.0,
        help="Size limit in decimal MB (default: 200).",
    )
    parser.add_argument(
        "--threshold-mib",
        type=float,
        default=None,
        help="Size limit in MiB (binary). If set, overrides --threshold-mb.",
    )
    parser.add_argument(
        "--headroom-mb",
        type=float,
        default=0.0,
        help="Optional headroom in decimal MB to subtract from the threshold when computing bitrate (default: 0).",
    )
    parser.add_argument(
        "--min-bitrate-kbps",
        type=int,
        default=96,
        help="If computed max bitrate is below this, skip conversion (default: 96).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Output sample rate (default: 44100).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Output channels (default: 2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work without encoding.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit on number of files processed (0 = no limit).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max re-encode attempts if output is still too large (default: 2).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to append structured JSONL logs.",
    )

    args = parser.parse_args()
    logger = JsonlLogger(args.log_file)
    try:
        try:
            logger.open()
        except OSError as e:
            print(f"Unable to open log file {args.log_file}: {e}", file=sys.stderr)
            return 2

        logger.log(
            "run_start",
            argv=sys.argv[1:],
            list=str(args.list),
            output_dir=str(args.output_dir),
            threshold_mb=args.threshold_mb,
            threshold_mib=args.threshold_mib,
            headroom_mb=args.headroom_mb,
            min_bitrate_kbps=args.min_bitrate_kbps,
            sample_rate=args.sample_rate,
            channels=args.channels,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            max_files=args.max_files,
            max_retries=args.max_retries,
        )

        for tool in ("ffprobe", "ffmpeg"):
            if shutil.which(tool) is None:
                print(f"Required tool not found in PATH: {tool}", file=sys.stderr)
                logger.log("validation_error", reason="missing_tool", tool=tool)
                return 2
        if args.min_bitrate_kbps <= 0:
            print("--min-bitrate-kbps must be > 0", file=sys.stderr)
            logger.log(
                "validation_error",
                reason="invalid_argument",
                argument="min_bitrate_kbps",
                value=args.min_bitrate_kbps,
            )
            return 2
        if args.sample_rate <= 0:
            print("--sample-rate must be > 0", file=sys.stderr)
            logger.log(
                "validation_error",
                reason="invalid_argument",
                argument="sample_rate",
                value=args.sample_rate,
            )
            return 2
        if args.channels <= 0:
            print("--channels must be > 0", file=sys.stderr)
            logger.log(
                "validation_error",
                reason="invalid_argument",
                argument="channels",
                value=args.channels,
            )
            return 2
        if args.max_files < 0:
            print("--max-files must be >= 0", file=sys.stderr)
            logger.log(
                "validation_error",
                reason="invalid_argument",
                argument="max_files",
                value=args.max_files,
            )
            return 2
        if args.max_retries < 0:
            print("--max-retries must be >= 0", file=sys.stderr)
            logger.log(
                "validation_error",
                reason="invalid_argument",
                argument="max_retries",
                value=args.max_retries,
            )
            return 2

        threshold_bytes = _bytes_from_threshold(args.threshold_mb, args.threshold_mib)
        headroom_bytes = int(max(args.headroom_mb, 0.0) * 1_000_000)

        if not args.list.exists():
            print(f"List file not found: {args.list}", file=sys.stderr)
            logger.log("validation_error", reason="list_not_found", list=str(args.list))
            return 2

        inputs = _read_list_file(args.list)
        if args.max_files and args.max_files > 0:
            inputs = inputs[: args.max_files]

        if not inputs:
            print(f"No inputs found in {args.list}")
            logger.log("run_summary", encoded=0, failed=0, total_inputs=0, exit_code=0)
            return 0

        args.output_dir.mkdir(parents=True, exist_ok=True)

        skipped_missing = 0
        skipped_under_threshold = 0
        skipped_bitrate_too_low = 0
        skipped_exists = 0
        skipped_cannot_fit = 0
        encoded = 0
        failed = 0
        total_inputs = len(inputs)

        for idx, input_str in enumerate(inputs, start=1):
            progress = _progress_prefix(idx, total_inputs)
            listed_path = Path(os.path.expanduser(input_str))
            input_path = listed_path
            if not input_path.is_absolute():
                input_path = (args.list.parent / input_path).resolve()

            if not input_path.exists():
                print(
                    f"{progress} SKIP missing: {input_path} (listed as: {listed_path})"
                )
                logger.log(
                    "file_skipped",
                    index=idx,
                    total=total_inputs,
                    reason="missing",
                    listed_path=str(listed_path),
                    input_path=str(input_path),
                )
                skipped_missing += 1
                continue

            input_size = input_path.stat().st_size
            if input_size <= threshold_bytes:
                print(f"{progress} SKIP already <= threshold: {input_path}")
                logger.log(
                    "file_skipped",
                    index=idx,
                    total=total_inputs,
                    reason="already_under_threshold",
                    input_path=str(input_path),
                    input_size=input_size,
                    threshold_bytes=threshold_bytes,
                )
                skipped_under_threshold += 1
                continue

            try:
                duration_seconds = _ffprobe_duration_seconds(input_path)
            except Exception as e:
                print(
                    f"{progress} FAIL ffprobe duration: {input_path} ({e})",
                    file=sys.stderr,
                )
                logger.log(
                    "file_failed",
                    index=idx,
                    total=total_inputs,
                    stage="ffprobe",
                    input_path=str(input_path),
                    error=str(e),
                )
                failed += 1
                continue

            max_kbps = _compute_max_bitrate_kbps(
                duration_seconds, threshold_bytes, headroom_bytes
            )
            if max_kbps < args.min_bitrate_kbps:
                print(
                    f"{progress} SKIP bitrate<{args.min_bitrate_kbps}k: {input_path} "
                    f"(duration={duration_seconds}s -> max={max_kbps}k)"
                )
                logger.log(
                    "file_skipped",
                    index=idx,
                    total=total_inputs,
                    reason="computed_bitrate_too_low",
                    input_path=str(input_path),
                    duration_seconds=duration_seconds,
                    computed_kbps=max_kbps,
                    min_bitrate_kbps=args.min_bitrate_kbps,
                )
                skipped_bitrate_too_low += 1
                continue

            out_rel = _safe_output_relpath(listed_path).with_suffix(".m4a")
            output_path = args.output_dir / out_rel

            if output_path.exists() and not args.overwrite:
                print(f"{progress} SKIP exists: {output_path}")
                logger.log(
                    "file_skipped",
                    index=idx,
                    total=total_inputs,
                    reason="output_exists",
                    output_path=str(output_path),
                )
                skipped_exists += 1
                continue

            bitrate = max_kbps
            success = False
            count_as_failed = True
            for attempt in range(args.max_retries + 1):
                try:
                    print(
                        f"{progress} encode attempt {attempt + 1}/{args.max_retries + 1}: "
                        f"{input_path.name} -> {output_path} @ {bitrate}k"
                    )
                    logger.log(
                        "encode_attempt",
                        index=idx,
                        total=total_inputs,
                        attempt=attempt + 1,
                        max_attempts=args.max_retries + 1,
                        input_path=str(input_path),
                        output_path=str(output_path),
                        bitrate_kbps=bitrate,
                    )
                    _encode_aac_m4a(
                        input_path=input_path,
                        output_path=output_path,
                        bitrate_kbps=bitrate,
                        sample_rate=args.sample_rate,
                        channels=args.channels,
                        dry_run=args.dry_run,
                    )
                    if args.dry_run:
                        success = True
                        logger.log(
                            "file_encoded",
                            index=idx,
                            total=total_inputs,
                            input_path=str(input_path),
                            output_path=str(output_path),
                            dry_run=True,
                            bitrate_kbps=bitrate,
                        )
                        break

                    out_size = output_path.stat().st_size
                    if out_size <= threshold_bytes:
                        print(
                            f"{progress} OK size={out_size} bytes (<= {threshold_bytes})"
                        )
                        logger.log(
                            "file_encoded",
                            index=idx,
                            total=total_inputs,
                            input_path=str(input_path),
                            output_path=str(output_path),
                            output_size=out_size,
                            threshold_bytes=threshold_bytes,
                            bitrate_kbps=bitrate,
                            dry_run=False,
                        )
                        success = True
                        break

                    # Too big: scale bitrate down proportionally and retry.
                    scaled = int((bitrate * threshold_bytes) // out_size) - 1
                    if scaled < args.min_bitrate_kbps:
                        print(
                            f"{progress} SKIP cannot fit under threshold without going <{args.min_bitrate_kbps}k "
                            f"(last size={out_size} bytes, next={scaled}k). Deleting output."
                        )
                        logger.log(
                            "file_skipped",
                            index=idx,
                            total=total_inputs,
                            reason="cannot_fit_threshold",
                            input_path=str(input_path),
                            output_path=str(output_path),
                            output_size=out_size,
                            threshold_bytes=threshold_bytes,
                            next_bitrate_kbps=scaled,
                            min_bitrate_kbps=args.min_bitrate_kbps,
                        )
                        output_path.unlink(missing_ok=True)
                        skipped_cannot_fit += 1
                        count_as_failed = False
                        break

                    print(
                        f"{progress} retry: output too large (size={out_size}); next bitrate={scaled}k"
                    )
                    logger.log(
                        "encode_retry",
                        index=idx,
                        total=total_inputs,
                        input_path=str(input_path),
                        output_path=str(output_path),
                        output_size=out_size,
                        threshold_bytes=threshold_bytes,
                        next_bitrate_kbps=scaled,
                    )
                    bitrate = scaled
                except subprocess.CalledProcessError as e:
                    print(
                        f"{progress} FAIL ffmpeg: {input_path} ({e})",
                        file=sys.stderr,
                    )
                    logger.log(
                        "file_failed",
                        index=idx,
                        total=total_inputs,
                        stage="ffmpeg",
                        input_path=str(input_path),
                        output_path=str(output_path),
                        error=str(e),
                    )
                    output_path.unlink(missing_ok=True)
                    break
                except Exception as e:
                    print(
                        f"{progress} FAIL: {input_path} ({e})",
                        file=sys.stderr,
                    )
                    logger.log(
                        "file_failed",
                        index=idx,
                        total=total_inputs,
                        stage="unknown",
                        input_path=str(input_path),
                        output_path=str(output_path),
                        error=str(e),
                    )
                    output_path.unlink(missing_ok=True)
                    break

            if success:
                encoded += 1
            elif count_as_failed:
                failed += 1

        print(
            "Done:",
            f"encoded={encoded}",
            f"skipped_missing={skipped_missing}",
            f"skipped_under_threshold={skipped_under_threshold}",
            f"skipped_bitrate_too_low={skipped_bitrate_too_low}",
            f"skipped_exists={skipped_exists}",
            f"skipped_cannot_fit={skipped_cannot_fit}",
            f"failed={failed}",
        )
        exit_code = 0 if failed == 0 else 1
        logger.log(
            "run_summary",
            total_inputs=total_inputs,
            encoded=encoded,
            skipped_missing=skipped_missing,
            skipped_under_threshold=skipped_under_threshold,
            skipped_bitrate_too_low=skipped_bitrate_too_low,
            skipped_exists=skipped_exists,
            skipped_cannot_fit=skipped_cannot_fit,
            failed=failed,
            exit_code=exit_code,
        )
        return exit_code
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
