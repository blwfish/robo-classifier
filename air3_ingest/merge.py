"""
Discovery, grouping, and ffmpeg merge logic for Air3 ingest.

Pipeline:
  1. discover_clips()  -- find every *.MP4 under source_dir, pair with its
     .SRT sidecar (if present), parse telemetry, sort by wall-clock start.
  2. group_clips()     -- collapse clips into recording sessions using a
     single gap threshold: gap < threshold => same output file (whether
     that gap is ~50ms from a file-size-triggered chunk split, or minutes
     from a stills break mid-flight); gap >= threshold => new output file.
  3. merge_group()     -- ffmpeg concat (stream copy, no re-encode) + merged
     telemetry muxed in as an embedded mov_text subtitle track + standard
     creation_time/location container metadata.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from srt_parser import SrtCue, SrtParseError, format_cue_block, parse_srt

FFPROBE = "ffprobe"
FFMPEG = "ffmpeg"

GAP_THRESHOLD_DEFAULT_S = 300.0

FILENAME_TS_RE = re.compile(r"DJI_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_")


@dataclass
class ClipProbe:
    duration_s: float
    codec_name: str
    width: int
    height: int
    r_frame_rate: str
    creation_time: str | None  # container tag, e.g. "2026-06-29T14:27:07.000000Z"


@dataclass
class Clip:
    mp4_path: Path
    srt_path: Path | None
    cues: list[SrtCue]
    srt_error: str | None
    probe: ClipProbe
    start_dt: datetime
    end_dt: datetime
    start_is_estimated: bool  # True when there was no usable SRT and we fell
                               # back to filename timestamp + ffprobe duration


@dataclass
class ClipGroup:
    clips: list[Clip]
    gap_to_next_s: float | None = None


@dataclass
class MergeResult:
    ok: bool
    output_path: Path | None
    source_files: list[str]
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def _run(cmd: list[str]) -> str:
    # errors="replace" (not the subprocess default "strict"): ffmpeg/ffprobe
    # stderr is diagnostic text we only ever log, never parse, so a stray
    # non-UTF-8 byte should degrade gracefully rather than raising
    # UnicodeDecodeError -- which is not a RuntimeError and would bypass the
    # `except RuntimeError` callers use to turn a failed run into a clean,
    # per-clip/per-group error instead of an unhandled crash.
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout


def probe_clip(mp4_path: Path) -> ClipProbe:
    out = _run([
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate",
        "-show_entries", "format=duration",
        "-show_entries", "format_tags=creation_time",
        "-of", "json",
        str(mp4_path),
    ])
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"{mp4_path.name}: ffprobe found no video stream (corrupt or non-video file?)")
    stream = streams[0]
    fmt = data.get("format", {})
    tags = fmt.get("tags", {})
    if "duration" not in fmt:
        raise RuntimeError(f"{mp4_path.name}: ffprobe reported no container duration")
    return ClipProbe(
        duration_s=float(fmt["duration"]),
        codec_name=stream["codec_name"],
        width=int(stream["width"]),
        height=int(stream["height"]),
        r_frame_rate=stream["r_frame_rate"],
        creation_time=tags.get("creation_time"),
    )


def _filename_start(mp4_path: Path) -> datetime:
    m = FILENAME_TS_RE.search(mp4_path.name)
    if not m:
        raise ValueError(f"can't parse start time from filename: {mp4_path.name}")
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    return datetime(y, mo, d, h, mi, s)


def discover_clips(source_dir: Path) -> tuple[list[Clip], list[str]]:
    """Returns (clips sorted by wall-clock start time, list of warning strings)."""
    warnings: list[str] = []
    mp4_paths = sorted(
        p for p in Path(source_dir).rglob("*")
        if p.is_file() and p.suffix.lower() == ".mp4"
    )
    clips: list[Clip] = []
    for mp4_path in mp4_paths:
        probe = probe_clip(mp4_path)

        srt_path = mp4_path.with_suffix(".SRT")
        if not srt_path.exists():
            srt_path = mp4_path.with_suffix(".srt")
        srt_exists = srt_path.exists()

        cues: list[SrtCue] = []
        srt_error = None
        if srt_exists:
            try:
                cues = parse_srt(srt_path)
            except SrtParseError as e:
                srt_error = str(e)
                warnings.append(f"{mp4_path.name}: failed to parse SRT telemetry: {e}")
        else:
            srt_error = "no .SRT sidecar found"
            warnings.append(f"{mp4_path.name}: no .SRT sidecar found; grouping and "
                             f"metadata for this clip fall back to filename timestamp only")

        start_is_estimated = not bool(cues)
        if cues:
            start_dt = cues[0].wall_clock
            end_dt = cues[-1].wall_clock
        else:
            start_dt = _filename_start(mp4_path)
            end_dt = start_dt + timedelta(seconds=probe.duration_s)

        clips.append(Clip(
            mp4_path=mp4_path,
            srt_path=srt_path if srt_exists else None,
            cues=cues,
            srt_error=srt_error,
            probe=probe,
            start_dt=start_dt,
            end_dt=end_dt,
            start_is_estimated=start_is_estimated,
        ))

    clips.sort(key=lambda c: c.start_dt)
    return clips, warnings


def group_clips(clips: list[Clip], gap_threshold_s: float = GAP_THRESHOLD_DEFAULT_S) -> list[ClipGroup]:
    if not clips:
        return []
    buckets: list[list[Clip]] = [[clips[0]]]
    for prev, cur in zip(clips, clips[1:]):
        gap = (cur.start_dt - prev.end_dt).total_seconds()
        if gap < gap_threshold_s:
            buckets[-1].append(cur)
        else:
            buckets.append([cur])
    groups = [ClipGroup(clips=b) for b in buckets]
    for i in range(len(groups) - 1):
        gap = (groups[i + 1].clips[0].start_dt - groups[i].clips[-1].end_dt).total_seconds()
        groups[i].gap_to_next_s = gap
    return groups


def _first_gps_cue(clips: list[Clip]) -> SrtCue | None:
    for c in clips:
        for cue in c.cues:
            if cue.has_gps:
                return cue
    return None


def _last_gps_cue(clips: list[Clip]) -> SrtCue | None:
    for c in reversed(clips):
        for cue in reversed(c.cues):
            if cue.has_gps:
                return cue
    return None


def group_summary(group: ClipGroup) -> dict:
    clips = group.clips
    first, last = clips[0], clips[-1]
    total_duration = sum(c.probe.duration_s for c in clips)
    total_size = sum(c.mp4_path.stat().st_size for c in clips)
    start_cue = _first_gps_cue(clips)
    end_cue = _last_gps_cue(clips)
    start_loc = (start_cue.latitude, start_cue.longitude) if start_cue else None
    end_loc = (end_cue.latitude, end_cue.longitude) if end_cue else None
    return {
        "clip_count": len(clips),
        "clip_names": [c.mp4_path.name for c in clips],
        "start_dt": first.start_dt.isoformat(),
        "end_dt": last.end_dt.isoformat(),
        "total_duration_s": total_duration,
        "total_size_bytes": total_size,
        "start_location": start_loc,
        "end_location": end_loc,
        "start_is_estimated": first.start_is_estimated,
        "missing_srt": [c.mp4_path.name for c in clips if c.srt_error],
        "gap_to_next_s": group.gap_to_next_s,
    }


def _check_uniform_stream(clips: list[Clip]) -> str | None:
    first = clips[0].probe
    for c in clips[1:]:
        key = (c.probe.codec_name, c.probe.width, c.probe.height, c.probe.r_frame_rate)
        first_key = (first.codec_name, first.width, first.height, first.r_frame_rate)
        if key != first_key:
            return (
                f"video stream mismatch within group: {clips[0].mp4_path.name} is "
                f"{first.codec_name} {first.width}x{first.height}@{first.r_frame_rate}, "
                f"but {c.mp4_path.name} is {c.probe.codec_name} {c.probe.width}x{c.probe.height}"
                f"@{c.probe.r_frame_rate}. Refusing to concat mismatched streams."
            )
    return None


def _iso6709(lat: float, lon: float, alt: float) -> str:
    return f"{lat:+.4f}{lon:+.4f}{alt:+.3f}/"


def _quote_concat_path(p: Path) -> str:
    return "file '" + str(p.resolve()).replace("'", "'\\''") + "'"


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    i = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def merge_group(group: ClipGroup, dest_dir: Path) -> MergeResult:
    clips = group.clips
    source_names = [c.mp4_path.name for c in clips]
    warnings = [f"{c.mp4_path.name}: {c.srt_error}" for c in clips if c.srt_error]

    mismatch = _check_uniform_stream(clips)
    if mismatch:
        return MergeResult(ok=False, output_path=None, source_files=source_names, error=mismatch)

    first = clips[0]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = _next_available_path(dest_dir / f"AIR3_{first.start_dt.strftime('%Y%m%d_%H%M%S')}.mp4")

    has_telemetry = any(c.cues for c in clips)

    with tempfile.TemporaryDirectory(prefix="air3_ingest_") as tmp:
        tmp_path = Path(tmp)
        concat_list = tmp_path / "concat.txt"
        concat_list.write_text("\n".join(_quote_concat_path(c.mp4_path) for c in clips) + "\n")

        cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list)]

        if has_telemetry:
            merged_srt = tmp_path / "merged.srt"
            blocks = []
            cumulative_offset = 0.0
            cue_index = 1
            for c in clips:
                for cue in c.cues:
                    blocks.append(format_cue_block(
                        cue_index,
                        cue.cue_start_s + cumulative_offset,
                        cue.cue_end_s + cumulative_offset,
                        cue,
                    ))
                    cue_index += 1
                cumulative_offset += c.probe.duration_s
            merged_srt.write_text("\n".join(blocks))
            cmd += ["-i", str(merged_srt)]

        cmd += ["-map", "0:v:0"]
        if has_telemetry:
            cmd += ["-map", "1:0", "-c:v", "copy", "-c:s", "mov_text",
                    "-metadata:s:s:0", "handler_name=Air3 Telemetry"]
        else:
            cmd += ["-c:v", "copy"]

        if first.probe.creation_time:
            cmd += ["-metadata", f"creation_time={first.probe.creation_time}"]
        gps_cue = _first_gps_cue(clips)
        if gps_cue:
            cmd += ["-metadata", f"location={_iso6709(gps_cue.latitude, gps_cue.longitude, gps_cue.abs_alt)}"]

        cmd += [str(out_path)]

        try:
            _run(cmd)
        except RuntimeError as e:
            return MergeResult(ok=False, output_path=None, source_files=source_names,
                                error=str(e), warnings=warnings)

    return MergeResult(ok=True, output_path=out_path, source_files=source_names, warnings=warnings)
