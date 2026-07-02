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
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import procs
from srt_parser import SrtCue, SrtParseError, format_cue_block, parse_srt

FFPROBE = "ffprobe"
FFMPEG = "ffmpeg"

# Thin aliases: these used to be private to this module; now shared with
# audio_merge.py via procs.py. Kept under their original names since
# existing tests and call sites below reference them as `_run`/etc.
_run = procs.run
_quote_concat_path = procs.quote_concat_path
_next_available_path = procs.next_available_path

GAP_THRESHOLD_DEFAULT_S = 300.0

FILENAME_TS_RE = re.compile(r"DJI_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_")


@dataclass
class ClipProbe:
    """ffprobe field disposition (stream + format sections), so nothing is
    silently missed from a hand-picked allow-list:

      extracted:  codec_name, width, height, r_frame_rate, pix_fmt, rotation
                  (side_data Display Matrix, falling back to the legacy
                  `tags.rotate` string), duration (prefers the video
                  stream's own reported duration over the container-level
                  `format.duration`, which is sometimes rounded/padded
                  differently -- still an approximation for VFR footage,
                  not a frame-accurate fix), creation_time (format tag),
                  container_location (format tag, GPS fallback when a clip
                  has no SRT sidecar)
      raw-only:   bit_rate, nb_frames -- useful diagnostics for spotting a
                  truncated/corrupt chunk, not currently acted on by any
                  merge decision
      dropped-with-reason: codec_long_name/profile/level/color_*/
                  chroma_location/field_order/refs/nal_length_size/
                  sample_aspect_ratio/display_aspect_ratio/time_base/
                  start_pts/start_time/format_name/probe_score/disposition
                  -- generic container/codec bookkeeping not needed for
                  either the uniform-stream merge-safety check or
                  telemetry; ffmpeg's `-c:v copy` preserves the underlying
                  bitstream regardless of what ffprobe reports here, so
                  there's nothing this tool would act on by capturing them.
    """
    duration_s: float
    codec_name: str
    width: int
    height: int
    r_frame_rate: str
    pix_fmt: str
    rotation: int  # degrees; 0 if the clip carries no rotation side data/tag
    bit_rate: int | None
    nb_frames: int | None
    creation_time: str | None  # container tag, e.g. "2026-06-29T14:27:07.000000Z"
    container_location: str | None  # raw ISO6709-ish string from format tags


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
    filename_suffix: str | None = None  # raw text after the embedded
        # timestamp (e.g. "0001_D"), only ever populated on the filename-
        # estimated-start path, where it's used purely as a deterministic
        # secondary sort key for same-second ties -- not parsed further,
        # since DJI's sequence/lens-suffix filename schema isn't
        # independently verified (see CLAUDE.md Spec Review Rule)


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


def _required_stream_field(stream: dict, key: str, mp4_path: Path):
    value = stream.get(key)
    if value is None:
        raise RuntimeError(
            f"{mp4_path.name}: ffprobe stream is missing '{key}' (corrupt or unusual container?)"
        )
    return value


def _stream_rotation(stream: dict) -> int:
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return int(side_data["rotation"])
    rotate_tag = (stream.get("tags") or {}).get("rotate")
    if rotate_tag is not None:
        return int(rotate_tag)
    return 0


def probe_clip(mp4_path: Path, warnings: list[str] | None = None) -> ClipProbe:
    out = _run([
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_streams", "-show_format",
        "-of", "json",
        str(mp4_path),
    ], warnings)
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"{mp4_path.name}: ffprobe found no video stream (corrupt or non-video file?)")
    stream = streams[0]
    fmt = data.get("format", {})
    tags = fmt.get("tags", {})
    if "duration" not in fmt:
        raise RuntimeError(f"{mp4_path.name}: ffprobe reported no container duration")

    stream_duration = stream.get("duration")
    duration_s = float(stream_duration) if stream_duration not in (None, "N/A") else float(fmt["duration"])

    def _int_or_none(v):
        return int(float(v)) if v not in (None, "N/A") else None

    return ClipProbe(
        duration_s=duration_s,
        codec_name=_required_stream_field(stream, "codec_name", mp4_path),
        width=int(_required_stream_field(stream, "width", mp4_path)),
        height=int(_required_stream_field(stream, "height", mp4_path)),
        r_frame_rate=_required_stream_field(stream, "r_frame_rate", mp4_path),
        pix_fmt=_required_stream_field(stream, "pix_fmt", mp4_path),
        rotation=_stream_rotation(stream),
        bit_rate=_int_or_none(stream.get("bit_rate") or fmt.get("bit_rate")),
        nb_frames=_int_or_none(stream.get("nb_frames")),
        creation_time=tags.get("creation_time"),
        container_location=tags.get("location") or tags.get("com.apple.quicktime.location.ISO6709"),
    )


def _filename_start(mp4_path: Path) -> tuple[datetime, str]:
    """Returns (embedded start timestamp, raw filename remainder after the
    timestamp, e.g. "0001_D")."""
    name = mp4_path.name
    m = FILENAME_TS_RE.search(name)
    if not m:
        raise ValueError(f"can't parse start time from filename: {name}")
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    start_dt = datetime(y, mo, d, h, mi, s)
    suffix = Path(name).stem[m.end():]
    return start_dt, suffix


def discover_clips(source_dir: Path) -> tuple[list[Clip], list[str]]:
    """Returns (clips sorted by wall-clock start time, list of warning strings)."""
    warnings: list[str] = []
    mp4_paths = sorted(
        p for p in Path(source_dir).rglob("*")
        if p.is_file() and p.suffix.lower() == ".mp4"
    )
    clips: list[Clip] = []
    for mp4_path in mp4_paths:
        try:
            probe = probe_clip(mp4_path, warnings)
        except RuntimeError as e:
            # One corrupt/zero-byte clip anywhere on the card shouldn't
            # abort discovery of the entire source directory -- record it
            # as a per-item failure and keep going.
            warnings.append(f"{mp4_path.name}: skipped -- ffprobe failed: {e}")
            continue

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
        filename_suffix = None
        if cues:
            start_dt = cues[0].wall_clock
            end_dt = cues[-1].wall_clock
        else:
            try:
                start_dt, filename_suffix = _filename_start(mp4_path)
            except ValueError as e:
                warnings.append(f"{mp4_path.name}: skipped -- {e}")
                continue
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
            filename_suffix=filename_suffix,
        ))

    clips.sort(key=lambda c: (c.start_dt, c.filename_suffix or ""))
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
        # Group identity for the scan->select->process round trip: full
        # paths, not names. DJI cameras paginate onto multiple *MEDIA
        # folders and can restart file numbering per folder, so two clips
        # in different subfolders can share a filename -- name-only
        # identity would silently alias one selected group onto another.
        "clip_paths": [str(c.mp4_path) for c in clips],
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
    def key(probe: ClipProbe) -> tuple:
        return (probe.codec_name, probe.width, probe.height, probe.r_frame_rate,
                probe.pix_fmt, probe.rotation)

    def describe(probe: ClipProbe) -> str:
        return (f"{probe.codec_name} {probe.width}x{probe.height}@{probe.r_frame_rate} "
                f"{probe.pix_fmt} rotate={probe.rotation}")

    first = clips[0].probe
    for c in clips[1:]:
        if key(c.probe) != key(first):
            return (
                f"video stream mismatch within group: {clips[0].mp4_path.name} is "
                f"{describe(first)}, but {c.mp4_path.name} is {describe(c.probe)}. "
                f"Refusing to concat mismatched streams."
            )
    return None


def _iso6709(lat: float, lon: float, alt: float) -> str:
    return f"{lat:+.4f}{lon:+.4f}{alt:+.3f}/"


def _build_merged_telemetry(clips: list[Clip]) -> tuple[str | None, list[str]]:
    """Builds merged-SRT text for a clip group.

    Returns (srt_text, extra_warnings). srt_text is None when no clip in
    the group has any parsed cues (nothing to mux). Any cueless clip
    within an otherwise-telemetry-bearing group contributes zero cue
    blocks but cumulative_offset still advances by its probed duration
    (so later clips' cues stay in sync) -- which means the merged
    subtitle track has a real time gap over that clip's span. That used
    to be discoverable only indirectly via the generic "no .SRT sidecar"
    warning; this makes the actual consequence (a gap, of this duration,
    at this point in the merged file) explicit.

    Pulled out as a pure function (no ffmpeg/ffprobe calls) so this exact
    logic -- the source of a real silent-gap bug -- can be unit tested
    directly instead of only being reachable through the full merge_group
    subprocess pipeline.
    """
    if not any(c.cues for c in clips):
        return None, []
    warnings: list[str] = []
    blocks: list[str] = []
    cumulative_offset = 0.0
    cue_index = 1
    for c in clips:
        if not c.cues:
            warnings.append(
                f"{c.mp4_path.name}: no telemetry parsed for this clip -- merged "
                f"subtitle track will have a ~{c.probe.duration_s:.1f}s gap here"
            )
        for cue in c.cues:
            blocks.append(format_cue_block(
                cue_index,
                cue.cue_start_s + cumulative_offset,
                cue.cue_end_s + cumulative_offset,
                cue,
            ))
            cue_index += 1
        cumulative_offset += c.probe.duration_s
    return "\n".join(blocks), warnings


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
    # ffmpeg writes to a hidden .partial name and we rename on success only,
    # so a mid-run failure (disk full, killed, decode error) never leaves a
    # broken file sitting at the real output name -- indistinguishable from
    # a real output to Lightroom or a plain directory listing.
    # Must keep the real extension (".mp4") -- not just append ".partial" --
    # because ffmpeg's output muxer is selected from the filename
    # extension; a name ending in ".partial" makes it fail immediately
    # with "Unable to choose an output format" before ever writing bytes.
    tmp_out_path = out_path.with_name(f".{out_path.stem}.partial{out_path.suffix}")

    merged_srt_text, telemetry_warnings = _build_merged_telemetry(clips)
    warnings += telemetry_warnings

    with tempfile.TemporaryDirectory(prefix="air3_ingest_") as tmp:
        tmp_path = Path(tmp)
        concat_list = tmp_path / "concat.txt"
        concat_list.write_text("\n".join(_quote_concat_path(c.mp4_path) for c in clips) + "\n")

        # -hide_banner -loglevel warning: without this, ffmpeg's stderr
        # carries its full version banner + per-frame progress line even
        # on success, which would drown out genuine non-fatal warnings
        # (e.g. timestamp discontinuities) in the warnings list surfaced
        # to the caller.
        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "warning",
               "-f", "concat", "-safe", "0", "-i", str(concat_list)]

        if merged_srt_text is not None:
            merged_srt = tmp_path / "merged.srt"
            merged_srt.write_text(merged_srt_text)
            cmd += ["-i", str(merged_srt)]

        cmd += ["-map", "0:v:0"]
        if merged_srt_text is not None:
            cmd += ["-map", "1:0", "-c:v", "copy", "-c:s", "mov_text",
                    "-metadata:s:s:0", "handler_name=Air3 Telemetry"]
        else:
            cmd += ["-c:v", "copy"]

        if first.probe.rotation:
            # ffmpeg's concat demuxer + stream copy doesn't reliably
            # propagate source rotation side data/tags across the concat
            # boundary; set it explicitly on the output video stream so
            # playback orientation isn't silently lost or corrupted.
            cmd += ["-metadata:s:v:0", f"rotate={first.probe.rotation}"]

        if first.probe.creation_time:
            cmd += ["-metadata", f"creation_time={first.probe.creation_time}"]
        gps_cue = _first_gps_cue(clips)
        if gps_cue:
            cmd += ["-metadata", f"location={_iso6709(gps_cue.latitude, gps_cue.longitude, gps_cue.abs_alt)}"]
        elif first.probe.container_location:
            # No SRT-derived GPS anywhere in the group, but the container
            # itself carries a location tag (e.g. written by DJI's own app)
            # -- use it as a fallback rather than leaving location unset.
            cmd += ["-metadata", f"location={first.probe.container_location}"]

        cmd += [str(tmp_out_path)]

        try:
            _run(cmd, warnings)
        except RuntimeError as e:
            tmp_out_path.unlink(missing_ok=True)
            return MergeResult(ok=False, output_path=None, source_files=source_names,
                                error=str(e), warnings=warnings)

    tmp_out_path.rename(out_path)
    return MergeResult(ok=True, output_path=out_path, source_files=source_names, warnings=warnings)
