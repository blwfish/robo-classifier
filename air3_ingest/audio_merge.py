"""
Discovery, grouping, and ffmpeg+bwfmetaedit merge logic for audio-recorder
ingest (Zoom F3, Sound Devices MixPre-6, Zoom H3-VR).

TASCAM DR-40 and any other/unrecognized device are never auto-merged: no
confirmed real split example exists for TASCAM, and it carries no BWF/iXML
metadata at all to detect one from (see audio_probe.py's module docstring).

Continuity detection is per-device, NOT a single shared gap-threshold rule
like DJI Air3 (merge.py) -- confirmed by direct inspection of real recorder
output, each device's own "official" multi-part metadata
(BwfxmlFileSetTotalFiles/FileSetIndex) is unreliable on every device tested,
so a different signal is used per device:

  Zoom F3:      matching "<prefix>_<NNN>" filename with a consecutively
                incrementing numeric suffix, same calendar date, AND an
                exact (~0-sample) gap between the end of clip N and the
                start of clip N+1 (BWF TimeReference + duration_ts
                arithmetic). Confirmed real split: "...Swoope_001.WAV" ->
                "...Swoope_002.WAV", exact 0-sample gap.
  MixPre-6:     matching "MixPre-<NNN>.WAV" numeric sequence, same
                calendar date, a non-empty and *matching* BwfxmlNote text
                (BwfxmlTake increments per file on this device, so it is
                NOT usable as a continuity signal here), AND an exact
                (~0-sample) gap. Confirmed real split: MixPre-275.WAV ->
                MixPre-276.WAV, exact 0-sample gap.
  Zoom H3-VR:   matching "<prefix>_<NNN>" numeric sequence, same calendar
                date, AND a constant (non-empty, equal) BwfxmlTake across
                the pair -- NOT sample-arithmetic, which was confirmed to
                produce an identical timing signature (-17,280 samples /
                -0.36s) on both a genuine split and two separate,
                manually-triggered takes, so it cannot distinguish the two
                on its own. Once take-number confirms continuity, a real
                split still overlaps by a small, per-instance-computed
                number of samples that must be trimmed during merge (not
                just appended), via ffmpeg concat's `inpoint` directive.

A boundary discrepancy larger than MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S
(confirmed real splits were 0s and ~0.36s) is treated as a sign the
continuity signal was misleading -- refuse the auto-merge (F3/MixPre-6) or
fall back to a plain, untrimmed append with a warning (H3-VR, where
continuity is already confirmed by take number) rather than silently
guessing at a large trim.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import procs
from audio_probe import (
    DEVICE_F3,
    DEVICE_H3VR,
    DEVICE_MIXPRE6,
    AudioProbe,
    AudioProbeError,
    probe_audio,
)

FFMPEG = "ffmpeg"
BWFMETAEDIT = "bwfmetaedit"

MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S = 2.0

_TRAILING_DIGITS_RE = re.compile(r"^(?P<prefix>.*?)(?P<idx>\d+)$")
F3_FILENAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)$")
MIXPRE_FILENAME_RE = re.compile(r"^MixPre-(?P<idx>\d+)$", re.IGNORECASE)
H3VR_FILENAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)$")

DEVICE_TAG = {DEVICE_F3: "F3", DEVICE_MIXPRE6: "MixPre6", DEVICE_H3VR: "H3VR"}


@dataclass
class AudioClip:
    wav_path: Path
    probe: AudioProbe
    start_dt: datetime  # probe.start_dt if available, else file mtime
    start_is_estimated: bool


@dataclass
class AudioClipGroup:
    clips: list[AudioClip]


@dataclass
class AudioMergeResult:
    ok: bool
    output_path: Path | None
    source_files: list[str]
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def _natural_sort_key(path: Path) -> tuple[str, int]:
    """Sorts '...Swoope_9' before '...Swoope_10' -- a plain string sort would
    not, since '1' < '9' lexically. Generic across all three devices' naming
    conventions (all end in a numeric take/file index); the actual
    prefix/index *validation* used for continuity decisions is still each
    device's own regex, this is only for a sane default display order."""
    m = _TRAILING_DIGITS_RE.match(path.stem)
    if m:
        return (m.group("prefix"), int(m.group("idx")))
    return (path.stem, -1)


def discover_audio_clips(source_dir: Path) -> tuple[list[AudioClip], list[str]]:
    """Returns (clips sorted by device then natural filename order, warnings)."""
    warnings: list[str] = []
    wav_paths = sorted(
        (p for p in Path(source_dir).rglob("*") if p.is_file() and p.suffix.lower() == ".wav"),
        key=_natural_sort_key,
    )
    clips: list[AudioClip] = []
    for wav_path in wav_paths:
        try:
            probe = probe_audio(wav_path, warnings)
        except AudioProbeError as e:
            warnings.append(f"{wav_path.name}: skipped -- probe failed: {e}")
            continue

        if probe.start_dt is not None:
            start_dt, start_is_estimated = probe.start_dt, False
        else:
            start_dt = datetime.fromtimestamp(wav_path.stat().st_mtime)
            start_is_estimated = True
            warnings.append(f"{wav_path.name}: no bext origination date/time; using file mtime as an estimate")

        clips.append(AudioClip(wav_path=wav_path, probe=probe, start_dt=start_dt,
                                start_is_estimated=start_is_estimated))

    clips.sort(key=lambda c: (c.probe.device, _natural_sort_key(c.wav_path)))
    return clips, warnings


def _same_calendar_date(prev: AudioClip, cur: AudioClip) -> bool:
    return prev.start_dt.date() == cur.start_dt.date()


def _sample_boundary(prev: AudioClip, cur: AudioClip) -> tuple[int, float] | None:
    """Returns (overlap_samples, discrepancy_s) from BWF TimeReference +
    duration_ts arithmetic, or None if either clip lacks a TimeReference.
    overlap_samples > 0 means cur starts that many samples before prev's
    computed end (needs trimming); <= 0 means a gap (no trim)."""
    if prev.probe.time_reference_samples is None or cur.probe.time_reference_samples is None:
        return None
    prev_end = prev.probe.time_reference_samples + prev.probe.duration_samples
    overlap_samples = prev_end - cur.probe.time_reference_samples
    discrepancy_s = abs(overlap_samples) / cur.probe.sample_rate
    return overlap_samples, discrepancy_s


def _f3_continuous(prev: AudioClip, cur: AudioClip, warnings: list[str]) -> int | None:
    m_prev, m_cur = F3_FILENAME_RE.match(prev.wav_path.stem), F3_FILENAME_RE.match(cur.wav_path.stem)
    if not (m_prev and m_cur) or m_prev.group("prefix") != m_cur.group("prefix"):
        return None
    if int(m_cur.group("idx")) != int(m_prev.group("idx")) + 1:
        return None
    if not _same_calendar_date(prev, cur):
        return None
    boundary = _sample_boundary(prev, cur)
    if boundary is None:
        return None
    overlap_samples, discrepancy_s = boundary
    if discrepancy_s > MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S:
        warnings.append(
            f"{prev.wav_path.name} -> {cur.wav_path.name}: filename sequence matches a Zoom F3 "
            f"split, but the boundary is off by {discrepancy_s:.2f}s (confirmed real splits are "
            f"~0s) -- treating as separate takes rather than guessing"
        )
        return None
    return max(overlap_samples, 0)


def _mixpre6_continuous(prev: AudioClip, cur: AudioClip, warnings: list[str]) -> int | None:
    m_prev, m_cur = MIXPRE_FILENAME_RE.match(prev.wav_path.stem), MIXPRE_FILENAME_RE.match(cur.wav_path.stem)
    if not (m_prev and m_cur):
        return None
    if int(m_cur.group("idx")) != int(m_prev.group("idx")) + 1:
        return None
    if not _same_calendar_date(prev, cur):
        return None
    # BwfxmlTake increments per-file on this device (confirmed on a real
    # split), so it can't corroborate continuity; BwfxmlNote is the field
    # that stays constant instead. Require it non-empty too: filename
    # sequence + sample-exact timing alone, with no note text at all,
    # is one fewer corroborating signal than every case actually confirmed.
    if not prev.probe.note or prev.probe.note != cur.probe.note:
        return None
    boundary = _sample_boundary(prev, cur)
    if boundary is None:
        return None
    overlap_samples, discrepancy_s = boundary
    if discrepancy_s > MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S:
        warnings.append(
            f"{prev.wav_path.name} -> {cur.wav_path.name}: filename sequence and Note field match "
            f"a MixPre-6 split, but the boundary is off by {discrepancy_s:.2f}s (confirmed real "
            f"splits are ~0s) -- treating as separate takes rather than guessing"
        )
        return None
    return max(overlap_samples, 0)


def _h3vr_continuous(prev: AudioClip, cur: AudioClip, warnings: list[str]) -> int | None:
    m_prev, m_cur = H3VR_FILENAME_RE.match(prev.wav_path.stem), H3VR_FILENAME_RE.match(cur.wav_path.stem)
    if not (m_prev and m_cur) or m_prev.group("prefix") != m_cur.group("prefix"):
        return None
    if int(m_cur.group("idx")) != int(m_prev.group("idx")) + 1:
        return None
    if not _same_calendar_date(prev, cur):
        return None
    # Take number is the confirmed-reliable continuity signal on this
    # device -- NOT sample-timing, which produces the same ~0.36s signature
    # on both a genuine split and two unrelated back-to-back takes.
    if not prev.probe.take or prev.probe.take != cur.probe.take:
        return None
    boundary = _sample_boundary(prev, cur)
    if boundary is None:
        # Continuity is already confirmed via take number; without sample
        # data to compute a precise trim, fall back to a plain append
        # rather than refusing a confirmed continuation outright.
        warnings.append(
            f"{prev.wav_path.name} -> {cur.wav_path.name}: take number confirms one continuous "
            f"H3-VR recording, but no TimeReference was available to compute the usual overlap "
            f"trim -- merging with a plain append, the seam may have a brief (well under a "
            f"second) glitch or repeat"
        )
        return 0
    overlap_samples, discrepancy_s = boundary
    if discrepancy_s > MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S:
        warnings.append(
            f"{prev.wav_path.name} -> {cur.wav_path.name}: take number confirms one continuous "
            f"H3-VR recording, but the computed trim ({discrepancy_s:.2f}s) is implausibly large "
            f"(confirmed real splits are ~0.36s) -- merging with a plain append instead of "
            f"trimming, please verify the output"
        )
        return 0
    return max(overlap_samples, 0)


_CONTINUITY_CHECKERS = {
    DEVICE_F3: _f3_continuous,
    DEVICE_MIXPRE6: _mixpre6_continuous,
    DEVICE_H3VR: _h3vr_continuous,
}


def _check_continuity(prev: AudioClip, cur: AudioClip, warnings: list[str]) -> int | None:
    if prev.probe.device != cur.probe.device:
        return None
    checker = _CONTINUITY_CHECKERS.get(prev.probe.device)
    if checker is None:
        return None  # unknown/unverified device: never auto-merge
    return checker(prev, cur, warnings)


def group_audio_clips(clips: list[AudioClip]) -> tuple[list[AudioClipGroup], dict[tuple[Path, Path], int], list[str]]:
    """Returns (groups sorted by first-clip start time, trim_samples_by_clip_pair, warnings).

    trim_samples_by_clip_pair maps (prev.wav_path, cur.wav_path) -> samples
    to skip from cur's start during merge; only present for pairs the
    continuity check found genuinely overlapping.
    """
    warnings: list[str] = []
    if not clips:
        return [], {}, warnings

    buckets: list[list[AudioClip]] = [[clips[0]]]
    trims: dict[tuple[Path, Path], int] = {}
    for prev, cur in zip(clips, clips[1:]):
        trim = _check_continuity(prev, cur, warnings)
        if trim is not None:
            buckets[-1].append(cur)
            if trim:
                trims[(prev.wav_path, cur.wav_path)] = trim
        else:
            buckets.append([cur])

    groups = [AudioClipGroup(clips=b) for b in buckets]
    groups.sort(key=lambda g: g.clips[0].start_dt)
    return groups, trims, warnings


def audio_group_summary(group: AudioClipGroup) -> dict:
    clips = group.clips
    first, last = clips[0], clips[-1]
    total_duration = sum(c.probe.duration_s for c in clips)
    total_size = sum(c.wav_path.stat().st_size for c in clips)
    return {
        "clip_count": len(clips),
        "clip_names": [c.wav_path.name for c in clips],
        "clip_paths": [str(c.wav_path) for c in clips],
        "start_dt": first.start_dt.isoformat(),
        # Actual end (start + duration), not last.start_dt -- matching
        # merge.py's group_summary() convention. Confirmed by cold review:
        # this previously used last.start_dt, silently mislabeling the
        # last clip's *start* time as the group's end.
        "end_dt": (last.start_dt + timedelta(seconds=last.probe.duration_s)).isoformat(),
        "total_duration_s": total_duration,
        "total_size_bytes": total_size,
        "start_location": None,  # audio recorders carry no GPS
        "end_location": None,
        "start_is_estimated": first.start_is_estimated,
        "missing_srt": [],  # no sidecar-telemetry concept for audio; kept for a stable dict shape
        "gap_to_next_s": None,  # not computed for audio groups (see module docstring: per-device continuity, not a global gap threshold)
        "device": first.probe.device,
        "scene": first.probe.scene,
        "note": first.probe.note,
    }


def _check_uniform_stream(clips: list[AudioClip]) -> str | None:
    def key(p: AudioProbe) -> tuple:
        return (p.codec_name, p.sample_rate, p.channels, p.bits_per_sample)

    def describe(p: AudioProbe) -> str:
        return f"{p.codec_name} {p.sample_rate}Hz {p.channels}ch {p.bits_per_sample}-bit"

    first = clips[0].probe
    for c in clips[1:]:
        if key(c.probe) != key(first):
            return (
                f"audio stream mismatch within group: {clips[0].wav_path.name} is "
                f"{describe(first)}, but {c.wav_path.name} is {describe(c.probe)}. "
                f"Refusing to concat mismatched streams."
            )
    return None


def _sanitize_filename_part(s: str) -> str:
    return re.sub(r'[/:*?"<>|\\]', "_", s).strip() or "untitled"


def _stamp_provenance(out_path: Path, group: AudioClipGroup, trims: dict[tuple[Path, Path], int],
                       warnings: list[str]) -> None:
    """ffmpeg's WAV muxer can't write/extend the iXML chunk at all (only the
    fixed bext fields via -write_bext), so metadata is stamped in a separate
    pass with bwfmetaedit: carries the first clip's own scene/note forward,
    plus a custom <AIR3_INGEST_MERGE> block recording provenance (source
    files, device, any trims applied) -- the audio equivalent of the mov_text
    telemetry track embedded in merged Air3 video output."""
    first = group.clips[0].probe
    trim_parts = []
    for prev, cur in zip(group.clips, group.clips[1:]):
        n = trims.get((prev.wav_path, cur.wav_path), 0)
        if n:
            trim_parts.append(f"{prev.wav_path.name}->{cur.wav_path.name}: trimmed {n} samples")
    trims_text = "; ".join(trim_parts) or "none"

    ixml_xml = (
        "<BWFXML>\n"
        f"<SCENE>{xml_escape(first.scene or '')}</SCENE>\n"
        f"<NOTE>{xml_escape(first.note or '')}</NOTE>\n"
        "<AIR3_INGEST_MERGE>\n"
        f"<SOURCE_FILES>{xml_escape(','.join(c.wav_path.name for c in group.clips))}</SOURCE_FILES>\n"
        f"<DEVICE>{xml_escape(first.device)}</DEVICE>\n"
        f"<TRIMS>{xml_escape(trims_text)}</TRIMS>\n"
        "</AIR3_INGEST_MERGE>\n"
        "</BWFXML>\n"
    )
    ixml_path = out_path.with_name(out_path.name + ".iXML.xml")
    ixml_path.write_text(ixml_xml)
    try:
        cmd = [BWFMETAEDIT]
        if first.encoded_by:
            cmd.append(f"--Originator={first.encoded_by}")
        start_dt = group.clips[0].start_dt
        cmd += [f"--OriginationDate={start_dt.strftime('%Y-%m-%d')}",
                f"--OriginationTime={start_dt.strftime('%H:%M:%S')}"]
        if first.time_reference_samples is not None:
            cmd.append(f"--Timereference={first.time_reference_samples}")
        cmd += ["--in-iXML-xml", str(out_path)]
        procs.run(cmd, warnings)
    except RuntimeError as e:
        warnings.append(
            f"bwfmetaedit provenance stamping failed (merged audio itself is fine, just "
            f"missing embedded scene/take/note metadata): {e}"
        )
    finally:
        ixml_path.unlink(missing_ok=True)


def merge_audio_group(group: AudioClipGroup, trims: dict[tuple[Path, Path], int], dest_dir: Path) -> AudioMergeResult:
    clips = group.clips
    source_names = [c.wav_path.name for c in clips]
    warnings: list[str] = []

    mismatch = _check_uniform_stream(clips)
    if mismatch:
        return AudioMergeResult(ok=False, output_path=None, source_files=source_names, error=mismatch)

    first = clips[0]
    label = first.probe.note or first.probe.scene or first.wav_path.stem
    device_tag = DEVICE_TAG.get(first.probe.device, "Audio")
    date_str = first.start_dt.strftime("%Y%m%d")
    out_name = f"{device_tag}_{date_str}_{_sanitize_filename_part(label)}.wav"

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = procs.next_available_path(dest_dir / out_name)
    # Same partial-then-rename pattern as video merges: a mid-run failure
    # never leaves a broken file sitting at the real output name.
    tmp_out_path = out_path.with_name(f".{out_path.stem}.partial{out_path.suffix}")

    with tempfile.TemporaryDirectory(prefix="audio_ingest_") as tmp:
        tmp_path = Path(tmp)
        concat_list = tmp_path / "concat.txt"
        lines = []
        for i, c in enumerate(clips):
            lines.append(procs.quote_concat_path(c.wav_path))
            if i > 0:
                trim = trims.get((clips[i - 1].wav_path, c.wav_path), 0)
                if trim:
                    trim_s = trim / c.probe.sample_rate
                    lines.append(f"inpoint {trim_s:.9f}")
        concat_list.write_text("\n".join(lines) + "\n")

        # -rf64 auto: classic RIFF WAV has a 32-bit size field (~4GiB cap) --
        # exactly the same kind of limit that split the source files in the
        # first place. A merged group can easily exceed it (confirmed: a
        # real 3-clip H3-VR merge hit 5.75GB and produced a silently
        # corrupt/unreadable output without this flag). RF64 is the
        # standard 64-bit-capable extension; "auto" only engages it when
        # the output actually needs it, so small merges are unaffected.
        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "warning",
               "-f", "concat", "-safe", "0", "-i", str(concat_list),
               "-c:a", "copy", "-rf64", "auto", str(tmp_out_path)]
        try:
            procs.run(cmd, warnings)
        except RuntimeError as e:
            tmp_out_path.unlink(missing_ok=True)
            return AudioMergeResult(ok=False, output_path=None, source_files=source_names,
                                     error=str(e), warnings=warnings)

    tmp_out_path.rename(out_path)
    _stamp_provenance(out_path, group, trims, warnings)
    return AudioMergeResult(ok=True, output_path=out_path, source_files=source_names, warnings=warnings)
