"""
BWF/iXML probing and device classification for audio-recorder ingest.

Field disposition (ffprobe + exiftool combined view of a WAV's bext/iXML
metadata), verified against real sample recordings from three confirmed
devices (Zoom F3, Sound Devices MixPre-6, Zoom H3-VR):

  extracted:  codec_name, sample_rate, channels, bits_per_sample (ffprobe
              stream); duration_ts/time_reference (ffprobe format tags --
              sample-accurate, used for continuity arithmetic in
              audio_merge.py); encoded_by (device identification);
              date/creation_time (wall-clock start); BwfxmlScene,
              BwfxmlTake, BwfxmlNote (exiftool iXML -- the per-device
              continuity signals confirmed by direct investigation of real
              files, see audio_merge.py)
  raw-only:   none currently captured beyond what's listed above -- this
              module only reads what audio_merge.py's continuity checks
              and provenance stamping actually consume.
  dropped-with-reason: BWF_UMID (observed all-zero on every device
              tested, never populated); BwfxmlFileSetTotalFiles/
              BwfxmlFileSetFileSetIndex (present on every device tested,
              but PROVEN UNRELIABLE -- confirmed false, i.e. reporting
              "1"/"A", on real verified multi-file splits on all three of
              Zoom F3, Zoom H3-VR, and Sound Devices MixPre-6; recording
              it would invite a future caller to trust it for a grouping
              decision, which real data has already disproven);
              BwfxmlProject/BwfxmlTape (present on some devices, not
              needed for continuity detection); the raw `comment`/
              CodingHistory free-text (some firmwares duplicate the iXML
              Scene/Take/Note fields as human-readable key=value text in
              the bext Description field -- exiftool's parsed Bwfxml*
              fields are used as the single canonical source instead of
              also regex-parsing this duplicate representation, per the
              project's syntactic-semantic-seam discipline: one field,
              one source of truth).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import procs

FFPROBE = "ffprobe"
EXIFTOOL = "exiftool"

DEVICE_F3 = "zoom_f3"
DEVICE_H3VR = "zoom_h3vr"
DEVICE_MIXPRE6 = "mixpre6"
DEVICE_UNKNOWN = "unknown"


class AudioProbeError(RuntimeError):
    pass


@dataclass
class AudioProbe:
    duration_s: float
    duration_samples: int
    sample_rate: int
    channels: int
    bits_per_sample: int
    codec_name: str
    device: str  # DEVICE_* constant
    encoded_by: str | None  # raw device-identifying string, stripped
    start_dt: datetime | None  # from bext date + creation_time tags; None if absent/unparseable
    time_reference_samples: int | None  # BWF TimeReference: samples since local midnight
    scene: str | None  # Bwfxml Scene
    take: str | None  # Bwfxml Take, coerced to str (Sound Devices reports it as an int)
    note: str | None  # Bwfxml Note


def classify_device(encoded_by: str | None) -> str:
    """Single canonical device classifier -- every continuity check in
    audio_merge.py dispatches off this, never re-derives device identity
    from a filename pattern or any other signal."""
    if not encoded_by:
        return DEVICE_UNKNOWN
    tag = encoded_by.strip()
    if tag.startswith("ZOOM F3"):
        return DEVICE_F3
    if tag.startswith("ZOOM H3-VR"):
        return DEVICE_H3VR
    if tag.startswith("SoundDev: MixPre-6"):
        return DEVICE_MIXPRE6
    return DEVICE_UNKNOWN


def _run_json(cmd: list[str]) -> object:
    out = procs.run(cmd)
    return json.loads(out)


def probe_audio(wav_path: Path, warnings: list[str] | None = None) -> AudioProbe:
    ff = _run_json([
        FFPROBE, "-v", "error", "-select_streams", "a:0",
        "-show_streams", "-show_format", "-of", "json", str(wav_path),
    ])
    streams = ff.get("streams") or []
    if not streams:
        raise AudioProbeError(f"{wav_path.name}: ffprobe found no audio stream (corrupt or non-audio file?)")
    stream = streams[0]
    fmt = ff.get("format", {})
    tags = fmt.get("tags", {})

    if "duration_ts" not in stream:
        raise AudioProbeError(f"{wav_path.name}: ffprobe reported no sample-accurate duration")
    if "duration" not in fmt:
        raise AudioProbeError(f"{wav_path.name}: ffprobe reported no container duration")

    encoded_by = tags.get("encoded_by")
    device = classify_device(encoded_by)

    start_dt = None
    date_tag, time_tag = tags.get("date"), tags.get("creation_time")
    if date_tag and time_tag:
        try:
            start_dt = datetime.strptime(f"{date_tag} {time_tag}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            start_dt = None  # caller falls back to file mtime

    time_reference = tags.get("time_reference")

    exif = {}
    try:
        exif_out = _run_json([EXIFTOOL, "-json", str(wav_path)])
        exif = exif_out[0] if isinstance(exif_out, list) and exif_out else {}
    except RuntimeError as e:
        if warnings is not None:
            warnings.append(
                f"{wav_path.name}: exiftool failed ({e}); scene/take/note metadata "
                f"unavailable for this clip, so it won't be auto-merged with neighbors"
            )

    take = exif.get("BwfxmlTake")

    return AudioProbe(
        duration_s=float(fmt["duration"]),
        duration_samples=int(stream["duration_ts"]),
        sample_rate=int(stream["sample_rate"]),
        channels=int(stream["channels"]),
        bits_per_sample=int(stream.get("bits_per_sample") or 0),
        codec_name=stream["codec_name"],
        device=device,
        encoded_by=encoded_by.strip() if encoded_by else None,
        start_dt=start_dt,
        time_reference_samples=int(time_reference) if time_reference is not None else None,
        scene=exif.get("BwfxmlScene") or None,
        take=str(take) if take not in (None, "") else None,
        note=exif.get("BwfxmlNote") or None,
    )
