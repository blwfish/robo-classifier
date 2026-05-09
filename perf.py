"""
perf.py — performance observability for robo-classifier runs.

Records per-stage timing + throughput for every pipeline and ingest run,
appended to ~/.robo-classifier/perf_log.jsonl. One JSON line per run.

Usage (via PerfRecorder wrapper — not called directly):
    recorder = PerfRecorder(original_progress_cb, run_type="pipeline", ...)
    # pass recorder.cb as the progress_cb to run_pipeline() / ingest()
    # recorder saves automatically on the __end__ sentinel event

Standalone:
    python perf.py          # print last 10 runs as a table
    python perf.py --json   # raw jsonl tail
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Optional


PERF_LOG = Path.home() / ".robo-classifier" / "perf_log.jsonl"


# ---------------------------------------------------------------------------
# Hardware snapshot
# ---------------------------------------------------------------------------

def hardware_snapshot() -> dict:
    """
    Collect a one-time hardware fingerprint for this machine.
    Safe to call without torch installed (falls back gracefully).
    """
    snap: dict = {
        "hostname": platform.node(),
        "os":       platform.system(),
        "cpu":      platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count() or 0,
        "ram_gb":   _ram_gb(),
        "device":   "cpu",
        "device_name": "CPU",
    }

    try:
        import torch
        if torch.backends.mps.is_available():
            snap["device"] = "mps"
            snap["device_name"] = "Apple MPS"
        elif torch.cuda.is_available():
            snap["device"] = "cuda"
            idx = torch.cuda.current_device()
            snap["device_name"] = torch.cuda.get_device_name(idx)
            snap["cuda_mem_gb"] = round(
                torch.cuda.get_device_properties(idx).total_memory / 1e9, 1
            )
    except Exception:
        pass

    return snap


def _ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    # macOS fallback
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return round(int(out.strip()) / 1e9, 1)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Storage class detection
# ---------------------------------------------------------------------------

def storage_class(path: str | Path) -> str:
    """
    Infer storage class for the drive containing `path`.
    Returns one of: "nvme", "usb3", "usb2", "sata", "thunderbolt",
                    "network", "ram", "unknown".
    macOS: uses `diskutil info`. Windows/Linux: heuristic from mount point.
    """
    path = Path(path).resolve()

    if platform.system() == "Darwin":
        return _storage_class_macos(path)
    if platform.system() == "Windows":
        return _storage_class_windows(path)
    return _storage_class_linux(path)


def _storage_class_macos(path: Path) -> str:
    # diskutil info requires a device node or mount point, not a subdirectory.
    # Use `df` to find the device for this path, then query diskutil.
    try:
        df_out = subprocess.check_output(
            ["df", "-n", str(path)], text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        # df output: "Filesystem   512-blocks  Used  ..."
        # First field of second line is the device (e.g. /dev/disk4s1 or mount label)
        device = df_out.strip().splitlines()[-1].split()[0]
    except Exception:
        device = str(path)

    try:
        out = subprocess.check_output(
            ["diskutil", "info", device],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except Exception:
        return "unknown"

    protocol = ""
    media_type = ""
    for line in out.splitlines():
        lo = line.strip().lower()
        if "protocol:" in lo:
            protocol = lo.split(":", 1)[1].strip()
        elif "media type:" in lo:
            media_type = lo.split(":", 1)[1].strip()

    # PCI-Express / Apple Fabric → NVMe (internal SSDs on Apple Silicon / Intel Macs)
    if "pci-express" in protocol or "nvme" in protocol or "apple fabric" in protocol:
        return "nvme"
    if "thunderbolt" in protocol:
        return "thunderbolt"
    if "usb" in protocol:
        # diskutil doesn't always distinguish USB 3.2 vs 3.0 by protocol name,
        # but the bus speed is usually in the output.
        for line in out.splitlines():
            if "bus" in line.lower() and "10" in line:
                return "usb3"
        return "usb3"   # conservatively label USB as usb3 (common case for cards)
    if "sata" in protocol:
        if "solid state" in media_type:
            return "sata"
        return "hdd"
    # Network volumes: AFP, SMB, NFS
    if any(p in protocol for p in ("afp", "smb", "nfs", "apfs")):
        return "network"
    # Synthesized / RAM disk
    if "disk image" in media_type or "virtual" in media_type:
        return "ram"

    return "unknown"


def _storage_class_windows(path: Path) -> str:
    # Rough heuristic: removable drives are usually USB cards
    try:
        import ctypes
        drive = str(path.drive) + "\\"
        dtype = ctypes.windll.kernel32.GetDriveTypeW(drive)
        # DRIVE_REMOVABLE = 2, DRIVE_FIXED = 3, DRIVE_REMOTE = 4
        if dtype == 2:
            return "usb3"
        if dtype == 4:
            return "network"
    except Exception:
        pass
    return "unknown"


def _storage_class_linux(path: Path) -> str:
    # Check /sys/block for rotational flag
    try:
        import subprocess
        out = subprocess.check_output(
            ["lsblk", "-no", "ROTA,TRAN", str(path)],
            text=True, timeout=5,
        ).strip().splitlines()[0].split()
        rota, tran = out[0], out[1] if len(out) > 1 else ""
        if tran in ("nvme",):
            return "nvme"
        if tran in ("usb",):
            return "usb3"
        if rota == "1":
            return "hdd"
        return "sata"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# PerfRecorder — wraps a progress_cb, intercepts events, saves on completion
# ---------------------------------------------------------------------------

class PerfRecorder:
    """
    Wraps a pipeline or ingest progress_cb to transparently measure
    per-stage timing and throughput, then saves a record on __end__.

    Usage:
        rec = PerfRecorder(original_cb, run_type="pipeline",
                           input_path=str(d), preset="m1-max", profile="pca")
        # pass rec.cb to run_pipeline() or ingest()
    """

    def __init__(
        self,
        downstream_cb: Optional[Callable],
        run_type: str,          # "pipeline" | "ingest"
        input_path: str = "",
        preset: str = "",
        profile: str = "",
        extra: Optional[dict] = None,
    ):
        self._cb         = downstream_cb
        self._run_type   = run_type
        self._input_path = input_path
        self._preset     = preset
        self._profile    = profile
        self._extra      = extra or {}

        self._run_id     = uuid.uuid4().hex[:12]
        self._start_ts   = time.time()
        self._hw         = hardware_snapshot()
        self._storage    = storage_class(input_path) if input_path else "unknown"

        # Stage tracking
        self._stages: list[dict] = []
        self._cur_stage: Optional[str] = None
        self._cur_stage_start: float = 0.0
        self._cur_files: int = 0
        self._cur_total: int = 0

        # Byte tracking: populated from scan_done (ingest) or known file sizes
        self._stage_bytes: dict[str, int] = {}  # stage_name → bytes

    def cb(self, event: dict):
        """Drop-in replacement for the original progress_cb."""
        self._handle(event)
        if self._cb:
            self._cb(event)

    def set_stage_bytes(self, stage: str, total_bytes: int):
        """Call this externally if you know the byte total for a stage."""
        self._stage_bytes[stage] = total_bytes

    # ---- internal ----

    def _handle(self, event: dict):
        t = event.get("type")

        if t == "stage":
            self._close_stage()
            self._cur_stage = event.get("stage", "unknown")
            self._cur_stage_start = time.time()
            self._cur_files = 0
            self._cur_total = event.get("total", 0)

        elif t == "progress":
            stage = event.get("stage") or self._cur_stage
            if stage and stage != self._cur_stage:
                self._close_stage()
                self._cur_stage = stage
                self._cur_stage_start = time.time()
                self._cur_files = 0
            # Update file count from whichever field is present
            self._cur_files = event.get("done", event.get("current", self._cur_files))
            self._cur_total = max(self._cur_total, event.get("total", self._cur_total))

        elif t in ("scan_done", "exif_start"):
            # Ingest phases
            if t == "scan_done" and self._cur_stage is None:
                self._cur_stage = "scan"
                self._cur_stage_start = time.time()
            n = event.get("count", event.get("total", 0))
            self._cur_files = max(self._cur_files, n)
            self._cur_total = max(self._cur_total, n)

        elif t == "file_result":
            if event.get("status") == "copied":
                self._cur_files += 1

        elif t == "__end__":
            self._close_stage()
            self._save(event.get("summary"))

    def _close_stage(self):
        if self._cur_stage is None:
            return
        duration = time.time() - self._cur_stage_start
        stage_rec: dict = {
            "name":      self._cur_stage,
            "duration_s": round(duration, 2),
            "files":     self._cur_files,
            "total":     self._cur_total,
        }
        if duration > 0 and self._cur_files > 0:
            stage_rec["files_per_s"] = round(self._cur_files / duration, 1)
        nbytes = self._stage_bytes.get(self._cur_stage, 0)
        if nbytes and duration > 0:
            stage_rec["bytes"] = nbytes
            stage_rec["mb_per_s"] = round(nbytes / duration / 1e6, 1)
        self._stages.append(stage_rec)
        self._cur_stage = None

    def _save(self, summary: Optional[dict]):
        total_dur = time.time() - self._start_ts
        record = {
            "run_id":        self._run_id,
            "ts":            self._start_ts,
            "run_type":      self._run_type,
            "hostname":      self._hw["hostname"],
            "device":        self._hw["device"],
            "device_name":   self._hw["device_name"],
            "cpu_count":     self._hw["cpu_count"],
            "ram_gb":        self._hw["ram_gb"],
            "preset":        self._preset,
            "profile":       self._profile,
            "input_path":    self._input_path,
            "storage_class": self._storage,
            "stages":        self._stages,
            "total_duration_s": round(total_dur, 2),
        }
        if "cuda_mem_gb" in self._hw:
            record["cuda_mem_gb"] = self._hw["cuda_mem_gb"]
        if summary:
            record["summary"] = summary
        if self._extra:
            record.update(self._extra)

        _append_record(record)


# ---------------------------------------------------------------------------
# Log I/O
# ---------------------------------------------------------------------------

def _append_record(record: dict):
    PERF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PERF_LOG, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_recent(n: int = 20) -> list[dict]:
    """Return the last `n` records from the perf log, newest first."""
    if not PERF_LOG.exists():
        return []
    lines = []
    try:
        with open(PERF_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    except OSError:
        return []
    records = []
    for line in reversed(lines[-n:]):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


# ---------------------------------------------------------------------------
# Source-dir byte totals (used by pipeline runner to report MB/s for extract)
# ---------------------------------------------------------------------------

def measure_input_bytes(input_dir: Path, extensions: Optional[set] = None) -> int:
    """Return total byte size of image files in input_dir (non-recursive)."""
    total = 0
    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        if extensions is None or f.suffix.lower() in extensions:
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------
# CLI — pretty-print recent runs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Show recent robo-classifier perf runs.")
    parser.add_argument("-n", type=int, default=10, help="Number of runs to show (default 10)")
    parser.add_argument("--json", action="store_true", help="Raw JSON output")
    args = parser.parse_args()

    runs = load_recent(args.n)
    if not runs:
        print("No runs recorded yet.", file=sys.stderr)
        sys.exit(0)

    if args.json:
        for r in reversed(runs):
            print(json.dumps(r))
        sys.exit(0)

    # Pretty table
    from datetime import datetime

    def fmt_dur(s):
        if s < 60:   return f"{s:.0f}s"
        if s < 3600: return f"{s/60:.1f}m"
        return f"{s/3600:.1f}h"

    def fmt_rate(stage):
        parts = []
        if "files_per_s" in stage:
            parts.append(f"{stage['files_per_s']:.0f} f/s")
        if "mb_per_s" in stage:
            parts.append(f"{stage['mb_per_s']:.0f} MB/s")
        return "  ".join(parts) if parts else "—"

    SEP = "─" * 100
    for run in reversed(runs):
        ts = datetime.fromtimestamp(run["ts"]).strftime("%Y-%m-%d %H:%M")
        hw = f"{run['device_name']} · {run['cpu_count']}C · {run['ram_gb']}GB RAM"
        stor = run.get("storage_class", "?")
        preset = run.get("preset") or "(default)"
        profile = run.get("profile") or "—"
        total = fmt_dur(run.get("total_duration_s", 0))

        print(SEP)
        print(f"  {ts}  {run['hostname']}  [{run['run_type']}]  total={total}")
        print(f"  hw: {hw}  storage: {stor}  preset: {preset}  profile: {profile}")

        stages = run.get("stages", [])
        if stages:
            max_dur = max(s["duration_s"] for s in stages) if stages else 1
            for s in stages:
                bar_len = max(1, int(s["duration_s"] / max_dur * 20))
                bar = "█" * bar_len
                dur = fmt_dur(s["duration_s"])
                files = f"{s.get('files', 0):,} files"
                rate = fmt_rate(s)
                bottleneck = " ◄ BOTTLENECK" if s["duration_s"] == max_dur else ""
                print(f"    {s['name']:20s} {bar:20s} {dur:>7}  {files:>14}  {rate}{bottleneck}")

    print(SEP)
