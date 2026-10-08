"""Frame-accurate cue timing (D-103).

Cue starts and ends are snapped to video frames with at least two frames between consecutive cues. Optionally (setting
`snap_to_shots`, default off) an edge within 250 ms of a shot change moves to the cut, using the timestamps of
FFmpeg's scene detector (`select='gt(scene,0.3)'`), which are cached per video.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from app.core.errors import JobCancelled
from app.core.exporter import Cue

log = logging.getLogger(__name__)

MIN_GAP_FRAMES = 2
SHOT_WINDOW_S = 0.25
SCENE_THRESHOLD = 0.3
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_PTS = re.compile(r"pts_time:([0-9.]+)")


def video_fps(video: Path) -> float | None:
    """Frame rate of the first video stream, or None for audio-only files and unreadable media."""
    try:
        import av

        with av.open(str(video)) as container:
            if not container.streams.video:
                return None
            rate = container.streams.video[0].average_rate
            return float(rate) if rate and float(rate) > 0 else None
    except Exception as exc:                       # never fail a job over timing polish
        log.warning("Could not read the frame rate of %s: %s", video, exc)
        return None


def _snap(seconds: float, fps: float) -> float:
    return round(seconds * fps) / fps


def snap_cues(cues: Sequence[Cue], fps: float, min_gap_frames: int = MIN_GAP_FRAMES) -> list[Cue]:
    """Starts and ends on frame boundaries, every cue at least one frame long, at least `min_gap_frames` frames
    between a cue and the next one. A gap is made by ending the earlier cue sooner; only when that would leave it
    shorter than a frame is the later cue started later."""
    frame = 1.0 / fps
    gap = min_gap_frames * frame
    result = []
    for cue in sorted(cues, key=lambda c: c.start):
        start, end = _snap(cue.start, fps), _snap(cue.end, fps)
        result.append(replace(cue, start=start, end=end if end > start else start + frame))
    for earlier, later in zip(result, result[1:]):
        if later.start - earlier.end >= gap - 1e-6:
            continue
        end = _snap(later.start - gap, fps)
        if end >= earlier.start + frame - 1e-6:
            earlier.end = end
        else:
            later.start = _snap(earlier.end + gap, fps)
            later.end = max(later.end, later.start + frame)
    return [replace(c, start=round(c.start, 3), end=round(c.end, 3)) for c in result]


def shot_changes(video: Path, cache: Path, ffmpeg: str | None = None,
                 cancel: threading.Event | None = None,
                 progress: Callable[[float], None] | None = None) -> list[float]:
    """Times (seconds) of shot changes, cached in `cache` per video (JSON). Returns [] when FFmpeg is missing or
    fails; cancelling raises JobCancelled."""
    from app.utils.hashing import file_fingerprint

    fingerprint = file_fingerprint(video)
    try:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("fingerprint") == fingerprint and saved.get("threshold") == SCENE_THRESHOLD:
            return [float(t) for t in saved["shots"]]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if ffmpeg is None:
        import shutil
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log.warning("FFmpeg not found; shot changes are not used")
        return []
    # A small picture is enough for scene detection and makes it several times faster.
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-i", str(video), "-an", "-sn", "-vf",
           f"scale=320:-2,select='gt(scene,{SCENE_THRESHOLD})',showinfo", "-f", "null", "-"]
    shots: list[float] = []
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, errors="replace",
                               creationflags=_NO_WINDOW)
    try:
        for line in process.stderr:
            if cancel is not None and cancel.is_set():
                raise JobCancelled()
            match = _PTS.search(line) if "Parsed_showinfo" in line else None
            if match:
                shots.append(round(float(match.group(1)), 3))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
    if process.returncode not in (0, None):
        log.warning("Shot detection failed (FFmpeg exit %s); shot changes are not used", process.returncode)
        return []
    cache.write_text(json.dumps({"fingerprint": fingerprint, "threshold": SCENE_THRESHOLD, "shots": shots}),
                     encoding="utf-8")
    return shots


def snap_to_shots(cues: Sequence[Cue], shots: Sequence[float], fps: float, min_duration: float = 0.833,
                  window: float = SHOT_WINDOW_S) -> list[Cue]:
    """Move a cue start or end to a shot change within `window` seconds when the cue stays at least `min_duration`
    long and does not run into the neighbouring cue; the frame gaps are restored afterwards."""
    if not shots:
        return list(cues)
    ordered = [replace(c) for c in sorted(cues, key=lambda c: c.start)]

    def nearest(t: float) -> float | None:
        close = [s for s in shots if abs(s - t) <= window]
        return min(close, key=lambda s: abs(s - t)) if close else None

    for i, cue in enumerate(ordered):
        previous_end = ordered[i - 1].end if i else float("-inf")
        next_start = ordered[i + 1].start if i + 1 < len(ordered) else float("inf")
        cut = nearest(cue.start)
        if cut is not None and cut > previous_end and cue.end - cut >= min_duration:
            cue.start = cut
        cut = nearest(cue.end)
        if cut is not None and cut < next_start and cut - cue.start >= min_duration:
            cue.end = cut
    return snap_cues(ordered, fps)
