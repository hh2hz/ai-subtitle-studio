"""ETA estimates learned from this machine's own finished jobs.

The history already lives on this machine: every completed job stores its `stats` in the local SQLite
database, including `stats["stages"][stage]["real_time_factor"]` (processing seconds per second of
media), `media_duration_s` and `total_elapsed_s`. This module only turns those numbers into estimates:

- per-stage ETA = median real-time factor of that stage x media length x share left;
- whole-job ETA = the running stage plus the stages after it;
- pending-job ETA = media length x median processing seconds per second of media, falling back to the
  median job length when the media length is unknown (queued URLs).

Nothing here changes pipeline output, and no input path, subtitle text, key or log line is stored:
only durations and seconds.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import av

from app.core.pipeline import STAGES

log = logging.getLogger(__name__)

UNKNOWN = "--:--"          # shown when there is no history to estimate from yet
MAX_HISTORY_JOBS = 20      # finished jobs used for the medians


@dataclass(frozen=True)
class Timings:
    """Medians over the last finished jobs of this machine."""

    stage_rtf: dict[str, float] = field(default_factory=dict)       # stage -> seconds per media second
    stage_seconds: dict[str, float] = field(default_factory=dict)   # stage -> whole-stage seconds
    job_ratio: float | None = None      # processing seconds per second of media
    job_seconds: float | None = None    # length of a whole job when the media length is unknown
    samples: int = 0                    # finished jobs the medians come from


EMPTY_TIMINGS = Timings()


def timings_from_stats(stats_list: list[dict]) -> Timings:
    """Build the timing history from the `stats` of finished jobs (cached and skipped stages ignored)."""
    rtfs: dict[str, list[float]] = {}
    elapsed: dict[str, list[float]] = {}
    ratios: list[float] = []
    totals: list[float] = []
    samples = 0
    for stats in stats_list:
        if not isinstance(stats, dict):
            continue
        used = False
        for stage, entry in (stats.get("stages") or {}).items():
            if not isinstance(entry, dict) or "skipped" in entry or entry.get("cached"):
                continue
            rtf = entry.get("real_time_factor")
            if isinstance(rtf, (int, float)) and rtf > 0:
                rtfs.setdefault(stage, []).append(float(rtf))
                used = True
            seconds = entry.get("elapsed_s")
            if isinstance(seconds, (int, float)) and seconds > 0:
                elapsed.setdefault(stage, []).append(float(seconds))
                used = True
        media = stats.get("media_duration_s")
        total = stats.get("total_elapsed_s")
        if isinstance(media, (int, float)) and isinstance(total, (int, float)) and media > 0 and total > 0:
            ratios.append(float(total) / float(media))
            totals.append(float(total))
            used = True
        if used:
            samples += 1
    return Timings(
        stage_rtf={name: statistics.median(values) for name, values in rtfs.items()},
        stage_seconds={name: statistics.median(values) for name, values in elapsed.items()},
        job_ratio=statistics.median(ratios) if ratios else None,
        job_seconds=statistics.median(totals) if totals else None,
        samples=samples,
    )


def _share_left(fraction: float) -> float:
    return 1.0 - min(max(fraction, 0.0), 1.0)


def stage_remaining(timings: Timings, stage: str, duration: float | None, fraction: float) -> float | None:
    """Seconds left in one stage, or None when this machine has no timing for it yet."""
    rtf = timings.stage_rtf.get(stage)
    if rtf is None or not duration:
        return None
    return max(0.0, rtf * float(duration) * _share_left(fraction))


def job_remaining(timings: Timings, stage: str, duration: float | None, fraction: float) -> float | None:
    """Seconds left in the running job: the rest of this stage plus every stage after it."""
    if timings.samples == 0 or stage not in STAGES:
        return None
    seconds = 0.0
    known = False
    current = STAGES.index(stage)
    length = float(duration) if duration else 0.0
    if length and timings.stage_rtf.get(stage) is not None:
        seconds += timings.stage_rtf[stage] * length * _share_left(fraction)
        known = True
    elif timings.stage_seconds.get(stage) is not None:
        seconds += timings.stage_seconds[stage] * _share_left(fraction)
        known = True
    for name in STAGES[current + 1:]:
        if length and timings.stage_rtf.get(name) is not None:
            seconds += timings.stage_rtf[name] * length
            known = True
        elif timings.stage_seconds.get(name) is not None:
            seconds += timings.stage_seconds[name]
            known = True
    return seconds if known else None


def pending_job_seconds(timings: Timings, duration: float | None) -> float | None:
    """Estimated length of a job that has not started, from the media length when it is known."""
    if duration and timings.job_ratio:
        return float(duration) * timings.job_ratio
    return timings.job_seconds


def format_eta(seconds: float) -> str:
    """Human readable duration, always at least mm:ss."""
    total = max(0, int(round(seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def media_duration(path: Path) -> float | None:
    """Length of a media file in seconds, or None when it cannot be read."""
    try:
        with av.open(str(path)) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0
            for stream in container.streams:
                if stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
    except Exception:                     # unreadable or corrupt file: estimate without it
        log.debug("Could not read the duration of %s", path, exc_info=True)
    return None

