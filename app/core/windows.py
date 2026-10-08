"""Transcription windows of about ten minutes, cut at the longest silence near each boundary (D-092).

Whisper is fed one window at a time, so memory stays bounded and a resumed job restarts inside one window. A cut in
the middle of a sentence would damage both halves, so each boundary moves to the middle of the longest pause (found
with the Silero VAD) within +-45 s of the ideal position.
"""

from __future__ import annotations

import numpy as np

from app.core.audio_processor import SAMPLE_RATE

WINDOW_S = 600.0
SEARCH_S = 45.0
MIN_TAIL_S = 60.0


def _longest_pause(audio: np.ndarray, lo: int, hi: int, threshold: float) -> int | None:
    """Sample index in the middle of the longest speech-free stretch of audio[lo:hi], or None."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    speech = get_speech_timestamps(audio[lo:hi], VadOptions(threshold=threshold))
    edges = [0] + [p for s in speech for p in (s["start"], s["end"])] + [hi - lo]
    gaps = [(edges[i + 1] - edges[i], edges[i]) for i in range(0, len(edges), 2)]
    length, start = max(gaps)
    return lo + start + length // 2 if length >= SAMPLE_RATE // 5 else None


def plan_windows(audio: np.ndarray, vad_threshold: float = 0.35, window_s: float = WINDOW_S) -> list[tuple[float, float]]:
    """[(start_s, end_s), ...] covering the whole audio."""
    total = len(audio)
    size, search = int(window_s * SAMPLE_RATE), int(SEARCH_S * SAMPLE_RATE)
    cuts = [0]
    while total - cuts[-1] > size * 1.5 or (total - cuts[-1] > size + MIN_TAIL_S * SAMPLE_RATE):
        ideal = cuts[-1] + size
        cut = _longest_pause(audio, max(cuts[-1] + search, ideal - search), min(total, ideal + search), vad_threshold)
        cuts.append(cut if cut is not None else ideal)
    cuts.append(total)
    return [(round(a / SAMPLE_RATE, 3), round(b / SAMPLE_RATE, 3)) for a, b in zip(cuts, cuts[1:])]
