"""Master transcript: the per-segment source of truth for the rest of the pipeline.

In M1 the only evidence is independent ASR, so agreement is unknown (None) and confidence equals
audio confidence. Later milestones add subtitle sources, agreement and verification.
"""

from __future__ import annotations

import math
from typing import Iterable

from app.core.transcription import AsrSegment

SCHEMA_VERSION = 1

SEGMENT_FIELDS = (
    "id", "start", "end", "speaker", "text", "source_language", "confidence", "sources",
    "agreement", "audio_confidence", "verification_status",
)


def audio_confidence(segment: AsrSegment) -> float | None:
    """Mean word probability; falls back to exp(avg_logprob) when no word timestamps exist."""
    known = [w.probability for w in segment.words if w.probability is not None]
    if known:
        return round(sum(known) / len(known), 4)
    if segment.avg_logprob is not None:
        return round(math.exp(segment.avg_logprob), 4)
    return None


def build_segments(asr_segments: Iterable[AsrSegment], language: str, engine_names: Iterable[str]) -> list[dict]:
    result = []
    for index, (seg, engine) in enumerate(zip(asr_segments, engine_names)):
        conf = audio_confidence(seg)
        result.append({
            "id": index,
            "start": seg.start,
            "end": seg.end,
            "speaker": None,
            "text": seg.text,
            "source_language": language,
            "confidence": conf,
            "sources": [{"type": "asr", "engine": engine}],
            "agreement": None,
            "audio_confidence": conf,
            "verification_status": "unverified",
            "words": [w.__dict__ for w in seg.words],
            "asr": {
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
                "compression_ratio": seg.compression_ratio,
            },
        })
        if seg.redecoded:
            result[-1]["redecoded"] = True
    return result


def make_document(segments: list[dict], *, language: str, language_probability: float | None,
                  duration: float, job: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "job": job,
        "media": {"duration": round(duration, 3)},
        "language": {"code": language, "probability": language_probability},
        "segments": segments,
    }


def validate_document(doc: dict) -> list[str]:
    """Return a list of structural problems (empty when valid)."""
    problems = []
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version mismatch")
    previous_start = -1.0
    for i, seg in enumerate(doc.get("segments", [])):
        missing = [f for f in SEGMENT_FIELDS if f not in seg]
        if missing:
            problems.append(f"segment {i}: missing {missing}")
            continue
        if seg["id"] != i:
            problems.append(f"segment {i}: id {seg['id']} out of sequence")
        if not (0 <= seg["start"] <= seg["end"]):
            problems.append(f"segment {i}: invalid time range {seg['start']}..{seg['end']}")
        if seg["start"] < previous_start:
            problems.append(f"segment {i}: starts before previous segment")
        previous_start = seg["start"]
        conf = seg["confidence"]
        if conf is not None and not (0.0 <= conf <= 1.0):
            problems.append(f"segment {i}: confidence {conf} outside [0, 1]")
    return problems
