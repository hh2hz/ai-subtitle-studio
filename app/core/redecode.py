"""Second decode of low-confidence segments with a wider beam and sampling fallbacks (D-091).

A segment is low-confidence when its mean word probability is below 0.5 or it contains a run of at least three
words below 0.3. Its span (+-1 s) is decoded again, on the transcription audio and on the other audio variant when
there is one; the candidate with the best avg_logprob wins, provided it beats the original, has a compression ratio
of at most 2.4 and passes the hallucination filter. At most 20 % of the duration is re-decoded, worst first.
Finished spans are appended to a partial file, so a killed job resumes without decoding them again.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

from app.core import hallucination
from app.core.audio_processor import SAMPLE_RATE
from app.core.transcription import AsrSegment
from app.utils.atomic import JsonlWriter, load_jsonl

log = logging.getLogger(__name__)

MEAN_BELOW = 0.5
RUN_BELOW = 0.3
RUN_LENGTH = 3
PAD_S = 1.0
CAP_FRACTION = 0.2
MAX_COMPRESSION = 2.4


def _probabilities(segment: dict) -> list[float]:
    return [w["probability"] for w in segment.get("words", []) if w.get("probability") is not None]


def is_low_confidence(segment: dict) -> bool:
    probs = _probabilities(segment)
    if not probs:
        return False
    if sum(probs) / len(probs) < MEAN_BELOW:
        return True
    run = 0
    for p in probs:
        run = run + 1 if p < RUN_BELOW else 0
        if run >= RUN_LENGTH:
            return True
    return False


def select(segments: Sequence[dict], duration: float) -> list[int]:
    """Indexes of the segments to decode again: lowest mean probability first, within the duration cap."""
    scored = []
    for index, seg in enumerate(segments):
        if is_low_confidence(seg):
            probs = _probabilities(seg)
            scored.append((sum(probs) / len(probs), index))
    budget = CAP_FRACTION * duration
    chosen: list[int] = []
    for _mean, index in sorted(scored):
        cost = segments[index]["end"] - segments[index]["start"] + 2 * PAD_S
        if cost > budget:
            continue
        budget -= cost
        chosen.append(index)
    return chosen


def merge_candidate(found: Sequence[AsrSegment], start: float, end: float) -> dict | None:
    """One segment from the decoded pieces whose middle lies inside the original segment."""
    inside = [s for s in found if start <= (s.start + s.end) / 2 <= end and s.text.strip()]
    if not inside:
        return None
    weights = [max(s.end - s.start, 0.01) for s in inside]
    logprobs = [(s.avg_logprob, w) for s, w in zip(inside, weights) if s.avg_logprob is not None]
    return AsrSegment(
        start=min(s.start for s in inside), end=max(s.end for s in inside),
        text=" ".join(s.text.strip() for s in inside),
        words=[w for s in inside for w in s.words],
        avg_logprob=round(sum(v * w for v, w in logprobs) / sum(w for _, w in logprobs), 4) if logprobs else None,
        no_speech_prob=max((s.no_speech_prob for s in inside if s.no_speech_prob is not None), default=None),
        compression_ratio=max((s.compression_ratio for s in inside if s.compression_ratio is not None), default=None),
        redecoded=True,
    ).to_dict()


def pick(original: dict, candidates: Sequence[dict]) -> dict | None:
    """The valid candidate with the best avg_logprob that beats the original, or None."""
    floor = original.get("avg_logprob")
    best = None
    for cand in candidates:
        logprob = cand.get("avg_logprob")
        if logprob is None or (floor is not None and logprob <= floor):
            continue
        ratio = cand.get("compression_ratio")
        if ratio is not None and ratio > MAX_COMPRESSION:
            continue
        if hallucination.reason(cand):
            continue
        if best is None or logprob > best["avg_logprob"]:
            best = cand
    return best


def redecode(engine, sources: Sequence, records: list[dict], language: str, duration: float, partial: Path,
             check_cancel: Callable[[], None], progress: Callable[[float], None] | None = None) -> dict:
    """Replace low-confidence records in place. `sources` are audio arrays or loaders (the transcription audio first).
    Returns {"selected", "replaced", "spans": [{start, end, old, new}]}."""
    span = getattr(engine, "transcribe_span", None)
    indexes = select([r["segment"] for r in records], duration) if span else []
    done = {r["index"]: r for r in load_jsonl(partial)}
    changes = []
    loaded = None
    with JsonlWriter(partial) as writer:
        for n, index in enumerate(indexes):
            check_cancel()
            seg = records[index]["segment"]
            key = round(seg["start"], 3)
            entry = done.get(index)
            if entry is None or entry.get("start") != key:
                a, b = max(0.0, seg["start"] - PAD_S), min(duration, seg["end"] + PAD_S)
                candidates = []
                if loaded is None:                      # sources may be loaders (the original audio is on disk)
                    loaded = [src() if callable(src) else src for src in sources]
                for audio in loaded:
                    piece = audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)]
                    cand = merge_candidate(span(piece, language, a), seg["start"], seg["end"])
                    if cand:
                        candidates.append(cand)
                chosen = pick(seg, candidates)
                entry = {"index": index, "start": key, "segment": chosen}
                writer.write(entry)
            if entry["segment"]:
                changes.append({"start": seg["start"], "end": seg["end"], "old": seg["text"],
                                "new": entry["segment"]["text"]})
                records[index] = {**records[index], "segment": entry["segment"]}
            if progress:
                progress((n + 1) / len(indexes))
    log.info("Re-decoded %d low-confidence segments, %d replaced", len(indexes), len(changes))
    return {"selected": len(indexes), "replaced": len(changes), "spans": changes}
