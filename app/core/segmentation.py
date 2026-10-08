"""Subtitle cue segmentation from ASR segments.

Every cue comes from exactly one ASR segment, so separate utterances are never merged into one
subtitle. ASR segments that are too long to read as one subtitle are split with word timestamps,
preferring sentence ends, then clause punctuation, then pauses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.core.translation import TranslationUnit

MAX_CUE_CHARS = 84          # two lines of 42 characters (provisional, see DECISIONS)
MAX_CUE_SECONDS = 6.5
_SENTENCE_END = (".", "!", "?", "\u2026", "\u061f")
_CLAUSE_END = (",", ";", ":", "\u060c")
_PAUSE_S = 0.35


@dataclass
class _Word:
    start: float
    end: float
    text: str


def _fits(words: list[_Word]) -> bool:
    text = "".join(w.text for w in words).strip()
    return len(text) <= MAX_CUE_CHARS and words[-1].end - words[0].start <= MAX_CUE_SECONDS


def _best_split(words: list[_Word]) -> int:
    """Index after which to split (1..len-1), scored by boundary strength and balance."""
    best_index, best_score = len(words) // 2, -1.0
    total = len("".join(w.text for w in words))
    running = 0
    for i in range(1, len(words)):
        running += len(words[i - 1].text)
        left = words[i - 1].text.strip()
        gap = words[i].start - words[i - 1].end
        strength = 3.0 if left.endswith(_SENTENCE_END) else 2.0 if left.endswith(_CLAUSE_END) else (
            1.0 if gap >= _PAUSE_S else 0.0)
        balance = 1.0 - abs(running / total - 0.5) * 2      # 1 = perfectly balanced
        score = strength + balance
        if score > best_score:
            best_index, best_score = i, score
    return best_index


_REJOIN_GAP_S = 1.5


def _split(words: list[_Word]) -> list[list[_Word]]:
    if len(words) < 2 or _fits(words):
        return [words]
    index = _best_split(words)
    return _split(words[:index]) + _split(words[index:])


def _rejoin(chunks: list[list[_Word]]) -> list[list[_Word]]:
    """Recursive splitting can leave fragments such as a lone word; join neighbours again when the
    result still fits one subtitle and they are spoken close together."""
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        candidate = merged[-1] + chunk
        if chunk[0].start - merged[-1][-1].end <= _REJOIN_GAP_S and _fits(candidate):
            merged[-1] = candidate
        else:
            merged.append(chunk)
    return merged


# Part of the refine cache key: refine results are stored per unit id, so new unit boundaries must not reuse them.
SEGMENTATION_VERSION = 1


def segment_cues(segments: Sequence[dict]) -> list[TranslationUnit]:
    units: list[TranslationUnit] = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        words = [_Word(w["start"], w["end"], w["text"]) for w in seg.get("words") or [] if w["text"].strip()]
        if not words:
            units.append(TranslationUnit(len(units), [seg["id"]], seg["start"], seg["end"], text))
            continue
        for chunk in _rejoin(_split(words)):
            chunk_text = " ".join("".join(w.text for w in chunk).split())
            start = seg["start"] if chunk is words or chunk[0] is words[0] else chunk[0].start
            end = seg["end"] if chunk[-1] is words[-1] else chunk[-1].end
            units.append(TranslationUnit(len(units), [seg["id"]], round(start, 3), round(end, 3), chunk_text))
    return units
