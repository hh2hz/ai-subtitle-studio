"""Use the platform's own manual subtitles to complete the ASR text (D-095).

Only tracks that are human-made ("manual"), in the source language, and that already agree with the audio
transcription are used; automatic captions never are. Each cue goes to the ASR segment it overlaps most. The ASR
timing stays; the text is replaced only when the subtitle text agrees with the ASR text and is longer (more complete).
The word list is rebuilt with estimated timings (probability None) so later splitting follows the new text.
"""

from __future__ import annotations

from app.core.subtitle_formats import normalize_text, text_similarity
from app.providers.base import MANUAL

SOURCE_MARK = "platform_subtitle"
MIN_LOCAL_SIMILARITY = 0.6


def _best_track(records: list[dict], language: str, match_from: float) -> dict | None:
    tracks = [r for r in records if r.get("kind") == MANUAL and r.get("language") == language and r.get("cues")
              and (r.get("audio_agreement") or 0.0) >= match_from]
    return max(tracks, key=lambda r: r["audio_agreement"], default=None)


def _estimated_words(text: str, start: float, end: float) -> list[dict]:
    tokens = text.split()
    total = sum(len(t) for t in tokens) or 1
    words, cursor = [], start
    for token in tokens:
        step = (end - start) * len(token) / total
        words.append({"start": round(cursor, 3), "end": round(cursor + step, 3), "text": token, "probability": None})
        cursor += step
    return words


def apply(doc: dict, records: list[dict], match_from: float) -> list[dict]:
    """Replace segment texts in `doc` in place; returns [{"id", "old", "new"}] for the log."""
    language = doc["language"]["code"]
    track = _best_track(records, language, match_from)
    if track is None:
        return []
    segments = doc["segments"]
    assigned: dict[int, list[str]] = {}
    for cue in sorted(track["cues"], key=lambda c: c["start"]):
        best, best_overlap = None, 0.0
        for index, seg in enumerate(segments):
            overlap = min(seg["end"], cue["end"]) - max(seg["start"], cue["start"])
            if overlap > best_overlap:
                best, best_overlap = index, overlap
        if best is not None and best_overlap >= 0.5 * (cue["end"] - cue["start"]):
            assigned.setdefault(best, []).append(" ".join(cue["text"].split()))
    changes = []
    for index, parts in assigned.items():
        seg = segments[index]
        candidate = " ".join(parts).strip()
        old = seg["text"]
        if (len(normalize_text(candidate)) <= len(normalize_text(old))
                or text_similarity(old, candidate) < MIN_LOCAL_SIMILARITY):
            continue
        seg["asr_text"] = old
        seg["text"] = candidate
        seg["words"] = _estimated_words(candidate, seg["start"], seg["end"])
        seg["source"] = SOURCE_MARK
        seg["sources"] = seg["sources"] + [{"type": SOURCE_MARK, "provider": track["provider"],
                                            "reference": track.get("reference")}]
        changes.append({"id": seg["id"], "old": old, "new": candidate})
    return changes
