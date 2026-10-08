"""Reading third-party subtitle files (SRT, WebVTT incl. YouTube auto-caption VTT) and text normalisation."""

from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable

from app.core.exporter import Cue, parse_srt

_VTT_TIME = r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
_VTT_ARROW = re.compile(rf"^\s*{_VTT_TIME}\s*-->\s*{_VTT_TIME}")
_TAG = re.compile(r"<[^>]*>")
_ROLLING_MAX_S = 0.05


def _seconds(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_vtt(text: str) -> list[Cue]:
    """Line-based WebVTT parser. Whitespace-only lines inside a cue (common in YouTube captions) do not end
    the cue; only a truly empty line does."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    body: list[str] | None = None
    for line in text.split("\n"):
        match = _VTT_ARROW.match(line)
        if match:
            g = match.groups()
            body = []
            cues.append(Cue(_seconds(*g[:4]), _seconds(*g[4:]), ""))
            continue
        if body is None:
            continue
        if line == "":
            cues[-1].text = "\n".join(body)
            body = None
            continue
        clean = html.unescape(_TAG.sub("", line)).strip()
        if clean:
            body.append(clean)
    if body is not None and cues:
        cues[-1].text = "\n".join(body)
    return cues


def collapse_rolling_captions(cues: list[Cue]) -> list[Cue]:
    """Undo YouTube's rolling auto-caption layout, where each cue repeats the previous line.

    Only for automatic captions: manual subtitles may legitimately repeat a line.
    """
    out: list[Cue] = []
    recent: list[str] = []
    for cue in cues:
        lines = [line for line in cue.text.split("\n") if line]
        new = [line for line in lines if line not in recent]
        if not new:
            if out and out[-1].text.split("\n")[-1] in lines:
                out[-1].end = max(out[-1].end, cue.end)
            continue
        if cue.end - cue.start < _ROLLING_MAX_S and out:
            continue
        out.append(Cue(cue.start, cue.end, "\n".join(new)))
        recent = lines[-2:]
    return [c for c in out if c.text.strip()]


def parse_subtitle_text(text: str, rolling: bool = False) -> list[Cue]:
    """Detect SRT vs WebVTT and parse."""
    head = text.lstrip("\ufeff").lstrip()[:10].upper()
    cues = parse_vtt(text) if head.startswith("WEBVTT") else parse_srt(text)
    return collapse_rolling_captions(cues) if rolling else cues


def normalize_text(text: str) -> str:
    """Comparison form: no diacritics, no punctuation, case-folded, single spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    cleaned = "".join(c if c.isalnum() else " " for c in stripped.casefold())
    return " ".join(cleaned.split())


def text_similarity(a: str, b: str) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    # Character ratio alone rates unrelated text of similar length at ~0.3-0.4; the word ratio is ~0 for
    # unrelated text. The mean keeps spelling variants high and unrelated text low.
    chars = SequenceMatcher(None, na, nb, autojunk=False).ratio()
    words = SequenceMatcher(None, na.split(), nb.split(), autojunk=False).ratio()
    return (chars + words) / 2


def audio_agreement(segments: Iterable[dict], cues: list[Cue]) -> float | None:
    """Duration-weighted text similarity between ASR segments and time-overlapping cues.

    Only meaningful when both are in the same language. A rough first measure; proper alignment
    (allowing time offsets) is milestone M3.
    """
    total_weight = 0.0
    score = 0.0
    for seg in segments:
        duration = max(seg["end"] - seg["start"], 0.01)
        overlapping = [c.text for c in cues if c.start < seg["end"] and c.end > seg["start"]]
        similarity = text_similarity(seg["text"], " ".join(overlapping)) if overlapping else 0.0
        score += similarity * duration
        total_weight += duration
    return round(score / total_weight, 4) if total_weight else None


_LEGACY_ENCODINGS = {"tr": "cp1254", "ar": "cp1256", "fa": "cp1256", "ru": "cp1251", "uk": "cp1251"}


def decode_subtitle_bytes(data: bytes, language: str | None = None) -> str:
    """Decode subtitle files from sites that still serve legacy Windows code pages."""
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            text = data.decode(encoding)
            if encoding == "utf-16" and data[:2] not in (b"\xff\xfe", b"\xfe\xff"):
                continue
            return text
        except UnicodeDecodeError:
            continue
    legacy = _LEGACY_ENCODINGS.get((language or "").split("-")[0].lower())
    if legacy:
        try:
            return data.decode(legacy)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1")
