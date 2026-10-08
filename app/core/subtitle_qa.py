"""Technical subtitle QA on final cues. Limits are configurable; defaults follow the Netflix Arabic Timed Text
Style Guide (42 characters per line, 2 lines, 20 characters per second for adult programs) and Netflix's general
timing practice (minimum 5/6 s, maximum 7 s). Values verified 2026-10-03 (see DECISIONS D-028)."""

from __future__ import annotations

from dataclasses import dataclass

from app.core import style
from app.core.exporter import Cue

INVALID_TIMING = "invalid_timing"
OVERLAP = "overlap"
TOO_SHORT = "too_short"
TOO_LONG = "too_long"
READING_SPEED = "reading_speed"
LINE_TOO_LONG = "line_too_long"
TOO_MANY_LINES = "too_many_lines"
EMPTY = "empty_cue"
DUPLICATE = "duplicate_cue"

#: Every flag code this module can emit (D-106).
FLAGS = frozenset({INVALID_TIMING, OVERLAP, TOO_SHORT, TOO_LONG, READING_SPEED, LINE_TOO_LONG, TOO_MANY_LINES,
                   EMPTY, DUPLICATE})


@dataclass(frozen=True)
class QaLimits:
    max_line_chars: int = 42
    max_lines: int = 2
    max_cps: float = 20.0
    min_duration: float = 0.833
    max_duration: float = 7.0
    duplicate_gap: float = 0.5

    @classmethod
    def for_language(cls, language: str | None) -> "QaLimits":
        """Characters per line, characters per second and minimum duration of the target language's style guide
        (app/resources/style_guides.json, D-100); unknown languages get the generic guide."""
        return cls(max_line_chars=style.max_line(language), max_cps=style.cps(language),
                   min_duration=style.min_duration(language))


def _visible(text: str) -> str:
    return text.replace("\u200f", "").replace("\u200e", "")


def check_cues(cues: list[Cue], limits: QaLimits = QaLimits()) -> list[list[str]]:
    """Flags per cue, in input order."""
    results: list[list[str]] = []
    previous: Cue | None = None
    for cue in cues:
        flags = []
        text = _visible(cue.text).strip()
        duration = cue.end - cue.start
        lines = [line for line in text.split("\n") if line.strip()]
        if duration <= 0:
            flags.append(INVALID_TIMING)
        if previous is not None and cue.start < previous.end - 1e-6:
            flags.append(OVERLAP)
        if 0 < duration < limits.min_duration - 1e-6:
            flags.append(TOO_SHORT)
        if duration > limits.max_duration + 1e-6:
            flags.append(TOO_LONG)
        if not text:
            flags.append(EMPTY)
        else:
            chars = len(" ".join(lines))
            if duration > 0 and chars / duration > limits.max_cps:
                flags.append(READING_SPEED)
            if any(len(line) > limits.max_line_chars for line in lines):
                flags.append(LINE_TOO_LONG)
            if len(lines) > limits.max_lines:
                flags.append(TOO_MANY_LINES)
            if previous is not None and _visible(previous.text).strip() == text \
                    and cue.start - previous.end <= limits.duplicate_gap:
                flags.append(DUPLICATE)
        results.append(flags)
        previous = cue
    return results
