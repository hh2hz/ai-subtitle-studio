"""SRT reading/writing and output naming."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from app.utils.atomic import atomic_write_bytes

# Provisional line length until the subtitle QA defaults are researched in M4.
DEFAULT_MAX_LINE_CHARS = 42
DEFAULT_MAX_LINES = 2

_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})")
_ARROW = re.compile(rf"^\s*{_TIME.pattern}\s*-->\s*{_TIME.pattern}")
_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


@dataclass
class Cue:
    start: float
    end: float
    text: str
    uid: int | None = None          # id of the translation unit the cue came from; survives timing changes (D-097)
    speaker: str | None = None      # voice id of the unit (D-098); merging is only allowed inside one speaker
    absorbed: list[int] | None = None   # uids of cues merged into this one by polish_timing (D-100)


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def parse_timestamp(text: str) -> float:
    match = _TIME.fullmatch(text.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {text!r}")
    h, m, s, ms = (int(g) for g in match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


_BREAK_AFTER = ".,;:!?\u2026\u060c\u061b\u061f"
PUNCT_BONUS = 8.0          # a break after punctuation is worth this many characters of imbalance
CONJUNCTION_BONUS = 5.0    # so is a break before a conjunction or preposition
BOTTOM_HEAVY = 3.0         # a longer first line costs this much extra
FORBIDDEN_END = 1000.0     # never end a line with an article, preposition or conjunction when avoidable


def _word(token: str) -> str:
    return token.strip(".,;:!?\u2026\u060c\u061b\u061f\"'()-").casefold()


def wrap_text(text: str, max_chars: int = DEFAULT_MAX_LINE_CHARS, max_lines: int = DEFAULT_MAX_LINES,
              language: str | None = None) -> str:
    """Split into at most max_lines lines at spaces. Never drops words; a line may exceed max_chars when the text
    cannot fit (QA reports that later).

    An intentional line break (for example a dialogue cue with one dash line per speaker) is kept when the text has
    at most max_lines lines. Otherwise the break point of a two-line cue is chosen by, in this order: fewest
    characters beyond max_chars; then a cost that rewards a break after punctuation and before a conjunction or
    preposition, forbids ending a line with a word of the language's "no_end" list (D-101) and prefers a longer
    second line (bottom-heavy)."""
    if "\n" in text:
        lines = [" ".join(line.split()) for line in text.split("\n") if line.strip()]
        if 1 < len(lines) <= max_lines:
            return "\n".join(lines)
    text = " ".join(text.split())
    if len(text) <= max_chars or max_lines < 2:
        return text
    words = text.split(" ")
    if len(words) < 2:
        return text
    from app.core import style

    no_end, break_before = style.break_words(language)
    best = None
    for i in range(1, len(words)):
        first, second = " ".join(words[:i]), " ".join(words[i:])
        overflow = max(0, len(first) - max_chars) + max(0, len(second) - max_chars)
        cost = abs(len(first) - len(second)) + (BOTTOM_HEAVY if len(first) > len(second) else 0.0)
        if words[i - 1][-1] in _BREAK_AFTER:
            cost -= PUNCT_BONUS
        elif _word(words[i]) in break_before:
            cost -= CONJUNCTION_BONUS
        if _word(words[i - 1]) in no_end:
            cost += FORBIDDEN_END
        if best is None or (overflow, cost) < best[0]:
            best = ((overflow, cost), first, second)
    return f"{best[1]}\n{best[2]}"


def to_srt(cues: list[Cue]) -> str:
    blocks = []
    for index, cue in enumerate(cues, start=1):
        text = cue.text.replace("\r\n", "\n").strip()
        blocks.append(f"{index}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{text}\n")
    return "\n".join(blocks)


def write_srt(cues: list[Cue], path: Path, bom: bool = True) -> None:
    """Write UTF-8 SRT with CRLF line endings. The BOM helps some Windows players detect UTF-8."""
    data = to_srt(cues).replace("\n", "\r\n").encode("utf-8")
    atomic_write_bytes(path, (b"\xef\xbb\xbf" if bom else b"") + data)


def parse_srt(text: str) -> list[Cue]:
    """Lenient parser: tolerates BOM, CRLF, missing or wrong indices and extra blank lines."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        match = _ARROW.match(lines[i])
        if not match:
            i += 1
            continue
        g = match.groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
        i += 1
        body = []
        while i < len(lines) and lines[i].strip():
            # A numeric line directly followed by a timing line starts the next cue.
            if lines[i].strip().isdigit() and i + 1 < len(lines) and _ARROW.match(lines[i + 1]):
                break
            body.append(lines[i].rstrip())
            i += 1
        cues.append(Cue(start, end, "\n".join(body)))
    return cues


def read_srt(path: Path) -> list[Cue]:
    return parse_srt(Path(path).read_bytes().decode("utf-8-sig"))


def fix_overlaps(cues: list[Cue]) -> list[Cue]:
    """Clip each cue so it ends no later than the next cue starts."""
    fixed = [replace(c) for c in sorted(cues, key=lambda c: c.start)]
    for current, nxt in zip(fixed, fixed[1:]):
        if current.end > nxt.start:
            current.end = max(current.start, nxt.start)
    return fixed


_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def safe_basename(name: str, fallback: str = "Episode", max_length: int = 100) -> str:
    """File-name-safe text. Windows reserved device names get a suffix; long titles are cut (paths must stay
    below the 260-character Windows limit: folder + file + ".ar.subtitled.mp4" + yt-dlp temporary suffixes)."""
    cleaned = " ".join(_UNSAFE_NAME.sub("_", name).split()).strip(" ._")
    cleaned = cleaned[:max_length].rstrip(" ._") or fallback
    if cleaned.split(".")[0].lower() in _RESERVED:
        cleaned += "_"
    return cleaned


RLM = "\u200f"
MIN_VISIBLE_S = 0.3        # shortest cue polish_timing may produce


def mark_rtl(text: str) -> str:
    """Wrap every line in RIGHT-TO-LEFT MARKs so players lay out punctuation of RTL lines correctly.

    A leading mark alone is not enough: VLC lays SRT lines out left-to-right, so a final period jumped to the right
    end of the line (seen by the user, reproduced with VLC 3.0.20; D-039). With a mark at both ends the trailing
    punctuation is enclosed by strong RTL characters and stays at the left end."""
    return "\n".join(RLM + line.strip(RLM) + RLM for line in text.split("\n"))


PULL_START_S = 0.5         # a cue may appear this much earlier to become readable
MERGE_GAP_S = 1.0          # cues further apart than this are never merged
MERGE_SHORT_CHARS = 30     # only a short neighbour is merged into a too-fast cue


def _flat(text: str) -> str:
    return " ".join(text.replace(RLM, "").split())


def _cps(cue: Cue) -> float:
    duration = cue.end - cue.start
    return len(_flat(cue.text)) / duration if duration > 0 else float("inf")


def _merged(first: Cue, second: Cue, max_line_chars: int, max_lines: int, language: str | None = None) -> Cue:
    return Cue(first.start, second.end, wrap_text(_flat(first.text) + " " + _flat(second.text), max_line_chars,
                                                   max_lines, language),
               first.uid, first.speaker, (first.absorbed or []) + [second.uid] + (second.absorbed or []))


def _best_merge(result: list[Cue], i: int, max_duration: float, max_cps: float, max_line_chars: int,
                max_lines: int, language: str | None = None) -> tuple[int, Cue] | None:
    """The neighbour (next or previous) whose merge with cue i gives a readable two-line cue, or None."""
    best = None
    for j in (i + 1, i - 1):
        if not 0 <= j < len(result):
            continue
        a, b = (result[i], result[j]) if j > i else (result[j], result[i])
        other = result[j]
        if other.speaker != result[i].speaker or len(_flat(other.text)) > MERGE_SHORT_CHARS \
                or b.start - a.end > MERGE_GAP_S:
            continue
        merged = _merged(a, b, max_line_chars, max_lines, language)
        lines = merged.text.split("\n")
        if len(lines) > max_lines or any(len(line) > max_line_chars for line in lines) \
                or merged.end - merged.start > max_duration or _cps(merged) > max_cps:
            continue
        if best is None or _cps(merged) < best[2]:
            best = (j, merged, _cps(merged))
    return (best[0], best[1]) if best else None


def polish_timing(cues: list[Cue], min_duration: float = 0.833, min_gap: float = 0.083,
                  max_duration: float = 7.0, max_cps: float = 20.0, fix_speed: bool = False,
                  max_line_chars: int = DEFAULT_MAX_LINE_CHARS, max_lines: int = DEFAULT_MAX_LINES,
                  language: str | None = None) -> list[Cue]:
    """Professional timing defaults (see DECISIONS): each cue is extended into the following free time until it
    is readable (at least min_duration and at most max_cps characters per second), keeping min_gap before the
    next cue and never exceeding max_duration. Starts are never moved; cues are never shortened except to
    remove overlaps or cap max_duration.

    With fix_speed, a cue that is still above max_cps after the extension is repaired in this order (D-100):
    its start moves up to 0.5 s earlier into the gap before the previous cue (keeping min_gap); then it is merged
    with a short neighbouring cue of the same speaker when the merged text fits max_lines lines, the merged cue
    stays within max_duration and is readable; otherwise it stays as it is and QA flags it."""
    result = [replace(c, end=min(c.end, c.start + max_duration)) for c in sorted(cues, key=lambda c: c.start)]
    i = 0
    while i < len(result):
        cue = result[i]
        nxt = result[i + 1] if i + 1 < len(result) else None
        if nxt is not None and nxt.start < cue.start + MIN_VISIBLE_S + min_gap:
            # Two cues starting almost together: delay the next one slightly instead of shrinking this one to
            # a few milliseconds that never show on screen.
            nxt.start = round(cue.start + MIN_VISIBLE_S + min_gap, 3)
            nxt.end = max(nxt.end, nxt.start + MIN_VISIBLE_S)
        chars = len(_flat(cue.text))
        wanted = min(max(min_duration, chars / max_cps if max_cps else 0.0), max_duration)
        limit = result[i + 1].start - min_gap if i + 1 < len(result) else cue.start + max_duration
        if cue.end - cue.start < wanted:
            cue.end = max(cue.end, min(cue.start + wanted, limit))
        if i + 1 < len(result) and cue.end > result[i + 1].start - min_gap:
            cue.end = max(cue.start + 0.001, min(cue.end, result[i + 1].start - min_gap))
        if fix_speed and max_cps and _cps(cue) > max_cps + 1e-6:
            room = cue.start - (result[i - 1].end + min_gap if i else 0.0)
            needed = chars / max_cps - (cue.end - cue.start)
            take = min(PULL_START_S, max(0.0, room), max(0.0, needed))
            if take > 0:
                cue.start = round(cue.start - take, 3)
            if _cps(cue) > max_cps + 1e-6:
                found = _best_merge(result, i, max_duration, max_cps, max_line_chars, max_lines, language)
                if found:
                    j, merged = found
                    lo = min(i, j)
                    result[lo:lo + 2] = [merged]
                    i = max(lo - 1, 0)           # the merged cue gets the normal treatment again
                    continue
        i += 1
    return result


def output_paths(output_dir: Path, basename: str, target_language: str, source_language: str = "src") -> dict[str, Path]:
    """Episode folder layout: the final subtitle next to the video (same base name, so players load it
    automatically) and everything else in work/."""
    base = safe_basename(basename)
    work = output_dir / "work"
    return {
        "target_srt": output_dir / f"{base}.{target_language}.srt",
        "source_srt": work / f"{base}.{source_language}.source.srt",
        "master_transcript": work / "MasterTranscript.json",
        "review_required": work / "ReviewRequired.txt",
        "processing_log": work / "ProcessingLog.txt",
    }
