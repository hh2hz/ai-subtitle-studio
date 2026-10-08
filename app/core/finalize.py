"""Final subtitle assembly shared by the pipeline export and the review window.

Applies language fixes, wrapping and timing polish, runs language and technical QA on the final cues, and
assigns each line a confidence category. Units are annotated in place with final_text, qa_flags, confidence.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.core import confidence, frames, style
from app.core.exporter import Cue, fix_overlaps, format_timestamp, mark_rtl, polish_timing, wrap_text
from app.core.language_qa import for_language
from app.core.subtitle_qa import QaLimits, check_cues
from app.utils.atomic import atomic_write_bytes
from app.utils.languages import RTL_LANGUAGES


def _audio_confidence(unit: dict, segments_by_id: dict[int, dict]) -> float | None:
    values = [segments_by_id[sid].get("audio_confidence") for sid in unit.get("segment_ids", [])
              if sid in segments_by_id]
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


unit_audio_confidence = _audio_confidence


def speaker_split(unit: dict, segments: dict[int, dict]) -> dict | None:
    """{"first", "second", "ratio"} when the words of the unit's cue belong to two speakers (word `speaker` labels
    set by diarize.label_words), else None. `ratio` is the share of the source characters said by the first one."""
    for sid in unit.get("segment_ids", []):
        seg = segments.get(sid) or {}
        words = [w for w in seg.get("words") or []
                 if w.get("speaker") and unit["start"] - 0.05 <= (w["start"] + w["end"]) / 2 <= unit["end"] + 0.05]
        order = [w["speaker"] for w in words]
        groups = [s for i, s in enumerate(order) if i == 0 or s != order[i - 1]]
        if len(groups) == 2:                           # more than two turns in one cue are left alone
            cut = next(i for i, s in enumerate(order) if s == groups[1])
            first = sum(len(w["text"].strip()) for w in words[:cut])
            total = sum(len(w["text"].strip()) for w in words) or 1
            return {"first": groups[0], "second": groups[1], "ratio": round(min(max(first / total, 0.15), 0.85), 3)}
    return None


_SPLIT_AFTER = ".!?\u2026,;:\u060c\u061f"


def dialogue_text(text: str, ratio: float, language: str | None) -> str:
    """Two dash lines from one translation: the break is the word boundary nearest to `ratio` of the characters,
    sentence punctuation counts as closer. Text that already has a line break or fewer than two words is kept."""
    words = text.split()
    if "\n" in text or len(words) < 2:
        return text
    target = ratio * len(" ".join(words))
    best, best_cost, used = 1, None, 0
    for i in range(1, len(words)):
        used = len(" ".join(words[:i]))
        cost = abs(used - target) - (6 if words[i - 1][-1] in _SPLIT_AFTER else 0)
        if best_cost is None or cost < best_cost:
            best, best_cost = i, cost
    dash = style.dash(language)
    return f"{dash}{' '.join(words[:best])}\n{dash}{' '.join(words[best:])}"


def finalize_units(units: list[dict], segments: list[dict], target_language: str,
                   limits: QaLimits | None = None, fps: float | None = None,
                   shots: list[float] | None = None) -> list[Cue]:
    """Return the final target cues (in unit order, empty lines dropped) and annotate the units. Limits default to
    the target language's style guide (D-100)."""
    limits = limits or QaLimits.for_language(target_language)
    lqa = for_language(target_language)
    segments_by_id = {s["id"]: s for s in segments}
    for unit in units:
        raw = unit.get("reviewed_text") if unit.get("reviewed_text") is not None else unit["translation"]
        unit["final_text"] = lqa.fix(raw)
        if unit.get("speaker_split") and unit.get("reviewed_text") is None and unit["final_text"].strip():
            unit["final_text"] = dialogue_text(unit["final_text"], unit["speaker_split"]["ratio"], target_language)
    kept = [u for u in units if u["final_text"].strip()]
    cues = polish_timing(
        fix_overlaps([Cue(u["start"], u["end"], wrap_text(u["final_text"], limits.max_line_chars, limits.max_lines, target_language),
                          u["id"], u.get("speaker")) for u in kept]),
        min_duration=limits.min_duration, max_duration=limits.max_duration, max_cps=limits.max_cps, fix_speed=True,
        max_line_chars=limits.max_line_chars, max_lines=limits.max_lines, language=target_language)
    if fps:
        cues = frames.snap_cues(cues, fps)
        if shots:
            cues = frames.snap_to_shots(cues, shots, fps, limits.min_duration)
    technical = check_cues(cues, limits)
    # Cues carry the id of their unit through sorting and timing, so QA results are paired by id, not by order.
    by_id = {u["id"]: u for u in kept}
    for cue, tech in zip(cues, technical):
        flags = lqa.check(cue.text) + tech
        by_id[cue.uid]["qa_flags"] = flags
        for uid in cue.absorbed or []:                # units merged into this cue share its result
            by_id[uid]["qa_flags"] = list(flags)
    for unit in units:
        if not unit["final_text"].strip():
            unit["qa_flags"] = ["empty_cue"]
        unit["audio_confidence"] = _audio_confidence(unit, segments_by_id)
        flags = ([] if unit.get("reviewed_text") is not None else list(unit.get("flags", []))) + unit["qa_flags"]
        unit["confidence"] = confidence.category(flags, unit["audio_confidence"])
    if target_language.split("-")[0] in RTL_LANGUAGES:
        cues = [replace(c, text=mark_rtl(c.text)) for c in cues]
    return cues


def source_cues(units: list[dict]) -> list[Cue]:
    return polish_timing(fix_overlaps([Cue(u["start"], u["end"], wrap_text(u["text"]), u["id"]) for u in units]))


LOW_AUDIO = confidence.LOW_AUDIO_CONFIDENCE      # single source of the code, see confidence.ALL_FLAGS (D-106)


def reasons(unit: dict) -> list[str]:
    """All issue codes for a line, for display (edited lines drop the automatic translation flags)."""
    codes = [] if unit.get("reviewed_text") is not None else list(unit.get("flags", []))
    codes += unit.get("qa_flags", [])
    audio = unit.get("audio_confidence")
    if audio is not None and audio < confidence.ConfidenceRules().medium_audio:
        codes.append(LOW_AUDIO)
    return list(dict.fromkeys(codes))


def needs_review(unit: dict) -> bool:
    return not unit.get("approved") and unit.get("confidence", confidence.HIGH) != confidence.HIGH


def write_review_file(path: Path, units: list[dict]) -> int:
    entries = []
    for u in units:
        if not needs_review(u):
            continue
        entry = (f"#{u['id'] + 1}  {format_timestamp(u['start'])} --> {format_timestamp(u['end'])}  "
                 f"{u['confidence']}  [{', '.join(reasons(u))}]  audio={u.get('audio_confidence')}\n"
                 f"  SRC: {u['text']}\n  TGT: {u['final_text']}\n")
        if u.get("review"):
            entry += f"  SUGGESTED: {u['review']['text']}\n  WHY: {u['review'].get('reason', '')}\n"
        if u.get("disagreement"):                  # two AI models disagree (D-104): both candidates for the reviewer
            d = u["disagreement"]
            entry += f"  AI DISAGREEMENT ({d['choice']}): {d.get('reason', '')}\n  A: {d['a']}\n  B: {d['b']}\n"
        entries.append(entry)
    header = f"Lines needing review: {len(entries)} of {len(units)} (approved lines are not listed)\n\n"
    atomic_write_bytes(path, (header + "\n".join(entries)).encode("utf-8"))
    return len(entries)
