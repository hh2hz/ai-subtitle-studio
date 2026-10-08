"""Per-language subtitle style guides (app/resources/style_guides.json, DECISIONS D-060).

Target-language rules go into the translation prompts; numbers (cps, max_line, length_ratio) and the source-language
notes and negation words are data for later tasks (risky-line selection, reading-speed budget, QA limits).
Unknown languages fall back to the generic entry. The Arabic text is unchanged from before the move, so the rendered
Arabic prompts are byte-identical (tests/test_style.py compares them with tests/fixtures/style_golden).
"""

from __future__ import annotations

import json
import logging
import math

from app.utils.paths import resources_dir

log = logging.getLogger(__name__)

_FALLBACK_GENERIC = {"rules": [], "cps": 17, "max_line": 42, "length_ratio": [0.4, 2.5], "verified": False,
                     "source": "generic"}
_cache: dict | None = None


def _data() -> dict:
    global _cache
    if _cache is None:
        try:
            loaded = json.loads((resources_dir() / "style_guides.json").read_text(encoding="utf-8"))
            _cache = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            log.warning("style_guides.json is missing or invalid; using generic rules only")
            _cache = {}
    return _cache


def _code(language: str | None) -> str:
    return (language or "").split("-")[0].split("_")[0].lower()


def target_guide(language: str | None) -> dict:
    """Guide of a target language: the generic entry, overridden by the language's own fields."""
    data = _data()
    guide = {**_FALLBACK_GENERIC, **(data.get("generic") or {})}
    guide.update((data.get("target") or {}).get(_code(language)) or {})
    return guide


def _rules(guide: dict, key: str) -> list[str]:
    rules = guide.get(key)
    return [r for r in rules if isinstance(r, str) and r.strip()] if isinstance(rules, list) else []


def cloud_style(language: str | None) -> str:
    """Style text for the cloud prompts ({style}): one rule per line."""
    return "\n".join(_rules(target_guide(language), "rules"))


def local_style(language: str | None) -> str:
    """Style text for the local instruct prompt: bullet lines (`local_rules`, else the cloud rules)."""
    guide = target_guide(language)
    rules = _rules(guide, "local_rules") or _rules(guide, "rules")
    return "\n".join(f"- {r}" for r in rules)


def length_ratio(language: str | None) -> tuple[float, float]:
    ratio = target_guide(language).get("length_ratio")
    try:
        low, high = float(ratio[0]), float(ratio[1])
        if 0 < low < high:
            return low, high
    except (TypeError, ValueError, IndexError):
        pass
    return 0.4, 2.5


def cps(language: str | None) -> float:
    value = target_guide(language).get("cps")
    return float(value) if isinstance(value, (int, float)) and value > 0 else 17.0


def min_duration(language: str | None) -> float:
    value = target_guide(language).get("min_duration")
    return float(value) if isinstance(value, (int, float)) and value > 0 else 0.833


def max_line(language: str | None) -> int:
    value = target_guide(language).get("max_line")
    return int(value) if isinstance(value, int) and value > 0 else 42


def break_words(language: str | None) -> tuple[frozenset[str], frozenset[str]]:
    """(words a subtitle line must not end with, words a second line should preferably start with), lowercase."""
    guide = target_guide(language)
    return (frozenset(w.casefold() for w in _rules(guide, "no_end")),
            frozenset(w.casefold() for w in _rules(guide, "break_before")))


def dash(language: str | None) -> str:
    """Prefix of each speaker's line in a two-speaker cue ("- " unless the language's guide says otherwise)."""
    value = target_guide(language).get("dash")
    return value if isinstance(value, str) and value.strip() else "- "


def max_chars(duration: float, language: str | None) -> int:
    """Reading-speed budget in characters: floor(duration * cps), min 12, max 2 * max_line."""
    dur = max(0.0, float(duration))
    return max(12, min(math.floor(dur * cps(language)), 2 * max_line(language)))


def source_guide(language: str | None) -> dict:
    """Notes and negation words of a source language (empty lists for unknown languages)."""
    entry = (_data().get("source") or {}).get(_code(language)) or {}
    return {"notes": [n for n in entry.get("notes", []) if isinstance(n, str)],
            "negation": [n for n in entry.get("negation", []) if isinstance(n, str)]}
