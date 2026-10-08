"""Risky-line selection for the second-model double check (task 2.3, D-104).

A line is risky when a cheap rule suggests its translation may be wrong: a failed rule check, a name problem, a
weak model, an odd length ratio, a lost negation or question, or poor audio. Only risky lines cost extra requests.
"""

from __future__ import annotations

import re

from app.core import style
from app.core.verification import check_line

# Turkish negative verb stem (-ma/-me, also -mi/-mu/-mü before -yor) followed by a tense or person ending:
# "bilmiyorum", "gelmedi", "gitmeyecek", "yapmamalı", "istemem". Noisy on purpose: a false alarm costs one request.
_TR_NEGATIVE_VERB = re.compile(
    r"\b\w{2,}(?:ma|me)(?:d[ıiuü]|yacak|yecek|mal[ıi]|meli|yor|z|m|n|s[ıiuü]n|d[ıiuü]k|d[ıiuü]n)\w*\b"
    r"|\b\w{2,}m[ıiuü]yor\w*\b", re.I)
# Common words that look like a negative verb ("tamam" = okay, "zaman" = time, "hemen" = at once).
_TR_NOT_NEGATIVE = {"tamam", "tamamen", "hemen", "zaman", "zamanda", "madem", "memnun", "meme", "mama", "dama"}
_QUESTION_SOURCE = ("?", "¿", "\u061f", "？")
_QUESTION_TARGET = ("?", "\u061f", "？")
_SUBSTRING_LANGUAGES = {"zh", "ja", "ko"}           # no spaces between words
_WORD = re.compile(r"[\w']+", re.U)
MAX_RISKY_PER_BLOCK = 12
LOW_AUDIO = 0.5


def _tokens(text: str) -> list[str]:
    return [t.casefold() for t in _WORD.findall(text)]


def source_has_negation(text: str, language: str) -> bool:
    words = style.source_guide(language)["negation"]
    tokens = _tokens(text)
    if any(t.endswith("n't") for t in tokens):
        return True
    plain = {w.casefold() for w in words if not w.startswith("-")}
    if plain & set(tokens):
        return True
    if language.split("-")[0].lower() == "tr" and any(w.startswith("-") for w in words):
        return any(m.group(0).casefold() not in _TR_NOT_NEGATIVE for m in _TR_NEGATIVE_VERB.finditer(text))
    return False


def target_has_negation(text: str, language: str) -> bool:
    words = [w.casefold() for w in style.target_guide(language).get("negation", [])]
    if not words:
        return True                    # no word list for this language: nothing can be claimed to be lost
    low = text.casefold()
    if language.split("-")[0].lower() in _SUBSTRING_LANGUAGES:
        return any(w in low for w in words)
    tokens = set(_tokens(text))
    return any(w in tokens for w in words) or any(t.endswith("n't") for t in tokens)


def risk_reasons(source: str, target: str, target_language: str, source_language: str, *,
                 previous_pair: tuple[str, str] | None = None, weak: bool = False, name_issue: bool = False,
                 audio_confidence: float | None = None) -> list[str]:
    """Why a translated line is risky (empty list: not risky)."""
    reasons = []
    if check_line(source, target, target_language, previous_pair):
        reasons.append("check_line")
    if name_issue:
        reasons.append("name_mismatch")
    if weak:
        reasons.append("weak_model")
    if len(source) >= 8:
        low, high = style.length_ratio(target_language)
        if not low <= len(target) / max(len(source), 1) <= high:
            reasons.append("length_ratio")
    if source_has_negation(source, source_language) and not target_has_negation(target, target_language):
        reasons.append("negation_lost")
    if any(q in source for q in _QUESTION_SOURCE) and not any(q in target for q in _QUESTION_TARGET):
        reasons.append("question_lost")
    if audio_confidence is not None and audio_confidence < LOW_AUDIO:
        reasons.append("low_audio")
    return reasons


def select(reasons_by_id: dict[int, list[str]]) -> dict[int, list[str]]:
    """The riskiest lines (most reasons first, then document order), at most MAX_RISKY_PER_BLOCK."""
    ranked = sorted((i for i, r in reasons_by_id.items() if r), key=lambda i: (-len(reasons_by_id[i]), i))
    return {i: reasons_by_id[i] for i in sorted(ranked[:MAX_RISKY_PER_BLOCK])}
