"""Tier 1 (rule-based) translation checks. Always available; flags lines for human review."""

from __future__ import annotations

import re
import unicodedata

from app.core.subtitle_formats import text_similarity

_ARABIC_INDIC = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
                              "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9",
                              "01234567890123456789")
_NUMBER = re.compile(r"\d+")
_REPEAT = re.compile(r"(\b\w+\b)(?:\W+\1\b){3,}", re.I | re.U)

EMPTY = "empty_translation"
UNTRANSLATED = "untranslated_or_wrong_script"
NUMBERS = "numbers_differ"
LENGTH = "suspicious_length"
REPETITION = "repetition_loop"
SAME_AS_PREVIOUS = "same_as_previous_line"

#: Every flag code this module can emit (D-106).
FLAGS = frozenset({EMPTY, UNTRANSLATED, NUMBERS, LENGTH, REPETITION, SAME_AS_PREVIOUS})


# Unicode name prefixes of the script each target language is written in (languages not listed use the Latin
# script and are checked by similarity to the source instead).
_SCRIPTS = {
    **dict.fromkeys(("ar", "fa", "ur", "ps", "ckb", "ug", "sd"), ("ARABIC",)),
    **dict.fromkeys(("he", "yi"), ("HEBREW",)),
    **dict.fromkeys(("ru", "uk", "bg", "sr", "mk", "be", "kk", "ky", "mn", "tg"), ("CYRILLIC",)),
    "el": ("GREEK",), "hy": ("ARMENIAN",), "ka": ("GEORGIAN",), "th": ("THAI",), "lo": ("LAO",),
    "km": ("KHMER",), "my": ("MYANMAR",), "am": ("ETHIOPIC",), "ko": ("HANGUL",),
    **dict.fromkeys(("hi", "mr", "ne", "sa"), ("DEVANAGARI",)),
    "bn": ("BENGALI",), "ta": ("TAMIL",), "te": ("TELUGU",), "kn": ("KANNADA",), "ml": ("MALAYALAM",),
    "gu": ("GUJARATI",), "pa": ("GURMUKHI",), "si": ("SINHALA",),
    **dict.fromkeys(("zh", "yue"), ("CJK",)),
    "ja": ("CJK", "HIRAGANA", "KATAKANA"),
}


def _script_ratio(text: str, script_prefixes: tuple[str, ...]) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    matching = sum(1 for c in letters if unicodedata.name(c, "").startswith(script_prefixes))
    return matching / len(letters)


def numbers(text: str) -> list[str]:
    return sorted(_NUMBER.findall(text.translate(_ARABIC_INDIC)))


def check_line(source: str, target: str, target_language: str, previous: tuple[str, str] | None = None) -> list[str]:
    flags = []
    if not target.strip():
        return [EMPTY]
    script = _SCRIPTS.get(target_language.split("-")[0].lower())
    if script:
        if _script_ratio(target, script) < 0.6:
            flags.append(UNTRANSLATED)
    elif text_similarity(source, target) > 0.85 and len(source) > 6:
        flags.append(UNTRANSLATED)
    src_numbers, tgt_numbers = numbers(source), numbers(target)
    # Small numbers are often written as words in subtitles ("5" -> "the fifth hour"), which is correct.
    spelled_out = not tgt_numbers and all(int(n) <= 20 for n in src_numbers)
    if src_numbers != tgt_numbers and not spelled_out:
        flags.append(NUMBERS)
    if len(source) >= 12:
        ratio = len(target) / len(source)
        if ratio < 0.3 or ratio > 3.0:
            flags.append(LENGTH)
    if _REPEAT.search(target):
        flags.append(REPETITION)
    if previous and previous[1].strip() == target.strip() and previous[0].strip() != source.strip():
        flags.append(SAME_AS_PREVIOUS)
    return flags
