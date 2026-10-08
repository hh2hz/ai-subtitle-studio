"""Per-language checks and safe automatic fixes for subtitle text.

Arabic rules follow the Netflix Arabic Timed Text Style Guide (checked 2026-10-03): Arabic numerals 0-9 rather
than Arabic-Indic digits, a single ellipsis character, no space before punctuation, no "?!" combinations;
plus Arabic punctuation marks instead of Latin ones and no tatweel.
"""

from __future__ import annotations

import re

LATIN_PUNCTUATION = "latin_punctuation"
INDIC_DIGITS = "arabic_indic_digits"
SPACE_BEFORE_PUNCT = "space_before_punctuation"
THREE_DOTS = "three_dots"
COMBINED_MARKS = "combined_question_exclamation"
TATWEEL = "tatweel"
LATIN_WORDS = "latin_words"
WHITESPACE = "extra_whitespace"

#: Every flag code this module can emit (D-106).
FLAGS = frozenset({LATIN_PUNCTUATION, INDIC_DIGITS, SPACE_BEFORE_PUNCT, THREE_DOTS, COMBINED_MARKS, TATWEEL,
                   LATIN_WORDS, WHITESPACE})

_AR = "\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff"
_INDIC = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
                       "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9", "01234567890123456789")
_HAS_INDIC = re.compile("[\u0660-\u0669\u06f0-\u06f9]")
_ARABIC_LETTER = re.compile(f"[{_AR}]")
# In a line containing Arabic, Latin ? and ; are always wrong; a comma is wrong unless it separates digits (1,000).
_LATIN_PUNCT = re.compile(r"[?;]|(?<!\d),|,(?!\d)")
_SPACE_BEFORE = re.compile(r"[ \t]+([\u060c\u061f\u061b!.,:;?\u2026])")
_COMBINED = re.compile("[?!\u061f]{2,}")
_LATIN_WORD = re.compile(r"\b[A-Za-z]{2,}\b")
_ACRONYM = re.compile(r"^[A-Z0-9]{2,6}$")
_RLM = "\u200f"


class LanguageQA:
    """Language-neutral checks."""

    def fix(self, text: str) -> str:
        lines = [" ".join(line.split()) for line in text.split("\n")]
        return "\n".join(line for line in lines if line)

    def check(self, text: str) -> list[str]:
        flags = []
        if any(line != " ".join(line.split()) for line in text.split("\n")) or "\n\n" in text:
            flags.append(WHITESPACE)
        return flags


class ArabicLanguageQA(LanguageQA):
    def fix(self, text: str) -> str:
        text = super().fix(text.replace(_RLM, ""))
        text = text.translate(_INDIC)
        text = text.replace("\u0640", "")
        text = re.sub(r"\.{3,}", "\u2026", text)
        text = "\n".join(
            _LATIN_PUNCT.sub(lambda m: {"?": "\u061f", ",": "\u060c", ";": "\u061b"}[m.group(0)], line)
            if _ARABIC_LETTER.search(line) else line
            for line in text.split("\n"))
        text = _SPACE_BEFORE.sub(r"\1", text)
        return text

    def check(self, text: str) -> list[str]:
        text = text.replace(_RLM, "")
        flags = super().check(text)
        if any(_ARABIC_LETTER.search(line) and _LATIN_PUNCT.search(line) for line in text.split("\n")):
            flags.append(LATIN_PUNCTUATION)
        if _HAS_INDIC.search(text):
            flags.append(INDIC_DIGITS)
        if _SPACE_BEFORE.search(text):
            flags.append(SPACE_BEFORE_PUNCT)
        if "..." in text:
            flags.append(THREE_DOTS)
        if _COMBINED.search(text):
            flags.append(COMBINED_MARKS)
        if "\u0640" in text:
            flags.append(TATWEEL)
        if any(not _ACRONYM.match(w) for w in _LATIN_WORD.findall(text)):
            flags.append(LATIN_WORDS)
        return flags


def for_language(code: str) -> LanguageQA:
    return ArabicLanguageQA() if code.split("-")[0].lower() == "ar" else LanguageQA()
