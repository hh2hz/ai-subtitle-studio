"""M4 acceptance: every QA rule has a positive and a negative test; confidence categories."""

import pytest

from app.core import confidence as c
from app.core import language_qa as lq
from app.core import subtitle_qa as sq
from app.core.exporter import Cue

AR = "\u0647\u0644 \u0623\u0646\u062a \u0645\u062a\u0623\u0643\u062f"     # an Arabic sentence without punctuation
Q = "\u061f"          # Arabic question mark
COMMA = "\u060c"      # Arabic comma
ARQ = AR + Q
qa = lq.ArabicLanguageQA()


@pytest.mark.parametrize("bad,good,flag", [
    (AR + "?", ARQ, lq.LATIN_PUNCTUATION),
    (AR + " , " + AR, AR + COMMA + " " + AR, lq.LATIN_PUNCTUATION),
    (AR + " \u0665", AR + " 5", lq.INDIC_DIGITS),
    (AR + " " + Q, ARQ, lq.SPACE_BEFORE_PUNCT),
    (AR + "...", AR + "\u2026", lq.THREE_DOTS),
    ("\u0645\u0640\u0640\u0627\u0634\u064a", "\u0645\u0627\u0634\u064a", lq.TATWEEL),
    (AR + "  " + AR, AR + " " + AR, lq.WHITESPACE),
])
def test_arabic_rules_flag_and_fix(bad, good, flag):
    assert flag in qa.check(bad)
    fixed = qa.fix(bad)
    assert fixed == good
    assert flag not in qa.check(fixed)


@pytest.mark.parametrize("flag,bad", [
    (lq.COMBINED_MARKS, AR + Q + "!"),
    (lq.LATIN_WORDS, AR + " hello"),
])
def test_arabic_rules_flag_only(flag, bad):
    assert flag in qa.check(bad)
    assert flag not in qa.check(ARQ)


def test_acronyms_and_rlm_are_allowed():
    assert qa.check("\u200f" + AR + " PLT" + Q) == []


def test_generic_language_qa():
    generic = lq.for_language("tr")
    assert type(generic) is lq.LanguageQA and isinstance(lq.for_language("ar"), lq.ArabicLanguageQA)
    assert generic.check("Merhaba  dunya") == [lq.WHITESPACE] and generic.check("Merhaba dunya") == []
    assert generic.fix(" a \n\n b ") == "a\nb"


@pytest.mark.parametrize("cues,index,flag", [
    ([Cue(1, 1, "a")], 0, sq.INVALID_TIMING),
    ([Cue(0, 2, "a"), Cue(1.5, 3, "b")], 1, sq.OVERLAP),
    ([Cue(0, 0.5, "a")], 0, sq.TOO_SHORT),
    ([Cue(0, 8, "a")], 0, sq.TOO_LONG),
    ([Cue(0, 1, "x" * 30)], 0, sq.READING_SPEED),
    ([Cue(0, 5, "x" * 43)], 0, sq.LINE_TOO_LONG),
    ([Cue(0, 5, "a\nb\nc")], 0, sq.TOO_MANY_LINES),
    ([Cue(0, 2, "\u200f ")], 0, sq.EMPTY),
    ([Cue(0, 2, "same"), Cue(2.1, 4, "same")], 1, sq.DUPLICATE),
])
def test_subtitle_rules_positive(cues, index, flag):
    assert flag in sq.check_cues(cues)[index]


def test_subtitle_rules_negative():
    good = [Cue(0, 2, "Hello there"), Cue(2.1, 5.0, "x" * 42 + "\n" + "y" * 10), Cue(10, 12, "Hello there")]
    assert sq.check_cues(good) == [[], [], []]
    assert sq.check_cues([Cue(0, 1, "x" * 30)], sq.QaLimits(max_cps=40)) == [[]]   # limits are configurable


@pytest.mark.parametrize("flags,audio,expected", [
    ([], 0.95, c.HIGH),
    ([], None, c.HIGH),
    (["three_dots"], 0.95, c.HIGH),                 # neutral, auto-fixed
    (["reading_speed"], 0.95, c.MEDIUM),
    ([], 0.6, c.MEDIUM),
    (["numbers_differ"], 0.95, c.LOW),
    ([], 0.4, c.LOW),
    (["not_refined_by_ai"], 0.99, c.LOW),
])
def test_confidence_category(flags, audio, expected):
    assert c.category(flags, audio) == expected


def test_confidence_rules_configurable():
    strict = c.ConfidenceRules(medium_audio=0.99)
    assert c.category([], 0.95, strict) == c.MEDIUM
