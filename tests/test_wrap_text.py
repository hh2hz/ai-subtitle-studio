"""Task 5.4: grammar-aware line breaks; intentional line breaks are kept (D-101)."""

import random

import pytest

from app.core import style
from app.core.exporter import wrap_text


def test_break_after_punctuation_is_preferred():
    text = "I told him that I would be there, but he did not believe me at all"
    assert wrap_text(text, 42, 2, "en") == "I told him that I would be there,\nbut he did not believe me at all"


def test_a_line_never_ends_with_an_article_preposition_or_pronoun_when_avoidable():
    text = "We should leave now before they find out what we did to them"
    assert wrap_text(text, 42, 2, None) == "We should leave now before they\nfind out what we did to them"
    assert wrap_text(text, 42, 2, "en") == "We should leave now before\nthey find out what we did to them"
    for text in ("She walked into the old room and looked at the strange picture on the wall",
                 "He gave the book to the man at the door of the house and went away from there"):
        first = wrap_text(text, 42, 2, "en").split("\n")[0].split()[-1].lower()
        assert first not in style.break_words("en")[0]


def test_break_before_a_conjunction_when_the_shape_allows_it():
    text = "Ben bunu bilmiyorum ama sen biliyorsun ve bunu herkes biliyor artık"
    assert wrap_text(text, 42, 2, "tr").split("\n")[1].startswith("ve ")


def test_overflow_still_comes_first_and_the_second_line_is_not_shorter_when_equal():
    text = "one two three four five six seven eight nine ten eleven"
    assert wrap_text(text, 42, 2, "en") == "one two three four five six\nseven eight nine ten eleven"
    # a punctuation break that would push a line over the limit loses to a clean one
    assert wrap_text("Yes, " + "word " * 15 + "end", 42, 2, "en").count("\n") == 1
    assert all(len(line) <= 42 for line in wrap_text("Yes, " + "word " * 14 + "end", 42, 2, "en").split("\n"))


def test_intentional_line_breaks_are_kept():
    assert wrap_text("- Hello there\n- Hi, how are you doing today", 42, 2, "en") == \
        "- Hello there\n- Hi, how are you doing today"
    assert wrap_text("a  b\n\n c ", 42, 2) == "a b\nc"
    assert wrap_text("a\nb\nc", 42, 2) == "a b c"                 # more lines than allowed: wrapped again


@pytest.mark.parametrize("language", [None, "en", "tr", "ar", "fr", "zh", "xx"])
def test_wrapping_never_drops_or_reorders_words(language):
    rng = random.Random(7)
    vocabulary = ["the", "a", "and", "to", "of", "ve", "ama", "et", "word", "longerword", "x", "in,", "end."]
    for _ in range(60):
        text = " ".join(rng.choice(vocabulary) for _ in range(rng.randint(2, 30)))
        wrapped = wrap_text(text, 30, 2, language)
        assert wrapped.replace("\n", " ") == text
        assert wrapped.count("\n") <= 1


def test_word_lists_exist_for_the_spaced_languages_and_are_lowercase_sets():
    for code in ("en", "tr", "fr", "de", "es", "ru", "ar", "he", "fa"):
        no_end, before = style.break_words(code)
        assert no_end and before and all(w == w.casefold() for w in no_end | before), code
    assert style.break_words("xx") == (frozenset(), frozenset())
