"""Segmentation, Tier 1 checks and timing polish."""

import pytest

from app.core.exporter import RLM, Cue, mark_rtl, polish_timing
from app.core.segmentation import MAX_CUE_CHARS, segment_cues
from app.core import verification as v


def _seg(i, start, end, words):
    text = "".join(w[2] for w in words).strip()
    return {"id": i, "start": start, "end": end, "text": text,
            "words": [{"start": s, "end": e, "text": t} for s, e, t in words]}


# -- segmentation --------------------------------------------------------------------------------

def test_short_segments_are_never_merged():
    segs = [_seg(0, 0, 1, [(0, 1, " Seni orada")]), _seg(1, 1.1, 2, [(1.1, 2, " gordum.")])]
    units = segment_cues(segs)
    assert [u.text for u in units] == ["Seni orada", "gordum."]


def test_long_segment_split_at_sentence_end():
    words = [(0.0, 0.5, " Bunu"), (0.5, 1.0, " hemen"), (1.0, 1.6, " yapacaksin."), (1.8, 2.2, " Sonra"),
             (2.2, 2.8, " bana"), (2.8, 3.5, " haber"), (3.5, 4.0, " ver,"), (4.0, 5.0, " anladin"),
             (5.0, 7.5, " mi"), (7.5, 8.0, " kardesim?")]
    units = segment_cues([_seg(0, 0.0, 8.0, words)])
    assert units[0].text == "Bunu hemen yapacaksin."
    assert all(u.end - u.start <= 6.5 for u in units)
    assert " ".join(u.text for u in units) == "Bunu hemen yapacaksin. Sonra bana haber ver, anladin mi kardesim?"
    assert all(u.segment_ids == [0] for u in units)


def test_long_text_split_by_length():
    words = [(i * 0.3, i * 0.3 + 0.3, f" kelime{i}") for i in range(20)]
    units = segment_cues([_seg(0, 0, 6, words)])
    assert len(units) > 1 and all(len(u.text) <= MAX_CUE_CHARS for u in units)


def test_segment_without_words_kept():
    assert [u.text for u in segment_cues([{"id": 0, "start": 0, "end": 1, "text": "x", "words": []}])] == ["x"]


# -- Tier 1 checks --------------------------------------------------------------------------------

AR_OK = "\u0647\u0644 \u0623\u0646\u062a \u0645\u062a\u0623\u0643\u062f\u061f"


def test_tier1_flags():
    assert v.check_line("Emin misin?", AR_OK, "ar") == []
    assert v.check_line("Emin misin?", "", "ar") == [v.EMPTY]
    assert v.UNTRANSLATED in v.check_line("Emin misin?", "Emin misin?", "ar")
    assert v.UNTRANSLATED in v.check_line("Are you sure about it?", "Are you sure about it?", "de")
    assert v.NUMBERS not in v.check_line("Saat 5 te gel", AR_OK, "ar")        # small number spelled out
    assert v.NUMBERS in v.check_line("Saat 1995 te gel", AR_OK, "ar")         # large number dropped
    assert v.NUMBERS in v.check_line("Saat 5 te gel", "7 " + AR_OK, "ar")      # wrong digit
    five_ar_indic = "\u0661\u0665 " + AR_OK
    assert v.NUMBERS not in v.check_line("15 dakika", five_ar_indic, "ar")
    loop = " ".join([AR_OK.split()[0]] * 5)
    assert v.REPETITION in v.check_line("Hayir hayir", loop, "ar")
    assert v.LENGTH in v.check_line("Bu cok uzun bir cumle ama ceviri cok kisa", "\u0644\u0627", "ar")
    assert v.SAME_AS_PREVIOUS in v.check_line("B", AR_OK, "ar", previous=("A", AR_OK))


# -- export polish --------------------------------------------------------------------------------

def test_polish_timing():
    cues = polish_timing([Cue(0.0, 0.3, "a"), Cue(0.5, 1.0, "b"), Cue(3.0, 20.0, "c")])
    assert cues[0].end == pytest.approx(0.5 - 0.083)          # extended but keeps the gap
    assert cues[1].end == pytest.approx(1.333)                 # extended to the minimum duration
    assert cues[2].end == pytest.approx(10.0)                  # capped at 7 s
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))


def test_mark_rtl_is_idempotent():
    once = mark_rtl("a\nb")
    assert once == f"{RLM}a{RLM}\n{RLM}b{RLM}" and mark_rtl(once) == once
