"""Tasks 5.2 and 5.3: reading-speed repair in polish_timing and per-language QA limits (D-100)."""

import pytest

from app.core import style
from app.core.exporter import Cue, polish_timing
from app.core.finalize import finalize_units
from app.core.subtitle_qa import QaLimits, check_cues

TEXT_30 = "x" * 30


def cps(cue):
    return len(" ".join(cue.text.split())) / (cue.end - cue.start)


def test_limits_come_from_the_style_guide_with_a_generic_fallback():
    assert QaLimits.for_language("ar") == QaLimits(max_line_chars=42, max_cps=17.0, min_duration=0.833)
    assert QaLimits.for_language("zh").max_line_chars == 16 == QaLimits.for_language("ja").max_line_chars
    assert QaLimits.for_language("xx") == QaLimits.for_language(None) == QaLimits(max_cps=17.0)
    assert style.min_duration("ar") == pytest.approx(0.833)


def test_limits_change_what_qa_flags():
    cue = Cue(0.0, 2.0, "x" * 36)                                   # 18 cps
    assert check_cues([cue], QaLimits(max_cps=20.0)) == [[]]
    assert check_cues([cue], QaLimits.for_language("ar")) == [["reading_speed"]]
    assert check_cues([Cue(0.0, 3.0, "x" * 20)], QaLimits.for_language("zh")) == [["line_too_long"]]


def test_step1_extension_still_comes_first():
    cues = polish_timing([Cue(10.0, 11.0, "x" * 30, 0)], max_cps=17.0, fix_speed=True)
    assert cues[0].start == 10.0 and cps(cues[0]) <= 17.0            # the free time after the cue is enough


def test_step2_start_moves_up_to_half_a_second_earlier_keeping_the_gap():
    # The next cue leaves no room to extend; 0.5 s before the start is free.
    cues = polish_timing([Cue(5.0, 6.0, "x" * 20, 0), Cue(6.09, 8.0, "y" * 10, 1)], max_cps=17.0, fix_speed=True)
    first = cues[0]
    assert 4.5 <= first.start < 5.0 and cps(first) <= 17.0 + 1e-6      # moved only as far as needed
    # never into the previous cue: only 0.2 s of free time before this one
    cues = polish_timing([Cue(0.0, 4.8, "z" * 5, 0), Cue(5.0, 6.0, "x" * 20, 1), Cue(6.09, 8.0, "y", 2)],
                         max_cps=17.0, fix_speed=True)
    assert cues[1].start >= cues[0].end + 0.083 - 1e-6


def test_step3_merges_a_short_neighbour_of_the_same_speaker():
    a = Cue(0.0, 1.0, "hello there", 0, "S1")
    b = Cue(1.1, 3.1, "x" * 36, 1, "S1")                            # 18 cps and no room before or after
    c = Cue(3.2, 5.0, "next", 2, "S2")
    out = polish_timing([a, b, c], max_cps=17.0, fix_speed=True, min_gap=0.083)
    merged = out[0]
    assert merged.uid == 0 and merged.absorbed == [1] and merged.start == 0.0
    assert merged.text.replace("\n", " ") == "hello there " + "x" * 36
    assert len(out) == 2 and cps(merged) <= 17.0


def test_merge_needs_the_same_speaker_a_short_neighbour_and_a_fit_in_two_lines():
    fast = Cue(1.1, 3.1, "x" * 36, 1, "S1")
    other_voice = [Cue(0.0, 1.0, "hello", 0, "S2"), fast, Cue(3.2, 5.0, "y" * 5, 2, "S2")]
    assert len(polish_timing(other_voice, max_cps=17.0, fix_speed=True)) == 3
    long_neighbour = [Cue(0.0, 1.0, "w" * 60, 0, "S1"), fast]
    assert len(polish_timing(long_neighbour, max_cps=17.0, fix_speed=True)) == 2
    too_wide = [Cue(0.0, 1.0, "w" * 28, 0, "S1"), Cue(1.1, 3.1, "x" * 70, 1, "S1")]
    assert len(polish_timing(too_wide, max_cps=17.0, fix_speed=True)) == 2


def test_step4_leaves_a_cue_that_cannot_be_fixed_and_qa_flags_it():
    out = polish_timing([Cue(0.0, 1.0, "x" * 60, 0, "S1"), Cue(1.09, 3.0, "y" * 60, 1, "S2")], max_cps=17.0,
                        fix_speed=True)
    assert len(out) == 2 and check_cues(out, QaLimits.for_language("ar"))[0].count("reading_speed") == 1


def test_default_polish_timing_never_moves_starts_or_merges():
    cues = [Cue(5.0, 6.0, "x" * 40, 0, "S1"), Cue(6.09, 8.0, "y", 1, "S1")]
    out = polish_timing(cues)
    assert [c.start for c in out] == [5.0, 6.09] and len(out) == 2


def _unit(uid, start, end, text, speaker=None):
    return {"id": uid, "start": start, "end": end, "translation": text, "text": text, "segment_ids": [uid],
            "speaker": speaker}


def test_finalize_pairs_flags_with_merged_cues_and_keeps_every_unit_annotated():
    segments = [{"id": i, "audio_confidence": 0.95} for i in range(3)]
    units = [_unit(0, 0.0, 1.0, "hello there", "S1"), _unit(1, 1.1, 3.1, "x" * 36, "S1"),
             _unit(2, 3.2, 5.0, "next line", "S2")]
    cues = finalize_units(units, segments, "ar")
    assert len(cues) == 2 and cues[0].absorbed == [1]
    assert all("qa_flags" in u for u in units) and units[1]["qa_flags"] == units[0]["qa_flags"]
    assert "reading_speed" not in units[0]["qa_flags"]


class _StubRefiner:
    def __init__(self):
        self.calls = []

    def _condense(self, lines, result, glossary, factor=1.2):
        self.calls.append((lines, factor))
        for line in lines:
            result.translations[line["id"]] = "short"
            result.changed.add(line["id"])


def test_condense_after_timing_only_touches_lines_still_too_fast(tmp_path):
    from tests.fakes import FakeAsrEngine, FakeTranslator
    from tests.media import make_tone_file
    from tests.test_pipeline import _pipeline

    media = make_tone_file(tmp_path / "e.m4a", seconds=20.0)
    pipe = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator())[0]
    fast = {"id": 0, "start": 0.0, "end": 1.0, "translation": "x" * 60, "text": "src", "segment_ids": [0],
            "llm_translation": "x" * 60, "flags": []}
    slow = {"id": 1, "start": 10.0, "end": 14.0, "translation": "fine", "text": "src2", "segment_ids": [1],
            "llm_translation": "fine", "flags": []}
    units = [fast, slow]
    stub = _StubRefiner()
    # unit 0 is followed by a cue 1.1 s later: no room to extend, so it stays above 17 cps
    units[1]["start"] = 1.1
    assert pipe._condense_after_timing(stub, units, "ar", {}) == 1
    lines, factor = stub.calls[0]
    assert [l["id"] for l in lines] == [0] and factor == 1.0 and lines[0]["max_chars"] <= 20
    assert units[0]["translation"] == "short" and units[0]["ai_corrected"] and units[1]["translation"] == "fine"


def test_condense_after_timing_never_raises(tmp_path):
    from tests.fakes import FakeAsrEngine, FakeTranslator
    from tests.media import make_tone_file
    from tests.test_pipeline import _pipeline

    pipe = _pipeline(tmp_path, make_tone_file(tmp_path / "e.m4a", seconds=20.0), FakeAsrEngine(), FakeTranslator())[0]
    assert pipe._condense_after_timing(object(), [{"id": 0}], "ar", {}) == 0
