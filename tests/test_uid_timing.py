"""Task 5.1: cue uids survive sorting and timing; QA flags are paired by uid (D-097)."""

from app.core.exporter import Cue, fix_overlaps, polish_timing
from app.core.finalize import finalize_units, source_cues


def _unit(uid, start, end, text):
    return {"id": uid, "start": start, "end": end, "translation": text, "text": text, "segment_ids": [uid]}


def test_uid_survives_overlap_fix_and_timing_polish():
    cues = [Cue(5.0, 6.0, "b", 1), Cue(1.0, 2.0, "a", 0), Cue(5.1, 7.0, "c", 2)]
    assert [c.uid for c in fix_overlaps(cues)] == [0, 1, 2]
    assert [c.uid for c in polish_timing(cues)] == [0, 1, 2]
    assert Cue(0.0, 1.0, "x").uid is None


def test_qa_flags_follow_the_unit_when_units_are_not_in_time_order():
    segments = [{"id": i, "audio_confidence": 0.95} for i in range(3)]
    long_text = "word " * 30                                      # far above 20 cps in a 1 s slot
    units = [_unit(0, 10.0, 11.0, long_text.strip()), _unit(1, 0.0, 3.0, "short"), _unit(2, 5.0, 8.0, "also short")]
    finalize_units(units, segments, "en")
    assert "reading_speed" in units[0]["qa_flags"] or "line_too_long" in units[0]["qa_flags"]
    assert units[1]["qa_flags"] == [] and units[2]["qa_flags"] == []


def test_source_cues_carry_the_unit_id():
    assert [c.uid for c in source_cues([_unit(7, 3.0, 4.0, "a"), _unit(3, 0.0, 1.0, "b")])] == [3, 7]
