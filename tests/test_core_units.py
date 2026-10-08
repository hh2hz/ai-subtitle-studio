from app.core import master_transcript as mt
from app.core.transcription import AsrSegment, Word
from app.core.translation import build_units


def _seg(i, start, end, text):
    return {"id": i, "start": start, "end": end, "text": text}


def test_build_units_merges_open_sentences_only():
    segs = [
        _seg(0, 0.0, 1.5, "Seni orada"),
        _seg(1, 1.7, 3.0, "gordum."),
        _seg(2, 3.1, 4.0, "Tamam."),
        _seg(3, 6.0, 7.0, "Bekle"),          # gap 2 s: not merged with the next
        _seg(4, 9.5, 10.0, "simdi."),
        _seg(5, 10.1, 11.0, "   "),          # empty text skipped
    ]
    units = build_units(segs)
    assert [(u.segment_ids, u.text) for u in units] == [
        ([0, 1], "Seni orada gordum."), ([2], "Tamam."), ([3], "Bekle"), ([4], "simdi."),
    ]
    assert units[0].start == 0.0 and units[0].end == 3.0


def test_build_units_respects_duration_and_length():
    segs = [_seg(i, i * 2.0, i * 2.0 + 1.9, "kelime kelime") for i in range(10)]
    for unit in build_units(segs, max_duration=7.0, max_chars=90):
        assert unit.end - unit.start <= 7.0
        assert len(unit.text) <= 90


def test_master_document_structure_and_confidence():
    segs = [
        AsrSegment(0.0, 1.0, "a", [Word(0.0, 0.5, "a", 0.8), Word(0.5, 1.0, "b", 0.6)], -0.3),
        AsrSegment(1.0, 2.0, "b", [], avg_logprob=-0.6931),
        AsrSegment(2.0, 3.0, "c", []),
    ]
    built = mt.build_segments(segs, "tr", ["e1", "e1", "e2"])
    assert built[0]["confidence"] == 0.7
    assert abs(built[1]["confidence"] - 0.5) < 1e-3
    assert built[2]["confidence"] is None
    assert built[2]["sources"] == [{"type": "asr", "engine": "e2"}]
    doc = mt.make_document(built, language="tr", language_probability=0.9, duration=3.0, job={})
    assert mt.validate_document(doc) == []
    for field in mt.SEGMENT_FIELDS:
        assert field in doc["segments"][0]


def test_master_validation_catches_problems():
    doc = mt.make_document(
        [{f: None for f in mt.SEGMENT_FIELDS} | {"id": 0, "start": 2.0, "end": 1.0, "confidence": 1.5}],
        language="tr", language_probability=None, duration=3.0, job={})
    problems = mt.validate_document(doc)
    assert any("invalid time range" in p for p in problems)
    assert any("outside [0, 1]" in p for p in problems)
