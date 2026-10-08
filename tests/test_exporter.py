from pathlib import Path

import pytest

from app.core.exporter import (
    Cue, fix_overlaps, format_timestamp, output_paths, parse_srt, parse_timestamp, read_srt,
    safe_basename, to_srt, wrap_text, write_srt,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("seconds,text", [
    (0, "00:00:00,000"),
    (1.5, "00:00:01,500"),
    (59.9996, "00:01:00,000"),
    (3661.007, "01:01:01,007"),
    (-0.2, "00:00:00,000"),
    (360000.0, "100:00:00,000"),
])
def test_format_timestamp(seconds, text):
    assert format_timestamp(seconds) == text


def test_parse_timestamp_accepts_dot_and_rejects_garbage():
    assert parse_timestamp("01:02:03.004") == pytest.approx(3723.004)
    with pytest.raises(ValueError):
        parse_timestamp("1:2:3")


def test_write_read_round_trip(tmp_path):
    cues = [Cue(0.0, 1.234, "Merhaba d\u00fcnya"), Cue(2.0, 4.5, "\u0130ki sat\u0131r\nikinci sat\u0131r"), Cue(3725.5, 3726.0, "x")]
    path = tmp_path / "out.srt"
    write_srt(cues, path)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in raw
    back = read_srt(path)
    assert [(c.start, c.end, c.text) for c in back] == [(c.start, c.end, c.text) for c in cues]


def test_arabic_fixture_round_trip(tmp_path):
    cues = read_srt(FIXTURES / "arabic_sample.srt")
    assert len(cues) == 2 and cues[1].text.count("\n") == 1
    path = tmp_path / "ar.srt"
    write_srt(cues, path, bom=False)
    assert [(c.start, c.end, c.text) for c in read_srt(path)] == [(c.start, c.end, c.text) for c in cues]


def test_parse_is_lenient():
    text = "\ufeff\r\n\r\n7\r\n00:00:01,000 --> 00:00:02,000\r\nA\r\n\r\n\r\n00:00:03,000 --> 00:00:04,000 X1:0\nB\n2\n00:00:05,000 --> 00:00:06,000\nC"
    cues = parse_srt(text)
    assert [c.text for c in cues] == ["A", "B", "C"]
    assert cues[2].start == 5.0


def test_to_srt_numbering():
    out = to_srt([Cue(0, 1, "a"), Cue(1, 2, "b")])
    assert out.startswith("1\n00:00:00,000 --> 00:00:01,000\na\n\n2\n")


@pytest.mark.parametrize("text,expected", [
    ("short", "short"),
    ("one two three four five six seven eight nine ten eleven", "one two three four five six\nseven eight nine ten eleven"),
    ("a" * 60, "a" * 60),   # no space: never cut a word
])
def test_wrap_text(text, expected):
    assert wrap_text(text, max_chars=42) == expected


def test_wrap_never_drops_words():
    text = " ".join(f"word{i}" for i in range(40))
    wrapped = wrap_text(text, max_chars=42)
    assert wrapped.replace("\n", " ") == text
    assert wrapped.count("\n") == 1


def test_fix_overlaps():
    fixed = fix_overlaps([Cue(2, 5, "b"), Cue(0, 3, "a"), Cue(4, 4.5, "c")])
    assert [(c.start, c.end) for c in fixed] == [(0, 2), (2, 4), (4, 4.5)]


def test_safe_basename_and_paths(tmp_path):
    assert safe_basename('Ep: 1 / "Final"?') == "Ep_ 1 _ _Final"
    assert safe_basename("...") == "Episode"
    paths = output_paths(tmp_path, "Show S01E02", "ar", "tr")
    assert paths["target_srt"] == tmp_path / "Show S01E02.ar.srt"        # next to the video
    assert paths["source_srt"] == tmp_path / "work" / "Show S01E02.tr.source.srt"
    assert paths["master_transcript"].parent.name == "work"
