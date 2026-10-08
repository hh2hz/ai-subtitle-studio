from pathlib import Path

import pytest

from app.core.subtitle_formats import (
    audio_agreement, collapse_rolling_captions, decode_subtitle_bytes, normalize_text, parse_subtitle_text,
    parse_vtt, text_similarity,
)

FIXTURES = Path(__file__).parent / "fixtures"

# Same structure as YouTube automatic captions: each cue repeats the previous line, plus 10 ms bridge cues.
ROLLING_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000 align:start position:0%
 
hello<00:00:00.500><c> there</c>

00:00:02.000 --> 00:00:02.010 align:start position:0%
hello there
 

00:00:02.010 --> 00:00:04.000 align:start position:0%
hello there
general<00:00:03.000><c> kenobi</c>

00:00:04.000 --> 00:00:04.010 align:start position:0%
general kenobi
 

00:00:04.010 --> 00:00:06.000 align:start position:0%
general kenobi
&gt;&gt; [music]
"""


def test_parse_vtt_strips_tags_and_entities():
    cues = parse_vtt("WEBVTT\n\n01:02.500 --> 01:03.000\n<i>Tom &amp; Jerry</i>\n")
    assert len(cues) == 1
    assert cues[0].start == pytest.approx(62.5)
    assert cues[0].text == "Tom & Jerry"


def test_rolling_captions_are_collapsed():
    cues = parse_subtitle_text(ROLLING_VTT, rolling=True)
    assert [c.text for c in cues] == ["hello there", "general kenobi", ">> [music]"]
    assert cues[0].start == 0.0 and cues[1].start == pytest.approx(2.01)


def test_manual_tracks_keep_repeated_lines():
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHeat.\n\n00:00:03.000 --> 00:00:04.000\nHeat.\n"
    assert [c.text for c in parse_subtitle_text(vtt)] == ["Heat.", "Heat."]
    assert len(collapse_rolling_captions(parse_vtt(vtt))) == 1


def test_srt_detected():
    assert parse_subtitle_text("1\n00:00:01,000 --> 00:00:02,000\nx\n")[0].text == "x"


def test_normalize_and_similarity():
    assert normalize_text("  \u0130stanbul'a  GELD\u0130M! ") == "istanbul a geldim"
    assert text_similarity("Seni orada gordum.", "seni orada g\u00f6rd\u00fcm") == 1.0
    assert text_similarity("Seni orada gordum", "Seni odada gordum") > 0.75
    assert text_similarity("Cumle 0 burada.", "lorem ipsum dolor") < 0.3
    assert text_similarity("Seni orada gordum", "forh.") < 0.3
    assert text_similarity("", "") == 1.0 and text_similarity("a", "") == 0.0


def test_audio_agreement_separates_good_and_garbage():
    segments = [{"start": 0.0, "end": 2.0, "text": "Tir hazir mi dayi?"},
                {"start": 2.5, "end": 4.0, "text": "Kim kullanacak?"}]
    good = parse_vtt("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nTir hazir mi dayi?\n\n"
                     "00:00:02.500 --> 00:00:04.000\nKim kullanacak?\n")
    garbage = parse_vtt("WEBVTT\n\n00:00:00.000 --> 00:00:04.000\nforh.\n")
    assert audio_agreement(segments, good) == 1.0
    assert audio_agreement(segments, garbage) < 0.3
    assert audio_agreement([], good) is None


def test_decode_legacy_encodings():
    turkish = "G\u00fcle g\u00fcle \u015fehir"
    assert decode_subtitle_bytes(turkish.encode("cp1254"), "tr") == turkish
    arabic = (FIXTURES / "arabic_sample.srt").read_text(encoding="utf-8")
    assert decode_subtitle_bytes(arabic.encode("cp1256"), "ar") == arabic
    assert decode_subtitle_bytes(b"\xef\xbb\xbfabc") == "abc"
    assert decode_subtitle_bytes("x".encode("utf-16")) == "x"
