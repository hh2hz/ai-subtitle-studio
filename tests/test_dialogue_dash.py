"""Task 3.3: one cue with two speakers becomes two dash lines (D-102)."""

from app.core import diarize, style
from app.core.exporter import Cue, wrap_text
from app.core.finalize import dialogue_text, finalize_units, speaker_split


def _word(start, end, text, speaker=None):
    word = {"start": start, "end": end, "text": text, "probability": 0.9}
    if speaker:
        word["speaker"] = speaker
    return word


def _doc():
    words = [_word(0.0, 0.5, "Where"), _word(0.5, 1.0, "are"), _word(1.0, 1.5, "you?"),
             _word(1.6, 2.0, "Home."), _word(2.0, 2.4, "Why")]
    return {"segments": [{"id": 0, "start": 0.0, "end": 2.4, "text": "Where are you? Home. Why", "words": words,
                          "speaker": None}]}


def test_label_words_marks_each_word_when_two_speakers_share_a_segment():
    doc = _doc()
    diarize.apply_labels(doc, [(0.0, 1.55, 0), (1.55, 2.4, 1)])
    assert [w.get("speaker") for w in doc["segments"][0]["words"]] == ["S1", "S1", "S1", "S2", "S2"]
    single = _doc()
    diarize.apply_labels(single, [(0.0, 2.4, 0)])
    assert all("speaker" not in w for w in single["segments"][0]["words"])


def test_overlapping_speech_goes_to_the_shorter_covering_turn():
    doc = _doc()
    diarize.apply_labels(doc, [(0.0, 2.4, 0), (1.55, 2.4, 1)])
    assert [w.get("speaker") for w in doc["segments"][0]["words"]][-2:] == ["S2", "S2"]


def test_speaker_split_reports_the_share_of_the_first_speaker():
    doc = _doc()
    diarize.apply_labels(doc, [(0.0, 1.55, 0), (1.55, 2.4, 1)])
    unit = {"segment_ids": [0], "start": 0.0, "end": 2.4}
    split = speaker_split(unit, {0: doc["segments"][0]})
    assert split["first"] == "S1" and split["second"] == "S2" and 0.5 < split["ratio"] < 0.8
    assert speaker_split({"segment_ids": [0], "start": 0.0, "end": 1.5}, {0: doc["segments"][0]}) is None
    three = {"segment_ids": [0], "start": 0.0, "end": 9.0, "x": 1}
    words = [_word(0, 1, "a", "S1"), _word(1, 2, "b", "S2"), _word(2, 3, "c", "S1")]
    assert speaker_split(three, {0: {"words": words}}) is None


def test_dialogue_text_breaks_near_the_ratio_and_prefers_sentence_ends():
    text = "Where are you? I am at home now."
    assert dialogue_text(text, 0.45, "en") == "- Where are you?\n- I am at home now."
    assert dialogue_text("one two three four", 0.5, "en") == "- one two\n- three four"
    assert dialogue_text("single", 0.5, "en") == "single"
    assert dialogue_text("kept\nbreak", 0.5, "en") == "kept\nbreak"
    assert style.dash("en") == style.dash(None) == "- "


def _unit(split):
    return {"id": 0, "start": 0.0, "end": 3.0, "translation": "Where are you? I am at home now.", "text": "x",
            "segment_ids": [0], "speaker_split": split}


def test_finalize_writes_dash_lines_for_automatic_translations_only():
    split = {"first": "S1", "second": "S2", "ratio": 0.45}
    units = [_unit(split)]
    cues = finalize_units(units, [{"id": 0, "audio_confidence": 0.9}], "en")
    assert cues[0].text == "- Where are you?\n- I am at home now."
    edited = _unit(split)
    edited["reviewed_text"] = "Where are you? I am at home now."
    assert finalize_units([edited], [{"id": 0, "audio_confidence": 0.9}], "en")[0].text.count("\n") == 0
    assert finalize_units([_unit(None)], [{"id": 0, "audio_confidence": 0.9}], "en")[0].text.count("\n") == 0


def test_rtl_cue_keeps_both_dash_lines():
    split = {"first": "S1", "second": "S2", "ratio": 0.5}
    unit = {"id": 0, "start": 0.0, "end": 3.0, "translation": "\u0623\u064a\u0646 \u0623\u0646\u062a\u061f \u0641\u064a \u0627\u0644\u0628\u064a\u062a.",
            "text": "x", "segment_ids": [0], "speaker_split": split}
    cue = finalize_units([unit], [{"id": 0, "audio_confidence": 0.9}], "ar")[0]
    lines = cue.text.split("\n")
    assert len(lines) == 2 and all(line.strip("‏").startswith("- ") for line in lines)


def test_wrap_text_keeps_intentional_breaks_in_a_cue():
    assert wrap_text("- Where are you?\n- At home.", 42, 2, "en") == "- Where are you?\n- At home."
    assert isinstance(Cue(0, 1, "x").speaker, type(None))
