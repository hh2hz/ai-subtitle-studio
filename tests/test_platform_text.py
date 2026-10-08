"""Task 4.6: manual platform subtitles complete the ASR text (D-095)."""

from app.core import platform_text
from app.providers.base import AUTO, COMMUNITY, MANUAL


def _doc():
    seg = lambda i, s, e, t: {"id": i, "start": s, "end": e, "text": t, "sources": [{"type": "asr", "engine": "e"}],
                              "words": [{"start": s, "end": e, "text": t, "probability": 0.9}]}
    return {"language": {"code": "tr"}, "segments": [
        seg(0, 0.0, 3.0, "Ben bunu bilmiyorum"), seg(1, 3.0, 6.0, "Nereye gidiyorsun"), seg(2, 6.0, 9.0, "tamam")]}


def _track(kind=MANUAL, language="tr", agreement=0.8, cues=None):
    cues = cues if cues is not None else [
        {"start": 0.1, "end": 2.9, "text": "Ben bunu bilmiyorum, sen biliyor musun?"},
        {"start": 3.1, "end": 5.9, "text": "Nereye gidiyorsun"},
        {"start": 6.0, "end": 9.0, "text": "Tamam."}]
    return {"provider": "youtube", "kind": kind, "language": language, "audio_agreement": agreement,
            "reference": "tr", "cues": cues}


def test_longer_agreeing_text_replaces_asr_text_and_keeps_timing():
    doc = _doc()
    changes = platform_text.apply(doc, [_track()], 0.6)
    first = doc["segments"][0]
    assert first["text"] == "Ben bunu bilmiyorum, sen biliyor musun?" and first["asr_text"] == "Ben bunu bilmiyorum"
    assert (first["start"], first["end"]) == (0.0, 3.0) and first["source"] == "platform_subtitle"
    assert first["sources"][-1]["type"] == "platform_subtitle"
    assert [w["text"] for w in first["words"]][:2] == ["Ben", "bunu"] and first["words"][0]["probability"] is None
    assert first["words"][-1]["end"] == 3.0
    assert [c["id"] for c in changes] == [0]                        # equal text and shorter text are left alone
    assert doc["segments"][1]["text"] == "Nereye gidiyorsun" and "source" not in doc["segments"][1]


def test_automatic_community_other_language_and_weak_tracks_are_never_used():
    for track in (_track(kind=AUTO), _track(kind=COMMUNITY), _track(language="en"), _track(agreement=0.5)):
        doc = _doc()
        assert platform_text.apply(doc, [track], 0.6) == []
        assert doc["segments"][0]["text"] == "Ben bunu bilmiyorum"


def test_unrelated_text_is_not_adopted_and_cues_between_segments_go_to_the_best_overlap():
    doc = _doc()
    cues = [{"start": 0.0, "end": 3.0, "text": "Completely different words here and many more of them"},
            {"start": 2.9, "end": 6.0, "text": "Nereye gidiyorsun sen"}]
    changes = platform_text.apply(doc, [_track(cues=cues)], 0.6)
    assert doc["segments"][0]["text"] == "Ben bunu bilmiyorum"
    assert doc["segments"][1]["text"] == "Nereye gidiyorsun sen" and len(changes) == 1


def test_pipeline_completes_the_text_from_manual_subtitles(tmp_path):
    from app.core.metadata import SeriesInfo  # noqa: F401
    from app.utils.atomic import read_json
    from tests.fakes import FakeDownloader, matching_cues, vtt_from
    from tests.media import make_tone_file
    from tests.test_pipeline_m2 import _run

    media = make_tone_file(tmp_path / "src.m4a", seconds=20.0)
    cues = matching_cues()
    cues[1] = (cues[1][0], cues[1][1], "Yarim cumle 1 ve devami burada")
    info = {"id": "x", "title": "Clip", "duration": 20, "subtitles": {"tr": []}, "automatic_captions": {}}
    result = _run(tmp_path, FakeDownloader(media, info=info, subtitle_texts={("tr", False): vtt_from(cues)}))
    doc = read_json(result.outputs["master_transcript"])
    seg = doc["segments"][1]
    assert seg["text"] == "Yarim cumle 1 ve devami burada" and seg["source"] == "platform_subtitle"
    assert seg["asr_text"] == "Yarim cumle 1"
    assert result.stats["platform_text"]["replaced"] == 1
