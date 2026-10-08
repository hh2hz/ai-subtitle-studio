"""Task 3.1: diarization stage (D-098). No model is loaded: the diarizer is a fake."""

import numpy as np
import pytest

from app.core import diarize
from app.core.errors import JobCancelled
from app.utils.atomic import read_json
from tests.fakes import FakeAsrEngine, FakeTranslator
from tests.test_pipeline import _pipeline, media  # noqa: F401


def test_link_speakers_merges_by_similarity_and_never_inside_a_window():
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    windows = [
        {"segments": [[0.0, 2.0, 0], [2.0, 4.0, 1]], "embeddings": {"0": a, "1": b}},
        # in window 2 local 0 is person B and local 1 is person A; a third voice (local 2) is new
        {"segments": [[10.0, 12.0, 0], [12.0, 14.0, 1], [14.0, 15.0, 2]],
         "embeddings": {"0": [0.05, 1.0, 0.0], "1": [1.0, 0.1, 0.0], "2": [0.0, 0.0, 1.0]}},
    ]
    turns = diarize.link_speakers(windows)
    assert [t[2] for t in turns] == [0, 1, 1, 0, 2]


def test_two_speakers_of_one_window_are_not_merged_even_when_similar():
    window = {"segments": [[0.0, 1.0, 0], [1.0, 2.0, 1]], "embeddings": {"0": [1.0, 0.0], "1": [1.0, 0.01]}}
    assert [t[2] for t in diarize.link_speakers([window])] == [0, 1]


def test_speaker_without_embedding_becomes_its_own_speaker():
    window = {"segments": [[0.0, 0.5, 0]], "embeddings": {"0": None}}
    assert diarize.link_speakers([window, window])[1][2] == 1


def test_assign_uses_the_largest_overlap_and_leaves_unmatched_segments_empty():
    segments = [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 6.0}, {"start": 20.0, "end": 22.0}]
    turns = [(0.0, 1.0, 0), (1.0, 4.5, 1), (4.5, 6.0, 0)]
    assert diarize.assign(segments, turns) == {0: 1, 1: 0}


def test_apply_labels_names_speakers_in_order_of_appearance():
    doc = {"segments": [{"start": 0.0, "end": 2.0, "speaker": None}, {"start": 2.0, "end": 3.0, "speaker": None},
                       {"start": 3.0, "end": 5.0, "speaker": None}]}
    speakers = diarize.apply_labels(doc, [(0.0, 2.0, 7), (2.0, 3.0, 3), (3.0, 5.0, 7)])
    assert [s["speaker"] for s in doc["segments"]] == ["S1", "S2", "S1"]
    assert speakers == {"S1": {"seconds": 4.0, "segments": 2}, "S2": {"seconds": 1.0, "segments": 1}}


def test_model_pins_are_complete():
    for model in (diarize.SEGMENTATION, diarize.EMBEDDING):
        assert len(model.sha256) == 64 and model.url.startswith("https://github.com/k2-fsa/sherpa-onnx/")


def test_download_checks_the_checksum_before_use(tmp_path, monkeypatch):
    import io

    class Response(io.BytesIO):
        headers = {"Content-Length": "3"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(diarize.urllib.request, "urlopen", lambda *a, **k: Response(b"bad"))
    with pytest.raises(Exception, match="checksum"):
        diarize.ensure_models(tmp_path)
    assert not list(tmp_path.rglob("*.onnx")) and not list(tmp_path.rglob("*.part"))


class _Diarizer:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def __call__(self, audio, windows, partial, progress=None, cancel=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model missing")
        progress(1.0)
        return [(0.0, 10.0, 0), (10.0, 20.0, 1)]


def test_pipeline_labels_speakers_caches_and_exports(tmp_path, media):  # noqa: F811
    diarizer = _Diarizer()
    result = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), diarizer=diarizer)[0].run()
    doc = read_json(result.outputs["master_transcript"])
    assert {s["speaker"] for s in doc["segments"]} == {"S1", "S2"} and set(doc["speakers"]) == {"S1", "S2"}
    assert result.stats["stages"]["diarize"]["speakers"] == 2
    second = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), diarizer=diarizer)[0].run()
    assert diarizer.calls == 1 and second.stats["stages"]["diarize"]["cached"]


def test_pipeline_continues_when_diarization_fails_and_skips_when_disabled(tmp_path, media):  # noqa: F811
    result = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), diarizer=_Diarizer(fail=True))[0].run()
    assert any("Speaker detection failed" in w for w in result.warnings)
    assert all(s["speaker"] is None for s in read_json(result.outputs["master_transcript"])["segments"])
    off = _Diarizer()
    result = _pipeline(tmp_path / "b", media, FakeAsrEngine(), FakeTranslator(), diarizer=off,
                       asr_settings={"diarization": False})[0].run()
    assert off.calls == 0 and result.stats["stages"]["diarize"] == {"skipped": "disabled"}


def test_pipeline_cancel_during_diarization_propagates(tmp_path, media):  # noqa: F811
    def cancelling(audio, windows, partial, progress=None, cancel=None):
        raise JobCancelled()
    with pytest.raises(JobCancelled):
        _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), diarizer=cancelling)[0].run()


def test_diarization_setting_defaults_on(tmp_path):
    from app.database.database import Database
    from app.database.settings import DEFAULTS, Settings

    assert DEFAULTS["diarization"] is True
    settings = Settings(Database(tmp_path / "s.db"))
    settings.set("diarization", False)
    assert settings.get("diarization") is False
    with pytest.raises(TypeError):
        settings.set("diarization", "yes")


def test_diarize_resumes_finished_windows(tmp_path, monkeypatch):
    built = []

    class FakeSegments(list):
        def sort_by_start_time(self):
            return self

    class Turn:
        def __init__(self, start, end, speaker):
            self.start, self.end, self.speaker = start, end, speaker

    class Sd:
        def process(self, chunk):
            return FakeSegments([Turn(0.0, 2.0, 0)])

    class Extractor:
        def create_stream(self):
            class S:
                def accept_waveform(self, **kw):
                    pass

                def input_finished(self):
                    pass
            return S()

        def compute(self, stream):
            return [1.0, 0.0]

    monkeypatch.setattr(diarize, "_build", lambda *a: built.append(1) or (Sd(), Extractor()))
    audio = np.zeros(16000 * 20, dtype=np.float32)
    partial = tmp_path / "d.partial.jsonl"
    windows = [(0.0, 10.0), (10.0, 20.0)]
    cancel = __import__("threading").Event()

    class Progress:
        def __call__(self, f):
            if f >= 0.5:
                cancel.set()
    with pytest.raises(JobCancelled):
        diarize.diarize(audio, tmp_path, windows, partial, progress=Progress(), cancel=cancel)
    assert len(partial.read_text().splitlines()) == 1
    turns = diarize.diarize(audio, tmp_path, windows, partial)
    assert [t[:2] for t in turns] == [(0.0, 2.0), (10.0, 12.0)] and {t[2] for t in turns} == {0}
    assert len(built) == 2 and len(partial.read_text().splitlines()) == 2
