"""Phase 4: re-decoding low-confidence spans, transcription windows, batched decoding (D-091, D-092)."""

import numpy as np
import pytest

from app.core import redecode
from app.core.audio_processor import SAMPLE_RATE
from app.core.errors import JobCancelled
from app.core.transcription import AsrOptions, AsrSegment, FasterWhisperEngine, Word
from app.core.windows import plan_windows
from app.utils.atomic import load_jsonl


def _seg(start, end, text, probs, logprob=-1.0, ratio=1.2):
    step = (end - start) / max(len(probs), 1)
    words = [Word(start + i * step, start + (i + 1) * step, f"w{i}", p) for i, p in enumerate(probs)]
    return AsrSegment(start, end, text, words, logprob, 0.01, ratio).to_dict()


def test_low_confidence_rules():
    assert redecode.is_low_confidence(_seg(0, 3, "a", [0.4, 0.4, 0.4]))              # mean < 0.5
    assert redecode.is_low_confidence(_seg(0, 3, "a", [0.9, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9]))  # run of 3
    assert not redecode.is_low_confidence(_seg(0, 3, "a", [0.9, 0.2, 0.2, 0.9, 0.2, 0.9]))      # run of 2 only
    assert not redecode.is_low_confidence(_seg(0, 3, "a", []))


def test_select_is_worst_first_and_capped():
    segs = [_seg(i * 10, i * 10 + 4, "x", [0.1 + 0.05 * i] * 3) for i in range(6)]
    # each span costs 4 + 2 s; a 20 % cap of 60 s allows two of them: the two lowest-probability ones
    assert redecode.select(segs, 60.0) == [0, 1]
    assert redecode.select(segs, 600.0) == [0, 1, 2, 3, 4, 5]


def test_pick_rules():
    original = _seg(0, 3, "orig", [0.3] * 3, logprob=-1.2)
    good = _seg(0, 3, "better", [0.8] * 3, logprob=-0.4)
    assert redecode.pick(original, [good]) == good
    assert redecode.pick(original, [_seg(0, 3, "worse", [0.8], logprob=-1.5)]) is None
    assert redecode.pick(original, [_seg(0, 3, "loop", [0.8], logprob=-0.2, ratio=3.0)]) is None
    assert redecode.pick(original, [_seg(0, 3, "Altyazı M.K.", [0.8], logprob=-0.2)]) is None
    best = _seg(0, 3, "best", [0.9], logprob=-0.1)
    assert redecode.pick(original, [good, best]) == best


class _SpanEngine:
    def __init__(self):
        self.calls = []

    def transcribe_span(self, audio, language, offset):
        self.calls.append(offset)
        return [AsrSegment(offset + 1.0, offset + 3.0, "fixed text", [Word(offset + 1.0, offset + 3.0, "fixed", 0.9)],
                           -0.3, 0.01, 1.1)]


def _records():
    return [{"segment": _seg(0, 2, "fine", [0.9, 0.9]), "engine": "e", "plan": 0},
            {"segment": _seg(10, 12, "bad", [0.2, 0.2, 0.2], logprob=-1.4), "engine": "e", "plan": 0}]


def test_redecode_replaces_marks_and_resumes(tmp_path):
    audio = np.zeros(SAMPLE_RATE * 30, dtype=np.float32)
    partial = tmp_path / "re.partial.jsonl"
    engine, records = _SpanEngine(), _records()
    info = redecode.redecode(engine, [audio], records, "tr", 30.0, partial, lambda: None)
    assert info["selected"] == 1 and info["replaced"] == 1
    assert records[1]["segment"]["text"] == "fixed text" and records[1]["segment"]["redecoded"] is True
    assert records[0]["segment"]["text"] == "fine"
    assert engine.calls == [9.0]                                   # span start = segment start - 1 s
    again, records2 = _SpanEngine(), _records()
    redecode.redecode(again, [audio], records2, "tr", 30.0, partial, lambda: None)
    assert again.calls == [] and records2[1]["segment"]["text"] == "fixed text"   # finished spans are not repeated
    assert len(load_jsonl(partial)) == 1


def test_redecode_uses_the_second_audio_and_cancels(tmp_path):
    engine = _SpanEngine()
    loads = []

    def other():
        loads.append(1)
        return np.zeros(SAMPLE_RATE * 30, dtype=np.float32)

    redecode.redecode(engine, [np.zeros(SAMPLE_RATE * 30, dtype=np.float32), other], _records(), "tr", 30.0,
                      tmp_path / "a.jsonl", lambda: None)
    assert len(engine.calls) == 2 and loads == [1]

    def cancel():
        raise JobCancelled()
    with pytest.raises(JobCancelled):
        redecode.redecode(_SpanEngine(), [np.zeros(SAMPLE_RATE * 30, dtype=np.float32)], _records(), "tr", 30.0,
                          tmp_path / "b.jsonl", cancel)


def test_redecode_skipped_for_engines_without_span_support(tmp_path):
    records = _records()
    info = redecode.redecode(object(), [np.zeros(10, dtype=np.float32)], records, "tr", 30.0, tmp_path / "c.jsonl",
                             lambda: None)
    assert info["selected"] == 0 and records == _records()


def test_plan_windows_cuts_in_the_longest_pause(monkeypatch):
    import faster_whisper.vad as vad

    audio = np.zeros(SAMPLE_RATE * 1500, dtype=np.float32)

    def fake_speech(chunk, options=None):
        # Searching around 600 s (offset 555 s): speech except a 4 s pause that starts 20 s into the chunk.
        return [{"start": 0, "end": 20 * SAMPLE_RATE}, {"start": 24 * SAMPLE_RATE, "end": len(chunk)}]
    monkeypatch.setattr(vad, "get_speech_timestamps", fake_speech)
    windows = plan_windows(audio)
    assert windows[0] == (0.0, pytest.approx(555 + 22, abs=0.1))
    assert windows[-1][1] == pytest.approx(1500.0)
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))
    assert plan_windows(np.zeros(SAMPLE_RATE * 650, dtype=np.float32)) == [(0.0, 650.0)]


def test_plan_windows_without_a_pause_cuts_at_the_ideal_point(monkeypatch):
    import faster_whisper.vad as vad

    monkeypatch.setattr(vad, "get_speech_timestamps", lambda chunk, options=None: [{"start": 0, "end": len(chunk)}])
    windows = plan_windows(np.zeros(SAMPLE_RATE * 1250, dtype=np.float32))
    assert windows == [(0.0, 600.0), (600.0, 1250.0)]


def _engine(monkeypatch, tmp_path, model, batched=None, batch=8):
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *a, **k: model)
    if batched is not None:
        monkeypatch.setattr(faster_whisper, "BatchedInferencePipeline", lambda m: batched)
    return FasterWhisperEngine(tmp_path, "cpu", "int8", AsrOptions(beam_size=2, batch_size=batch))


class _Seg:
    def __init__(self, start, end, text="t"):
        self.start, self.end, self.text, self.words = start, end, text, []
        self.avg_logprob = self.no_speech_prob = self.compression_ratio = 0.0


def test_batched_decoding_halves_the_batch_on_out_of_memory(monkeypatch, tmp_path):
    sizes = []

    class Batched:
        def transcribe(self, audio, **kw):
            sizes.append(kw["batch_size"])
            if kw["batch_size"] > 2:
                raise RuntimeError("CUDA failed with error out of memory")
            return iter([_Seg(0.0, 1.0), _Seg(1.0, 2.0)]), None

    engine = _engine(monkeypatch, tmp_path, object(), Batched(), batch=8)
    segments = list(engine.transcribe(np.zeros(SAMPLE_RATE * 5, dtype=np.float32), "tr", 100.0, None))
    assert sizes == [8, 4, 2] and [s.start for s in segments] == [100.0, 101.0]
    assert engine._batch == 2


def test_out_of_memory_falls_back_to_the_sequential_decoder_after_the_last_segment(monkeypatch, tmp_path):
    seen = []

    class Batched:
        def transcribe(self, audio, **kw):
            def gen():
                yield _Seg(0.0, 1.0)
                raise RuntimeError("out of memory")
            return gen(), None

    class Model:
        def transcribe(self, audio, **kw):
            seen.append(len(audio))
            return iter([_Seg(0.0, 1.0, "tail")]), None

    engine = _engine(monkeypatch, tmp_path, Model(), Batched(), batch=2)
    segments = list(engine.transcribe(np.zeros(SAMPLE_RATE * 5, dtype=np.float32), "tr", 0.0, None))
    assert [s.text for s in segments] == ["t", "tail"] and [s.start for s in segments] == [0.0, 1.0]
    assert seen == [SAMPLE_RATE * 4]                      # only the audio after the first segment is decoded again
    assert engine._batch == 0


def test_other_runtime_errors_are_not_swallowed(monkeypatch, tmp_path):
    class Batched:
        def transcribe(self, audio, **kw):
            raise RuntimeError("cuDNN failed")

    engine = _engine(monkeypatch, tmp_path, object(), Batched())
    with pytest.raises(RuntimeError, match="cuDNN"):
        list(engine.transcribe(np.zeros(SAMPLE_RATE * 5, dtype=np.float32), "tr", 0.0, None))


def test_vad_parameters_only_include_set_values():
    assert AsrOptions().vad_parameters() == {"threshold": AsrOptions().vad_threshold}
    assert AsrOptions(vad_min_silence_ms=500, vad_max_speech_s=28).vad_parameters()["min_silence_duration_ms"] == 500


# -- pipeline integration ------------------------------------------------------------------------------------------

from app.core.pipeline import ASR_FILTER_VERSION  # noqa: E402
from tests.fakes import FakeAsrEngine, FakeTranslator  # noqa: E402
from tests.test_pipeline import _pipeline, media  # noqa: E402,F401


def test_pipeline_transcribes_window_by_window(tmp_path, media, monkeypatch):
    import app.core.pipeline as pipeline

    monkeypatch.setattr(pipeline, "plan_windows", lambda audio, threshold: [(0.0, 8.0), (8.0, 20.0)])
    engine = FakeAsrEngine()
    result = _pipeline(tmp_path, media, engine, FakeTranslator())[0].run()
    calls = [c for c in engine.calls if "offset" in c]
    assert [c["offset"] for c in calls] == [0.0, 8.0] and calls[0]["samples"] == SAMPLE_RATE * 8
    assert result.stats["stages"]["transcribe"]["segments"] == 10


def test_pipeline_redecodes_low_confidence_segments(tmp_path, media):
    class Engine(FakeAsrEngine):
        def transcribe(self, audio, language, offset, prompt):
            for seg in super().transcribe(audio, language, offset, prompt):
                seg.words[0].probability = 0.1 if seg.start == 4.0 else 0.9
                yield seg

        def transcribe_span(self, audio, language, offset):
            return [AsrSegment(offset + 1.0, offset + 2.0, "Duzeltilmis cumle.",
                               [Word(offset + 1.0, offset + 2.0, "Duzeltilmis", 0.9)], -0.1, 0.01, 1.0)]

    result = _pipeline(tmp_path, media, Engine(), FakeTranslator())[0].run()
    from app.utils.atomic import read_json
    doc = read_json(result.outputs["master_transcript"])
    fixed = [s for s in doc["segments"] if s.get("redecoded")]
    assert len(fixed) == 1 and fixed[0]["text"] == "Duzeltilmis cumle."
    assert result.stats["asr"]["redecoded"] == {"selected": 1, "replaced": 1}


def test_asr_filter_version_covers_the_new_stages():
    assert ASR_FILTER_VERSION >= 4
