"""Difficult-audio handling (D-044) and the Qwen3-ASR engine and modes (D-045)."""

import json

import numpy as np
import pytest

from app.core import enhance, hallucination
from app.core.audio_processor import SAMPLE_RATE
from app.core.modes import Mode
from app.core.pipeline import JobConfig, Pipeline
from app.core.transcription import Word
from app.services.local_runtime import LlamaServer
from app.utils.atomic import read_json
from tests.fakes import CPU_ASR, CPU_MT, Factory, FakeAsrEngine, FakeTranslator
from tests.media import make_tone_file


# -- hallucination filter ---------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Altyazı M.K.", "Thanks for watching!", "♪♪", "...",
                                  "\u0634\u0643\u0631\u0627 \u0644\u0644\u0645\u0634\u0627\u0647\u062f\u0629",
                                  "Subtitles by the Amara.org community", "evet " * 9])
def test_known_hallucinations_are_removed(text):
    assert hallucination.reason({"text": text}) is not None


@pytest.mark.parametrize("segment", [
    {"text": "Polat, bunu bana neden yaptın?"},
    {"text": "Thanks for watching the door, I owe you one."},          # phrase inside real dialogue
    {"text": "Evet evet, tamam.", "no_speech_prob": 0.7, "avg_logprob": -0.3},   # confident words
    {"text": "Altyazı hazır mı dedin sen bana demin abi?"},
])
def test_real_speech_is_kept(segment):
    assert hallucination.reason(segment) is None


@pytest.mark.parametrize("text", ["Altyaz\u0131 a\u00e7\u0131k m\u0131?", "Thank you for watching over him.",
                                  "Subtitles by tomorrow, okay?", "Transcription by hand is slow",
                                  "no, no, no, no, no!"])
def test_dialogue_that_looks_like_credits_is_kept(text):
    assert hallucination.reason({"text": text}) is None


def test_loops_are_removed():
    assert hallucination.reason({"text": "I mean " * 6}) == "repetition"
    assert hallucination.reason({"text": "x " * 30, "compression_ratio": 3.1}) == "repetition"


def test_unsure_non_speech_is_removed():
    assert hallucination.reason({"text": "Hmm.", "no_speech_prob": 0.8, "avg_logprob": -1.4}) == "probably not speech"


# -- audio enhancement ------------------------------------------------------------------------

def _tone(seconds, amplitude, freq=220.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_level_speech_raises_quiet_parts_only():
    quiet, loud, silence = _tone(3, 0.01), _tone(3, 0.3), np.zeros(SAMPLE_RATE * 3, np.float32)
    out = enhance.level_speech(np.concatenate([quiet, silence, loud]))
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    n = SAMPLE_RATE * 3
    assert rms(out[n // 2:n - SAMPLE_RATE // 2]) > 3 * rms(quiet)          # raised (up to +12 dB)
    assert rms(out[n + SAMPLE_RATE:2 * n - SAMPLE_RATE]) < 1e-4            # silence stays silent
    assert abs(rms(out[2 * n + SAMPLE_RATE:]) - rms(loud)) < 0.02          # loud part unchanged
    assert np.max(np.abs(out)) < 1.0


def _fake_separator(calls, accompaniment_scale):
    class FakeSeparator:
        def process(self, sample_rate, samples):
            calls.append(samples.shape)

            class Stem:
                def __init__(self, data):
                    self.data = data

            class Result:
                stems = [Stem(samples * 0.5), Stem(samples * accompaniment_scale)]
            Result.sample_rate = sample_rate
            return Result()
    return FakeSeparator()


def test_separation_is_chunked_and_keeps_length(monkeypatch):
    calls = []
    monkeypatch.setattr(enhance, "_separator", lambda *a: _fake_separator(calls, 2.0))
    audio = _tone(70.5, 0.2)
    out = enhance.separate_speech(audio, None, mix_back=0.0)
    assert len(out) == len(audio) and len(calls) == 3                     # 30 s chunks (+1 s overlap)
    assert all(shape[0] == 2 for shape in calls)                         # stereo input for Spleeter
    # Cross-fades keep the signal (the 16 <-> 44.1 kHz resampling adds a sub-sample delay).
    assert np.max(np.abs(out[100:-100] - audio[100:-100] * 0.5)) < 0.005


def test_separation_keeps_chunks_without_music_as_recorded(monkeypatch):
    """Accompaniment/vocals ratio 0.2 (<= 0.5): the separated voice is not used (D-094)."""
    monkeypatch.setattr(enhance, "_separator", lambda *a: _fake_separator([], 0.1))
    audio = _tone(40.0, 0.2)
    out = enhance.separate_speech(audio, None)
    assert np.max(np.abs(out[100:-100] - audio[100:-100])) < 1e-6


def test_separation_mixes_the_original_back_in_musical_chunks(monkeypatch):
    monkeypatch.setattr(enhance, "_separator", lambda *a: _fake_separator([], 2.0))
    audio = _tone(40.0, 0.2)
    out = enhance.separate_speech(audio, None)
    expected = audio * (0.5 + enhance.MIX_BACK)
    assert enhance.MIX_BACK == pytest.approx(0.1778, abs=1e-3)
    assert np.max(np.abs(out[100:-100] - expected[100:-100])) < 0.005


# -- Qwen3-ASR ----------------------------------------------------------------------------------

def test_llama_server_command_for_audio_models(tmp_path):
    server = LlamaServer(tmp_path / "llama-server.exe", tmp_path / "m.gguf", gpu=False,
                         mmproj=tmp_path / "mmproj.gguf", jinja=True)
    server.port = 1234
    cmd = server._command()
    assert "--jinja" in cmd and "--no-jinja" not in cmd
    assert cmd[cmd.index("--mmproj") + 1].endswith("mmproj.gguf") and "--no-mmproj-offload" in cmd
    plain = LlamaServer(tmp_path / "llama-server.exe", tmp_path / "m.gguf", gpu=True)
    plain.port = 1234
    assert "--no-jinja" in plain._command() and "--mmproj" not in plain._command()


# -- pipeline modes -----------------------------------------------------------------------------

class CreditsAsr(FakeAsrEngine):
    """Segment 4 is a subtitle credit Whisper invented over music."""

    def transcribe(self, audio, language, offset, initial_prompt):
        for seg in super().transcribe(audio, language, offset, initial_prompt):
            if seg.start == 8.0:
                seg.text = "Altyazı M.K."
                seg.words = [Word(seg.start, seg.end, seg.text, 0.3)]
            yield seg


def _pipe(tmp_path, asr, **kw):
    media = make_tone_file(tmp_path / "Episode 1.m4a", seconds=20.0)
    mode = kw.pop("mode", Mode.BALANCED)
    config = JobConfig(media, "tr", "ar", mode, output_dir=tmp_path / "out")
    return Pipeline(config, tmp_path / "jobs", [CPU_ASR], [CPU_MT], Factory({(CPU_ASR.model, "cpu"): asr}),
                    Factory({(CPU_MT.model, "cpu"): FakeTranslator()}), **kw)


def test_enhancement_runs_in_maximum_accuracy_and_is_cached(tmp_path):
    calls = []

    class SeenAsr(FakeAsrEngine):
        def transcribe(self, audio, *args):
            calls.append(("asr", float(np.max(np.abs(audio)))))
            yield from super().transcribe(audio, *args)

    def enhancer(audio, progress, cancel):
        calls.append(("enhance", len(audio)))
        progress(1.0)
        return audio * 0.0 + 0.25

    pipe = _pipe(tmp_path, SeenAsr(), mode=Mode.MAXIMUM_ACCURACY, enhancer=enhancer)
    pipe.run()
    assert calls[0][0] == "enhance" and calls[1] == ("asr", 0.25)
    assert list((tmp_path / "jobs").glob("*/enhanced.*.npy"))
    calls.clear()
    _pipe(tmp_path, SeenAsr(), mode=Mode.MAXIMUM_ACCURACY, enhancer=enhancer).run()
    assert calls == []                                                   # transcription cached

    calls.clear()
    _pipe(tmp_path, SeenAsr(), mode=Mode.BALANCED, enhancer=enhancer).run()   # "auto": off in Balanced
    assert [c[0] for c in calls] == ["asr"]


def test_failed_enhancement_falls_back_to_original_audio(tmp_path):
    def broken(audio, progress, cancel):
        raise RuntimeError("model download failed")

    pipe = _pipe(tmp_path, FakeAsrEngine(), asr_settings={"enhance": "on"}, enhancer=broken)
    result = pipe.run()
    assert any("Audio enhancement failed" in w for w in result.warnings)
    assert read_json(result.outputs["master_transcript"])["segments"]


def test_hallucinated_credit_is_removed_in_the_pipeline(tmp_path):
    result = _pipe(tmp_path, CreditsAsr()).run()
    texts = [s["text"] for s in read_json(result.outputs["master_transcript"])["segments"]]
    assert "Altyaz\u0131 M.K." not in texts and len(texts) == 9
    assert result.stats["asr"]["hallucinations_removed"] == 1


def test_fallback_transcript_is_not_cached_as_the_first_plan(tmp_path):
    from tests.fakes import GPU_ASR

    media = make_tone_file(tmp_path / "Episode 1.m4a", seconds=8.0)
    config = JobConfig(media, "tr", "ar", Mode.BALANCED, output_dir=tmp_path / "out")
    asr = Factory({(GPU_ASR.model, "cuda"): FakeAsrEngine(name="faster-whisper:small:cuda:f16", fail_after=1),
                   (CPU_ASR.model, "cpu"): FakeAsrEngine(name="faster-whisper:tiny:cpu:int8")})
    mt = Factory({(CPU_MT.model, "cpu"): FakeTranslator()})
    first = Pipeline(config, tmp_path / "jobs", [GPU_ASR, CPU_ASR], [CPU_MT], asr, mt).run()
    assert any("fallback engine" in w for w in first.warnings)
    asr2 = Factory({(GPU_ASR.model, "cuda"): FakeAsrEngine(name="faster-whisper:small:cuda:f16"),
                    (CPU_ASR.model, "cpu"): FakeAsrEngine(name="faster-whisper:tiny:cpu:int8")})
    second = Pipeline(config, tmp_path / "jobs", [GPU_ASR, CPU_ASR], [CPU_MT], asr2,
                      Factory({(CPU_MT.model, "cpu"): FakeTranslator()})).run()
    assert not second.stats["stages"]["transcribe"].get("cached")         # transcribed again on the GPU
    engines = {s["sources"][0]["engine"] for s in read_json(second.outputs["master_transcript"])["segments"]}
    assert engines == {"faster-whisper:small:cuda:f16"}
