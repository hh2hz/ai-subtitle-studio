"""The faster-whisper wrapper: regression tests for what it passes into the library.

No model is loaded: `faster_whisper.WhisperModel` is replaced by a recorder.
"""

import numpy as np
import pytest

from app.core.audio_processor import SAMPLE_RATE
from app.core.transcription import AsrOptions, FasterWhisperEngine


def test_detect_language_passes_vad_options(monkeypatch, tmp_path):
    """faster-whisper 1.2.1 gives `vad_parameters` straight to `get_speech_timestamps`, which reads attributes:
    a dict there raised `AttributeError: 'dict' object has no attribute 'threshold'` on every Auto Detect run."""
    import faster_whisper
    from faster_whisper.vad import VadOptions

    calls: list[dict] = []

    class FakeModel:
        def detect_language(self, audio, vad_filter=False, vad_parameters=None, language_detection_segments=1):
            calls.append({"vad_parameters": vad_parameters, "vad_filter": vad_filter,
                          "segments": language_detection_segments, "samples": len(audio)})
            return "tr", 0.9, None

    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *args, **kwargs: FakeModel())
    options = AsrOptions(beam_size=1)
    engine = FasterWhisperEngine(tmp_path, "cpu", "int8", options)

    language, probability = engine.detect_language(np.zeros(SAMPLE_RATE * 10, dtype=np.float32))

    assert language == "tr" and probability == pytest.approx(0.9)
    call = calls[0]
    assert isinstance(call["vad_parameters"], VadOptions)
    assert call["vad_parameters"].threshold == options.vad_threshold
    assert call["vad_filter"] is True and call["samples"] == SAMPLE_RATE * 10


def test_transcribe_keeps_dict_vad_parameters(monkeypatch, tmp_path):
    """`transcribe` converts a dict itself and adds `max_speech_duration_s=chunk_length`; that path stays a dict."""
    import faster_whisper

    calls: list[dict] = []

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return iter(()), None

    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *args, **kwargs: FakeModel())
    engine = FasterWhisperEngine(tmp_path, "cpu", "int8", AsrOptions(beam_size=2))

    assert list(engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), "tr", 0.0, None)) == []
    assert calls[0]["vad_parameters"] == {"threshold": AsrOptions(beam_size=2).vad_threshold}
    assert calls[0]["beam_size"] == 2
