import threading

import pytest

from app.core.audio_processor import SAMPLE_RATE, extract_audio, load_wav
from app.core.errors import JobCancelled, PipelineError
from tests.media import make_tone_file, make_video_only_file


def test_extract_resamples_to_16k_mono(tmp_path):
    src = make_tone_file(tmp_path / "tone.m4a", seconds=3.0, rate=44100, channels=2)
    wav = tmp_path / "out" / "audio.wav"
    seen = []
    duration = extract_audio(src, wav, progress=seen.append)
    assert duration == pytest.approx(3.0, abs=0.1)
    audio = load_wav(wav)
    assert audio.ndim == 1 and len(audio) == pytest.approx(3.0 * SAMPLE_RATE, abs=0.1 * SAMPLE_RATE)
    assert 0.2 < abs(audio).max() <= 1.0
    assert seen and seen[-1] <= 1.0
    assert not list(wav.parent.glob("*.part"))


def test_no_audio_stream(tmp_path):
    src = make_video_only_file(tmp_path / "video.mp4")
    with pytest.raises(PipelineError) as info:
        extract_audio(src, tmp_path / "a.wav")
    assert info.value.ui_key == "error.no_audio"


def test_not_media(tmp_path):
    src = tmp_path / "x.mp4"
    src.write_bytes(b"this is not a media file" * 100)
    with pytest.raises(PipelineError) as info:
        extract_audio(src, tmp_path / "a.wav")
    assert info.value.ui_key in ("error.media_open", "error.no_audio")


def test_cancel_leaves_no_output(tmp_path):
    src = make_tone_file(tmp_path / "tone.m4a", seconds=2.0)
    cancel = threading.Event()
    cancel.set()
    wav = tmp_path / "a.wav"
    with pytest.raises(JobCancelled):
        extract_audio(src, wav, cancel)
    assert not wav.exists() and not list(tmp_path.glob("*.part"))
