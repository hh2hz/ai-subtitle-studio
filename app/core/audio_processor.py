"""Audio extraction with PyAV (FFmpeg libraries bundled in the av wheel).

Output: 16 kHz mono 16-bit PCM WAV, the input format Whisper expects. No loudness processing is
applied by default because aggressive processing can damage intelligibility.
"""

from __future__ import annotations

import logging
import os
import threading
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from app.core.errors import JobCancelled, PipelineError

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def extract_audio(
    source: Path,
    wav_path: Path,
    cancel: threading.Event | None = None,
    progress: Callable[[float], None] | None = None,
) -> float:
    """Decode the first audio stream of source into wav_path. Returns the duration in seconds."""
    import av

    tmp_path = wav_path.with_name(wav_path.name + ".part")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    samples_written = 0
    skipped_packets = 0
    try:
        container = av.open(str(source))
    except (av.error.FFmpegError, OSError) as exc:
        raise PipelineError(f"Cannot open media file {source}: {exc}", "error.media_open", path=str(source)) from exc
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise PipelineError(f"No audio stream in {source}", "error.no_audio", path=str(source))
        duration = float(container.duration / av.time_base) if container.duration else None
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        with wave.open(str(tmp_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)

            def _write(frames) -> None:
                nonlocal samples_written
                for out in frames:
                    pcm = out.to_ndarray().reshape(-1).astype(np.int16, copy=False)
                    wav.writeframes(pcm.tobytes())
                    samples_written += pcm.size

            for packet in container.demux(stream):
                if cancel is not None and cancel.is_set():
                    raise JobCancelled()
                try:
                    frames = packet.decode()
                except av.error.InvalidDataError:
                    # Skip corrupt packets instead of failing the whole file.
                    skipped_packets += 1
                    continue
                for frame in frames:
                    _write(resampler.resample(frame))
                if progress and duration:
                    progress(min(samples_written / SAMPLE_RATE / duration, 1.0))
            _write(resampler.resample(None))
    except BaseException:
        container.close()
        tmp_path.unlink(missing_ok=True)
        raise
    container.close()
    if samples_written == 0:
        tmp_path.unlink(missing_ok=True)
        raise PipelineError(f"Audio stream of {source} decoded to zero samples", "error.no_audio", path=str(source))
    if skipped_packets:
        log.warning("Skipped %d corrupt audio packets in %s", skipped_packets, source)
    os.replace(tmp_path, wav_path)
    seconds = samples_written / SAMPLE_RATE
    log.info("Extracted %.1f s of audio to %s", seconds, wav_path)
    return seconds


def load_wav(path: Path) -> np.ndarray:
    """Load a 16 kHz mono 16-bit WAV as float32 in [-1, 1].

    Read in blocks into one preallocated float32 array: the old whole-file `readframes` + `astype` held the
    bytes, an int16 view and the float32 copy at the same time (D-096).
    """
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise PipelineError(f"Unexpected WAV format in {path}")
        total = wav.getnframes()
        out = np.empty(total, dtype=np.float32)
        block = SAMPLE_RATE * 60
        for first in range(0, total, block):
            data = np.frombuffer(wav.readframes(min(block, total - first)), dtype=np.int16)
            np.multiply(data, np.float32(1 / 32768.0), out=out[first:first + len(data)])
    return out
