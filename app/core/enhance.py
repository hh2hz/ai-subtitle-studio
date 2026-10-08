"""Clean-up of difficult audio before speech recognition (D-044).

1. Speech separation: Spleeter 2-stems (Deezer, MIT) run through sherpa-onnx (Apache-2.0, onnxruntime, CPU)
   keeps the voice and removes music and background noise. Measured RTF about 0.05 with 2 threads.
2. Gentle level control: quiet passages are raised (up to +12 dB, never lowered), peaks are soft-limited.

Input and output are the 16 kHz mono float arrays the ASR uses; the separator works in 30 s chunks with a
cross-faded 1 s overlap so long episodes need little memory.
"""

from __future__ import annotations

import hashlib
import logging
import tarfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from app.core.audio_processor import SAMPLE_RATE
from app.core.errors import JobCancelled, PipelineError

log = logging.getLogger(__name__)

Progress = Callable[[float], None]


@dataclass(frozen=True)
class SeparatorModel:
    url: str
    sha256: str
    folder: str
    vocals: str
    accompaniment: str


SPLEETER = SeparatorModel(
    url="https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/"
        "sherpa-onnx-spleeter-2stems-fp16.tar.bz2",
    sha256="d54561979bd2e08a51e7dbd99ac36bb47564e089eefd403636dbca93e811bba2",
    folder="sherpa-onnx-spleeter-2stems-fp16", vocals="vocals.fp16.onnx", accompaniment="accompaniment.fp16.onnx")
MODEL_RATE = 44100
CHUNK_S = 30.0
OVERLAP_S = 1.0
VERSION = 2          # part of the transcription cache key (2: per-chunk gate and original mix-back, D-094)
GATE_RATIO = 0.5     # separate only chunks whose accompaniment/vocals RMS ratio is above this
MIX_BACK = 10 ** (-15 / 20)   # -15 dB of the original is mixed back into separated chunks


def ensure_model(models_dir: Path, model: SeparatorModel = SPLEETER) -> Path:
    target = Path(models_dir) / "separation" / model.folder
    if (target / model.vocals).is_file() and (target / model.accompaniment).is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.parent / (model.folder + ".tar.bz2.part")
    log.info("Downloading the speech separation model (%s)", model.url)
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(model.url, timeout=60) as response, open(archive, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
    except OSError as exc:
        archive.unlink(missing_ok=True)
        raise PipelineError(f"Speech separation model download failed: {exc}") from exc
    if digest.hexdigest() != model.sha256:
        archive.unlink(missing_ok=True)
        raise PipelineError("Speech separation model checksum mismatch")
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.isfile() and name in (model.vocals, model.accompaniment):
                target.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(target / name, "wb") as dst:
                    dst.write(src.read())
    archive.unlink(missing_ok=True)
    return target


def _separator(model_dir: Path, model: SeparatorModel, threads: int):
    import sherpa_onnx

    config = sherpa_onnx.OfflineSourceSeparationConfig(model=sherpa_onnx.OfflineSourceSeparationModelConfig(
        spleeter=sherpa_onnx.OfflineSourceSeparationSpleeterModelConfig(
            vocals=str(model_dir / model.vocals), accompaniment=str(model_dir / model.accompaniment)),
        num_threads=threads, debug=False, provider="cpu"))
    if not config.validate():
        raise PipelineError("Invalid speech separation model configuration")
    return sherpa_onnx.OfflineSourceSeparation(config)


def _resample(signal: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Band-limited resampling with FFmpeg's resampler (PyAV)."""
    if rate == target:
        return signal.astype(np.float32)
    import av

    resampler = av.AudioResampler(format="flt", layout="mono", rate=target)
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(signal.astype(np.float32)[None, :]),
                                       format="flt", layout="mono")
    frame.sample_rate = rate
    parts = [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
    parts += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)))) if len(signal) else 0.0


def separate_speech(audio: np.ndarray, model_dir: Path, model: SeparatorModel = SPLEETER, threads: int = 4,
                    progress: Progress | None = None, cancel: threading.Event | None = None,
                    gate_ratio: float = GATE_RATIO, mix_back: float = MIX_BACK) -> np.ndarray:
    """Voice-only version of a 16 kHz mono signal (same length).

    Per 30 s chunk the accompaniment/vocals RMS ratio decides: below `gate_ratio` the chunk is speech over little
    music, separation can only damage it, so the original chunk is kept; above it the separated voice is used with
    `mix_back` of the original mixed in (artefact masking). `gate_ratio=0, mix_back=0` is the version-1 behaviour."""
    import time

    started = time.monotonic()
    separator = _separator(model_dir, model, threads)
    loaded = time.monotonic()
    audio = np.asarray(audio, dtype=np.float32)
    total = len(audio)
    gated = 0
    step = int(CHUNK_S * SAMPLE_RATE)
    overlap = int(OVERLAP_S * SAMPLE_RATE)
    out = np.zeros(total, np.float32)
    fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    start = 0
    while start < total:
        if cancel is not None and cancel.is_set():
            raise JobCancelled()
        end = min(total, start + step + overlap)
        chunk = audio[start:end]
        # The model needs a few seconds of signal; a short tail is padded with silence.
        padded = np.pad(chunk, (0, max(0, 2 * SAMPLE_RATE - len(chunk))))
        # Spleeter works at 44.1 kHz; resampling here avoids sherpa-onnx's own resampler (and its console log).
        wide = _resample(padded, SAMPLE_RATE, MODEL_RATE)
        result = separator.process(sample_rate=MODEL_RATE, samples=np.ascontiguousarray(np.stack([wide, wide])))
        vocals = result.stems[0].data.mean(axis=0)
        accompaniment = result.stems[1].data.mean(axis=0)
        ratio = _rms(accompaniment) / max(_rms(vocals), 1e-9)
        vocals = _resample(vocals, result.sample_rate)
        vocals = np.pad(vocals, (0, max(0, len(chunk) - len(vocals))))[: len(chunk)]
        if ratio <= gate_ratio and gate_ratio > 0:
            vocals = chunk.copy()
            gated += 1
        elif mix_back:
            vocals = vocals + mix_back * chunk
        if start > 0 and overlap:
            head = min(overlap, len(vocals))
            out[start:start + head] = out[start:start + head] * (1 - fade_in[:head]) + vocals[:head] * fade_in[:head]
            out[start + head:end] = vocals[head:]
        else:
            out[start:end] = vocals
        start += step
        if progress:
            progress(min(start / total, 1.0))
    log.info("Voice separation: %.1f s for %.1f s of audio (%d threads, model load %.1f s); "
             "%d chunks kept as recorded (little music)",
             time.monotonic() - loaded, total / SAMPLE_RATE, threads, loaded - started, gated)
    return out


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """Centered moving average (edge-padded, same length) in O(n) with a cumulative sum."""
    padded = np.pad(values.astype(np.float64), (width // 2, width - width // 2 - 1), mode="edge")
    total = np.concatenate([[0.0], np.cumsum(padded)])
    return (total[width:] - total[:-width]) / width


def level_speech(audio: np.ndarray, max_gain_db: float = 12.0, target_rms: float = 0.08,
                 window_s: float = 0.5) -> np.ndarray:
    """Raise quiet passages smoothly (never lowers loud ones); hard peaks stay below full scale.

    The level is measured on 10 ms frames, so a 100-minute episode takes seconds (a per-sample convolution took
    about 300 s for 4.6 minutes on the user's PC and hours for a full episode; D-046).
    """
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    if not n:
        return audio
    hop = SAMPLE_RATE // 100
    frames = -(-n // hop)
    padded = np.pad(audio, (0, frames * hop - n))
    energy = np.einsum("ij,ij->i", padded.reshape(frames, hop), padded.reshape(frames, hop)).astype(np.float64) / hop
    win = max(1, round(window_s * SAMPLE_RATE / hop))
    rms = np.sqrt(_moving_average(energy, win)) + 1e-6
    gain = np.clip(target_rms / rms, 1.0, 10 ** (max_gain_db / 20))
    gain[rms < 0.003] = 1.0                       # leave near-silence alone (no noise pumping)
    smooth = _moving_average(gain, 2 * win + 1)
    centers = (np.arange(frames) + 0.5) * hop
    result = np.empty(n, np.float32)
    step = 10_000_000                             # interpolate in blocks to keep memory low
    for first in range(0, n, step):
        idx = np.arange(first, min(n, first + step), dtype=np.float64)
        result[first:first + len(idx)] = audio[first:first + len(idx)] * np.interp(idx, centers, smooth)
    # Soft limiter instead of scaling the whole file down because of a few peaks.
    knee = 0.9
    over = np.abs(result) > knee
    result[over] = np.sign(result[over]) * (knee + 0.09 * np.tanh((np.abs(result[over]) - knee) / 0.09))
    return result
