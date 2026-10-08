"""ASR engine interface and the faster-whisper implementation."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from app.core.audio_processor import SAMPLE_RATE

log = logging.getLogger(__name__)


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float | None      # None: estimated word (engine without word timestamps)


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    redecoded: bool = False        # replaced by a second, slower decode of a low-confidence span (D-091)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AsrSegment":
        words = [Word(**w) for w in data.get("words", [])]
        return cls(**{**data, "words": words})


@dataclass(frozen=True)
class AsrOptions:
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    # Disabled to limit repetition loops on long audio; resume passes the last text as a prompt instead.
    condition_on_previous_text: bool = False
    # Silero VAD speech threshold (faster-whisper default 0.5). Lower keeps quiet speech and speech over music;
    # on episode 61 the default removed 59 % of the audio (D-044).
    vad_threshold: float = 0.35
    # Other Silero VAD parameters; None keeps the faster-whisper default (2000 ms, 400 ms, unlimited).
    vad_min_silence_ms: int | None = None
    vad_speech_pad_ms: int | None = None
    vad_max_speech_s: float | None = None
    # Seconds of silence in which a hallucinated segment is skipped (needs word timestamps); None disables (D-089).
    hallucination_silence_threshold: float | None = 2.0
    # faster-whisper BatchedInferencePipeline batch size; 0 or 1 = the sequential decoder (D-092).
    batch_size: int = 0
    # Batched decoding without timestamp tokens returns one long segment per 30 s chunk (8 instead of 73 segments on a
    # 5-minute clip, D-092); decoding with timestamp tokens keeps sentence-sized segments.
    batch_without_timestamps: bool = True

    def vad_parameters(self) -> dict:
        params: dict = {"threshold": self.vad_threshold}
        for key, value in (("min_silence_duration_ms", self.vad_min_silence_ms),
                           ("speech_pad_ms", self.vad_speech_pad_ms),
                           ("max_speech_duration_s", self.vad_max_speech_s)):
            if value is not None:
                params[key] = value
        return params


class AsrEngine(Protocol):
    name: str

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        ...

    def transcribe(self, audio: np.ndarray, language: str, offset: float,
                   initial_prompt: str | None) -> Iterator[AsrSegment]:
        """Yield segments of audio (which starts at offset seconds) with absolute timestamps."""
        ...


class FasterWhisperEngine:
    def __init__(self, model_dir: Path, device: str, compute_type: str, options: AsrOptions):
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper:{Path(model_dir).name}:{device}:{compute_type}"
        self._options = options
        self._batch = options.batch_size
        self._batched = None
        self._model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)
        log.info("Loaded ASR engine %s", self.name)

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Vote over 30 s speech windows from the start, middle and end of the file: an opening song or a
        foreign-language cold open must not decide the language of the whole episode (D-047)."""
        from faster_whisper.vad import VadOptions

        # `WhisperModel.detect_language` hands `vad_parameters` straight to `get_speech_timestamps`, which reads
        # attributes, so it must be a VadOptions object here (unlike `transcribe`, which converts a dict itself).
        vad = VadOptions(threshold=self._options.vad_threshold)
        size = SAMPLE_RATE * 600
        starts = sorted({0, max(0, len(audio) // 2 - size // 2), max(0, len(audio) - size)})
        votes: dict[str, float] = {}
        for start in starts:
            window = audio[start:start + size]
            if len(window) < SAMPLE_RATE * 5:
                continue
            language, probability, _ = self._model.detect_language(
                window, vad_filter=True, vad_parameters=vad,
                language_detection_segments=3)
            votes[language] = votes.get(language, 0.0) + float(probability)
        if not votes:
            return "en", 0.0
        language = max(votes, key=votes.get)
        return language, votes[language] / len(starts)

    def _decode(self, audio: np.ndarray, language: str, prompt: str | None, batch: int):
        opts = self._options
        if batch >= 2:
            if self._batched is None:
                from faster_whisper import BatchedInferencePipeline
                self._batched = BatchedInferencePipeline(self._model)
            # The batched pipeline decodes VAD chunks independently: no previous-text conditioning and no
            # hallucination_silence_threshold (it forces both off).
            segments, _info = self._batched.transcribe(
                audio, language=language, beam_size=opts.beam_size, vad_filter=opts.vad_filter,
                vad_parameters=opts.vad_parameters(), word_timestamps=opts.word_timestamps,
                initial_prompt=prompt, batch_size=batch, without_timestamps=opts.batch_without_timestamps)
            return segments
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=opts.beam_size,
            vad_filter=opts.vad_filter,
            vad_parameters=opts.vad_parameters(),
            word_timestamps=opts.word_timestamps,
            hallucination_silence_threshold=(
                opts.hallucination_silence_threshold if opts.word_timestamps else None),
            condition_on_previous_text=opts.condition_on_previous_text,
            initial_prompt=prompt,
        )
        return segments

    def transcribe(self, audio: np.ndarray, language: str, offset: float,
                   initial_prompt: str | None) -> Iterator[AsrSegment]:
        """Yield segments; on CUDA out-of-memory the batch size is halved (down to the unbatched path) and the
        audio after the last yielded segment is decoded again."""
        done = 0.0                                      # seconds of `audio` already yielded
        while True:
            base = done
            piece = audio[int(base * SAMPLE_RATE):]
            if len(piece) <= SAMPLE_RATE // 2:
                return
            try:
                for seg in self._decode(piece, language, initial_prompt, self._batch):
                    converted = _convert(seg, offset + base)
                    if converted is not None:
                        done = converted.end - offset
                        yield converted
                return
            except RuntimeError as exc:
                if self._batch < 2 or "out of memory" not in str(exc).lower():
                    raise
                self._batch = self._batch // 2 if self._batch >= 4 else 0
                log.warning("CUDA out of memory; batch size is now %s", self._batch or "off")

    def transcribe_span(self, audio: np.ndarray, language: str, offset: float) -> list[AsrSegment]:
        """Slow, careful decode of a short span (beam 10, sampling fallbacks) for re-decoding (D-091)."""
        opts = self._options
        segments, _info = self._model.transcribe(
            audio, language=language, beam_size=10, temperature=(0.0, 0.2, 0.4), vad_filter=opts.vad_filter,
            vad_parameters=opts.vad_parameters(), word_timestamps=True, condition_on_previous_text=False,
            hallucination_silence_threshold=opts.hallucination_silence_threshold)
        return [c for c in (_convert(s, offset) for s in segments) if c is not None]


def _convert(seg, offset: float) -> AsrSegment | None:
    text = seg.text.strip()
    if not text:
        return None
    words = [Word(round(w.start + offset, 3), round(w.end + offset, 3), w.word, round(float(w.probability), 4))
             for w in (seg.words or [])]
    return AsrSegment(
        start=round(seg.start + offset, 3), end=round(seg.end + offset, 3), text=text, words=words,
        avg_logprob=_finite(seg.avg_logprob), no_speech_prob=_finite(seg.no_speech_prob),
        compression_ratio=_finite(seg.compression_ratio))


def _finite(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return round(value, 4) if math.isfinite(value) else None
