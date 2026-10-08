"""Translation units and translation backends.

Seq2seq MT models translate one input at a time and cannot take surrounding dialogue as
context. To avoid translating sentence fragments in isolation, consecutive ASR segments that
form one sentence are merged into a translation unit (bounded by duration and length), and the
unit's time span becomes the subtitle cue. Speaker/gender/glossary steering is not supported by
this backend (see DECISIONS.md and the risk table in the specification).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

log = logging.getLogger(__name__)

_TERMINAL = (".", "!", "?", "\u2026", "\u061f", "\u3002", "\uff01", "\uff1f", "\"", "\u201d")


@dataclass
class TranslationUnit:
    id: int
    segment_ids: list[int]
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_units(
    segments: Sequence[dict],
    max_duration: float = 7.0,
    max_chars: int = 90,
    max_gap: float = 0.6,
) -> list[TranslationUnit]:
    """Group master-transcript segments (dicts with id/start/end/text) into translation units."""
    units: list[TranslationUnit] = []
    current: TranslationUnit | None = None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if current is not None:
            open_sentence = not current.text.rstrip().endswith(_TERMINAL)
            fits = (
                seg["start"] - current.end <= max_gap
                and seg["end"] - current.start <= max_duration
                and len(current.text) + 1 + len(text) <= max_chars
            )
            if open_sentence and fits:
                current.segment_ids.append(seg["id"])
                current.end = seg["end"]
                current.text = f"{current.text} {text}"
                continue
            units.append(current)
        current = TranslationUnit(len(units), [seg["id"]], seg["start"], seg["end"], text)
    if current is not None:
        units.append(current)
    return units


class TranslationBackend(Protocol):
    name: str

    def translate(self, texts: list[str], source_language: str, target_language: str,
                  context: list[tuple[str, str]] | None = None) -> list[str]:
        """context: previous (source, translation) pairs; backends that cannot use it ignore it."""
        ...


class MadladBackend:
    """MADLAD-400 (T5) in CTranslate2 format. The target language is selected with a <2xx> token."""

    def __init__(self, model_dir: Path, device: str, compute_type: str, beam_size: int = 4):
        import ctranslate2
        import sentencepiece

        model_dir = Path(model_dir)
        self.name = f"madlad400:{model_dir.name}:{device}:{compute_type}"
        self._beam_size = beam_size
        self._sp = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / "spiece.model"))
        self._translator = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type)
        log.info("Loaded translation backend %s", self.name)

    def translate(self, texts: list[str], source_language: str, target_language: str,
                  context: list[tuple[str, str]] | None = None) -> list[str]:
        # MADLAD cannot use context. It infers the source language; only the target tag is needed.
        batch = [self._sp.encode(f"<2{target_language}> {t}", out_type=str) + ["</s>"] for t in texts]
        results = self._translator.translate_batch(
            batch,
            beam_size=self._beam_size,
            max_batch_size=8,
            max_decoding_length=256,
        )
        return [self._sp.decode_pieces(r.hypotheses[0]).strip() for r in results]
