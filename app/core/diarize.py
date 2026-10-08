"""Offline speaker diarization on the CPU with sherpa-onnx (D-098).

Models (both from the sherpa-onnx release pages, pinned by SHA-256 and verified before use):
- segmentation: pyannote segmentation 3.0 (MIT), ONNX export by k2-fsa;
- speaker embedding: 3D-Speaker CAM++ zh+en (Apache-2.0).

The audio is processed in windows (the transcription windows, cut at pauses). Each window is clustered on its own;
the speakers of different windows are then linked by the cosine similarity of one embedding per window speaker.
Finished windows are appended to a partial file, so a killed job resumes at the next window.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tarfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from app.core.audio_processor import SAMPLE_RATE
from app.core.errors import JobCancelled, PipelineError
from app.utils.atomic import JsonlWriter, load_jsonl

log = logging.getLogger(__name__)

VERSION = 1                         # part of the diarize cache key
_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"


@dataclass(frozen=True)
class Model:
    url: str
    sha256: str
    folder: str
    file: str
    archive: bool


SEGMENTATION = Model(
    url=_RELEASES + "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    sha256="24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    folder="sherpa-onnx-pyannote-segmentation-3-0", file="model.onnx", archive=True)
EMBEDDING = Model(
    url=_RELEASES + "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    sha256="aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
    folder="campplus-zh-en", file="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx", archive=False)

CLUSTER_THRESHOLD = 0.5             # FastClustering distance threshold inside a window
LINK_THRESHOLD = 0.55               # cosine similarity above which two window speakers are one person
EMBED_SECONDS = 20.0                # speech used for a window speaker's embedding
MIN_EMBED_SECONDS = 1.0
Progress = Callable[[float], None]


def model_paths(models_dir: Path) -> tuple[Path, Path]:
    root = Path(models_dir) / "diarization"
    return root / SEGMENTATION.folder / SEGMENTATION.file, root / EMBEDDING.folder / EMBEDDING.file


def _download(model: Model, target: Path, progress: Callable[[float, str], None] | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.parent / (Path(model.url).name + ".part")
    digest = hashlib.sha256()
    log.info("Downloading the diarization model (%s)", model.url)
    try:
        with urllib.request.urlopen(model.url, timeout=60) as response, open(part, "wb") as out:
            size = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress and size:
                    progress(done / size, Path(model.url).name)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise PipelineError(f"Diarization model download failed: {exc}") from exc
    if digest.hexdigest() != model.sha256:       # checked before anything is extracted or used
        part.unlink(missing_ok=True)
        raise PipelineError("Diarization model checksum mismatch")
    if not model.archive:
        os.replace(part, target)
        return
    with tarfile.open(part, "r:bz2") as tar:
        for member in tar.getmembers():
            if member.isfile() and Path(member.name).name == model.file:    # only the model file, flat
                with tar.extractfile(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    part.unlink(missing_ok=True)


def ensure_models(models_dir: Path, progress: Callable[[float, str], None] | None = None) -> tuple[Path, Path]:
    paths = model_paths(models_dir)
    for model, path in zip((SEGMENTATION, EMBEDDING), paths):
        if not path.is_file():
            _download(model, path, progress)
    return paths


def _normalize(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    return array / max(float(np.linalg.norm(array)), 1e-12)


def link_speakers(windows: list[dict], threshold: float = LINK_THRESHOLD) -> list[tuple[float, float, int]]:
    """Merge the window-local speakers into global ids (0, 1, ...) by embedding similarity.

    `windows` are the partial records: {"segments": [[start, end, local], ...], "embeddings": {local: [..]|None}}.
    A local speaker without an embedding (too little speech) becomes its own global speaker."""
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    result = []
    for window in windows:
        mapping: dict[int, int] = {}
        taken: set[int] = set()
        for local, vector in sorted(window["embeddings"].items(), key=lambda kv: int(kv[0])):
            local = int(local)
            if vector is None:
                centroids.append(np.zeros(1))
                counts.append(0)
                mapping[local] = len(centroids) - 1
                continue
            emb = _normalize(vector)
            scores = [(float(emb @ c) if c.shape == emb.shape else -1.0, i) for i, c in enumerate(centroids)
                      if i not in taken]          # two speakers of one window are never the same person
            best = max(scores, default=(-1.0, -1))
            if best[0] >= threshold:
                index = best[1]
                centroids[index] = _normalize(centroids[index] * counts[index] + emb)
                counts[index] += 1
            else:
                centroids.append(emb)
                counts.append(1)
                index = len(centroids) - 1
            taken.add(index)
            mapping[local] = index
        for start, end, local in window["segments"]:
            result.append((start, end, mapping.get(int(local), -1)))
    return [r for r in result if r[2] >= 0]


def _build(models_dir: Path, threads: int, num_speakers: int | None):
    import sherpa_onnx

    seg_path, emb_path = model_paths(models_dir)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_path)),
            num_threads=threads, provider="cpu"),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_path), num_threads=threads,
                                                              provider="cpu"),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers or -1, threshold=CLUSTER_THRESHOLD),
        min_duration_on=0.3, min_duration_off=0.5)
    if not config.validate():
        raise PipelineError("Invalid diarization model configuration")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config.embedding)
    return sherpa_onnx.OfflineSpeakerDiarization(config), extractor


def _embed(extractor, chunk: np.ndarray, segments: list[tuple[float, float]]) -> list[float] | None:
    """One embedding from up to EMBED_SECONDS of a speaker's speech inside the window audio."""
    parts, total = [], 0.0
    for start, end in sorted(segments, key=lambda s: s[0] - s[1]):          # longest turns first
        parts.append(chunk[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)])
        total += end - start
        if total >= EMBED_SECONDS:
            break
    if total < MIN_EMBED_SECONDS:
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=np.ascontiguousarray(np.concatenate(parts)))
    stream.input_finished()
    return [float(x) for x in extractor.compute(stream)]


def diarize(audio: np.ndarray, models_dir: Path, windows: Sequence[tuple[float, float]], partial: Path,
            num_speakers: int | None = None, progress: Progress | None = None,
            cancel: threading.Event | None = None, threads: int | None = None) -> list[tuple[float, float, int]]:
    """[(start_s, end_s, speaker_id)] with speaker ids 0, 1, ... over the whole audio (absolute times)."""
    threads = threads or max(1, min(4, (os.cpu_count() or 2) // 2))
    done = load_jsonl(partial)
    engine = None
    with JsonlWriter(partial) as writer:
        for index, (w_start, w_end) in enumerate(windows):
            if cancel is not None and cancel.is_set():
                raise JobCancelled()
            if index < len(done):
                continue
            if engine is None:
                engine = _build(models_dir, threads, num_speakers)
            sd, extractor = engine
            chunk = audio[int(w_start * SAMPLE_RATE):int(w_end * SAMPLE_RATE)]
            if len(chunk) < SAMPLE_RATE:
                record = {"window": index, "segments": [], "embeddings": {}}
            else:
                found = sd.process(np.ascontiguousarray(chunk)).sort_by_start_time()
                segments = [(round(w_start + s.start, 3), round(w_start + s.end, 3), int(s.speaker)) for s in found]
                by_speaker: dict[int, list[tuple[float, float]]] = {}
                for s in found:
                    by_speaker.setdefault(int(s.speaker), []).append((s.start, s.end))
                record = {"window": index, "segments": [list(s) for s in segments],
                          "embeddings": {str(k): _embed(extractor, chunk, v) for k, v in by_speaker.items()}}
            writer.write(record)
            done.append(record)
            if progress:
                progress((index + 1) / len(windows))
    return link_speakers(done)


def assign(segments: list[dict], turns: Sequence[tuple[float, float, int]]) -> dict[int, int]:
    """Segment index -> global speaker id: the speaker with the largest overlap (segments without one are left out)."""
    result = {}
    for index, seg in enumerate(segments):
        overlap: dict[int, float] = {}
        for start, end, speaker in turns:
            if end <= seg["start"]:
                continue
            if start >= seg["end"]:
                break
            overlap[speaker] = overlap.get(speaker, 0.0) + min(end, seg["end"]) - max(start, seg["start"])
        if overlap:
            result[index] = max(overlap, key=overlap.get)
    return result


def apply_labels(doc: dict, turns: Sequence[tuple[float, float, int]]) -> dict[str, dict]:
    """Set `speaker` on the master segments ("S1", "S2", ... in order of first appearance) and describe them in
    doc["speakers"]. Returns that description."""
    assigned = assign(doc["segments"], turns)
    names: dict[int, str] = {}
    for index in sorted(assigned):
        names.setdefault(assigned[index], f"S{len(names) + 1}")
    speakers: dict[str, dict] = {name: {"seconds": 0.0, "segments": 0} for name in names.values()}
    for index, seg in enumerate(doc["segments"]):
        speaker = names.get(assigned.get(index, -1))
        seg["speaker"] = speaker
        if speaker:
            speakers[speaker]["segments"] += 1
            speakers[speaker]["seconds"] = round(speakers[speaker]["seconds"] + seg["end"] - seg["start"], 3)
    for seg in doc["segments"]:
        label_words(seg, turns, names, speakers)
    doc["speakers"] = speakers
    return speakers


def label_words(seg: dict, turns: Sequence[tuple[float, float, int]], names: dict[int, str],
                speakers: dict[str, dict]) -> None:
    """When two speakers share one segment, every word gets its own `speaker` (used to split a cue into two dash
    lines, D-102). A word inside overlapping speech belongs to the shortest turn that covers it."""
    words = seg.get("words") or []
    local = [t for t in turns if t[1] > seg["start"] and t[0] < seg["end"]]
    if len(words) < 2 or len({t[2] for t in local}) < 2:
        return
    labels: list[str | None] = []
    for word in words:
        middle = (word["start"] + word["end"]) / 2
        covering = [t for t in local if t[0] <= middle <= t[1]]
        if covering:
            speaker = min(covering, key=lambda t: t[1] - t[0])[2]
            if speaker not in names:
                names[speaker] = f"S{len(names) + 1}"
                speakers[names[speaker]] = {"seconds": 0.0, "segments": 0}
            labels.append(names[speaker])
        else:
            labels.append(labels[-1] if labels else seg.get("speaker"))
    if len({l for l in labels if l}) >= 2:
        for word, label in zip(words, labels):
            word["speaker"] = label or seg.get("speaker")
