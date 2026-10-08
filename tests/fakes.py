"""Test doubles for engines. Used only by tests, never by production code."""

from __future__ import annotations

import threading
import time

from app.core.audio_processor import SAMPLE_RATE
from app.core.transcription import AsrSegment, Word
from app.services.hardware_detection import EnginePlan

SEGMENT_SECONDS = 2.0


class FakeAsrEngine:
    """Emits one segment per SEGMENT_SECONDS of audio, aligned to absolute time."""

    def __init__(self, name="fake-asr", delay=0.0, fail_after=None, cancel_after=None,
                 cancel_event: threading.Event | None = None, language=("tr", 0.97), words=True):
        self.name = name
        self.delay = delay
        self.fail_after = fail_after
        self.cancel_after = cancel_after
        self.cancel_event = cancel_event
        self.language = language
        self.words = words
        self.calls: list[dict] = []
        self.emitted = 0

    def detect_language(self, audio):
        self.calls.append({"detect": True})
        return self.language

    def transcribe(self, audio, language, offset, initial_prompt):
        self.calls.append({"offset": offset, "prompt": initial_prompt, "samples": len(audio)})
        duration = len(audio) / SAMPLE_RATE
        index = round(offset / SEGMENT_SECONDS)
        start = index * SEGMENT_SECONDS
        while start + 0.5 <= offset + duration:
            if self.fail_after is not None and self.emitted >= self.fail_after:
                raise RuntimeError("CUDA failed with error out of memory")
            if self.delay:
                time.sleep(self.delay)
            end = min(start + SEGMENT_SECONDS - 0.2, offset + duration)
            text = f"Cumle {index} burada." if index % 3 != 1 else f"Yarim cumle {index}"
            words = [Word(start, end, text, 0.9)] if self.words else []
            yield AsrSegment(start=round(start, 3), end=round(end, 3), text=text, words=words,
                             avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.2)
            self.emitted += 1
            if self.cancel_after is not None and self.emitted >= self.cancel_after and self.cancel_event:
                self.cancel_event.set()
            index += 1
            start = index * SEGMENT_SECONDS


class FakeTranslator:
    def __init__(self, name="fake-mt", fail_after_batches=None):
        self.name = name
        self.fail_after_batches = fail_after_batches
        self.batches = 0
        self.inputs: list[str] = []
        self.contexts: list[list] = []

    def translate(self, texts, source_language, target_language, context=None):
        self.contexts.append(list(context or []))
        if self.fail_after_batches is not None and self.batches >= self.fail_after_batches:
            raise RuntimeError("CUDA out of memory")
        self.batches += 1
        self.inputs.extend(texts)
        return [f"[{target_language}] {t}" for t in texts]


class Factory:
    """Records loads and returns engines from a per-plan mapping."""

    def __init__(self, engines: dict):
        self.engines = engines
        self.loads: list[EnginePlan] = []

    def __call__(self, plan, *args):
        self.loads.append(plan)
        engine = self.engines[(plan.model, plan.device)]
        if isinstance(engine, Exception):
            raise engine
        return engine


CPU_ASR = EnginePlan("small", "cpu", "int8", "test")
GPU_ASR = EnginePlan("large-v3", "cuda", "int8_float16", "test")
CPU_MT = EnginePlan("madlad", "cpu", "int8", "test")
GPU_MT = EnginePlan("madlad", "cuda", "int8", "test")


class FakeDownloader:
    """Stands in for YtDlpAdapter: serves a local media file and canned subtitle tracks."""

    def __init__(self, media_path, info=None, subtitle_texts=None, fail_audio=None, fail_subtitles=None,
                 playlist_entries=None):
        self.media_path = media_path
        self.info = info or {"id": "abc", "title": "Show S01E02 - Pilot", "duration": 20,
                             "subtitles": {}, "automatic_captions": {}}
        self.subtitle_texts = subtitle_texts or {}
        self.fail_audio = fail_audio
        self.fail_subtitles = fail_subtitles or {}
        self.audio_downloads = 0
        self.subtitle_requests = []
        self.playlist_entries = playlist_entries or []

    def probe(self, url):
        return dict(self.info)

    def extract_playlist(self, url):
        return list(self.playlist_entries)

    def download_video(self, url, dest_base, cancel=None, progress=None, quality="best", warn=None, work_dir=None):
        import shutil

        if self.fail_audio:
            raise self.fail_audio
        self.audio_downloads += 1
        dest_base.parent.mkdir(parents=True, exist_ok=True)
        target = dest_base.with_name(dest_base.name + self.media_path.suffix)
        shutil.copyfile(self.media_path, target)
        if progress:
            progress(1.0)
        return target

    def download_subtitle(self, url, language, automatic, dest_dir):
        self.subtitle_requests.append((language, automatic))
        if language in self.fail_subtitles:
            raise self.fail_subtitles[language]
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"yt.{'auto' if automatic else 'manual'}.{language}.vtt"
        path.write_text(self.subtitle_texts[(language, automatic)], encoding="utf-8")
        return path


def vtt_from(cues):
    """Build a WebVTT document from (start, end, text) tuples."""
    def ts(sec):
        ms = round(sec * 1000)
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"
    blocks = ["WEBVTT", ""]
    for start, end, text in cues:
        blocks += [f"{ts(start)} --> {ts(end)}", text, ""]
    return "\n".join(blocks)


def matching_cues(count=10):
    """Cues identical to FakeAsrEngine output, as a correct same-language subtitle would be."""
    cues = []
    for index in range(count):
        start = index * SEGMENT_SECONDS
        text = f"Cumle {index} burada." if index % 3 != 1 else f"Yarim cumle {index}"
        cues.append((start, start + SEGMENT_SECONDS - 0.2, text))
    return cues
