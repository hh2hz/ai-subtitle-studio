"""Job pipeline: [download] -> audio -> independent ASR -> master transcript -> subtitle evidence
-> translation -> SRT export.

Every stage is checkpointed under <jobs_dir>/<job_key>/:
- stages/<stage>.json marks a completed stage with its cache key; a stage is skipped when the key
  matches and its outputs exist.
- Transcription and translation also append each finished segment/unit to a JSON Lines file, so a
  killed or cancelled job resumes from the last finished segment instead of from the start.
Cache keys chain: each key includes the key of the stage it depends on.
"""

from __future__ import annotations

import copy
import gc
import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from app.core import brief, confidence, diarize, enhance, frames, finalize, hallucination, platform_text, redecode, style
from app.core import master_transcript as mt
from app.core import metadata
from app.core import names
from app.core.names import SeriesGlossary, transliteration_ok
from app.core.names import supported_target as names_supported
from app.core.audio_processor import SAMPLE_RATE, extract_audio, load_wav
from app.core.downloader import CHAIN_VERSION
from app.core.errors import JobCancelled, PipelineError
from app.core.exporter import mark_rtl, output_paths, safe_basename, write_srt
from app.core.verification import check_line
from app.core.windows import plan_windows
from app.utils.languages import RTL_LANGUAGES
from app.utils.paths import default_output_dir
from app.core.metadata import SeriesInfo
from app.core.modes import Mode
from app.core.subtitle_formats import audio_agreement
from app.core.transcription import AsrEngine, AsrOptions, AsrSegment
from app.core.segmentation import SEGMENTATION_VERSION, segment_cues
from app.core.llm_translation import BlockResult
from app.core.translation import TranslationBackend
from app.providers.base import AUTO, SubtitleEvidence, SubtitleProvider, SubtitleRequest, collect_evidence
from app.services.hardware_detection import EnginePlan
from app.services.resources import AI, CPU, GPU, NET, ResourceSet
from app.utils.atomic import JsonlWriter, atomic_write_json, load_jsonl, read_json
from app.utils.hashing import file_fingerprint, stable_hash
from app.utils.languages import AUTO_DETECT
from app.utils.logging_setup import LOG_FORMAT

log = logging.getLogger(__name__)

STAGES = ("download", "audio", "transcribe", "diarize", "subtitles", "translate", "refine", "export")
_WEIGHTS = {"download": 0.07, "audio": 0.03, "transcribe": 0.37, "diarize": 0.05, "subtitles": 0.03,
            "translate": 0.17, "refine": 0.23, "export": 0.05}
# Bumped when prompts or name handling change so cached AI results are redone
# (2: names check, D-039; 4: style lock, D-051; 5: episode brief, D-055; 6: style guides, D-060; 7: more forward context, D-065; 8: reading-speed budget, D-066; 9: speakers, D-099; 10: condense after timing, D-100; 11: risky-line double check, D-104).
REFINE_VERSION = 11
REFINE_BLOCK = 25           # lines per LLM request
REFINE_CONTEXT_BEFORE = 6
REFINE_CONTEXT_AFTER = 8
# Same-language evidence whose text agrees with our ASR below this is rejected (wrong language, other cut...).
EVIDENCE_REJECT_BELOW = 0.3
EVIDENCE_MATCH_FROM = 0.6
_FALLBACK_DEVICES = ("cuda", "local")   # plans whose runtime failure moves on to the next plan
_TRANSLATION_BATCH = 8
PIPELINE_VERSION = 1
ASR_FILTER_VERSION = 4           # hallucination filter, silence threshold, re-decoding; part of the transcription cache key

MODE_PARAMS = {
    Mode.FAST: {"asr_beam": 1, "mt_beam": 2},
    Mode.BALANCED: {"asr_beam": 5, "mt_beam": 4},
    Mode.MAXIMUM_ACCURACY: {"asr_beam": 5, "mt_beam": 5},
}

DownloadProgress = Callable[[float, str], None]
AsrFactory = Callable[[EnginePlan, AsrOptions, DownloadProgress], AsrEngine]
MtFactory = Callable[[EnginePlan, int, DownloadProgress], TranslationBackend]
ProgressFn = Callable[[str, float, float, str], None]        # stage, stage fraction, overall, message
StageRecorder = Callable[[str, str, str | None, str | None], None]  # stage, status, cache key, error


@dataclass(frozen=True)
class JobConfig:
    input_path: Path | None          # local file, or None for URL input
    source_language: str
    target_language: str
    mode: Mode
    output_dir: Path | None = None   # output root; None: default_output_dir(). Each job gets <root>/<title>/
    source_url: str | None = None
    series_override: SeriesInfo | None = None   # user corrections; set fields win over detection


@dataclass
class JobResult:
    job_key: str
    job_dir: Path
    output_dir: Path
    outputs: dict[str, str]
    stats: dict
    warnings: list[str] = field(default_factory=list)
    series: dict = field(default_factory=dict)


_RUNNING_KEYS: set[str] = set()
_RUNNING_KEYS_COND = threading.Condition()


@contextmanager
def _exclusive_job(job_key: str, cancel: threading.Event):
    """Jobs of the same input share one job folder; when the same video is queued twice, the second waits for the
    first (and then finds its cached stages) instead of writing the same files at the same time (D-120)."""
    with _RUNNING_KEYS_COND:
        while job_key in _RUNNING_KEYS:
            if cancel.is_set():
                raise JobCancelled()
            _RUNNING_KEYS_COND.wait(0.2)
        _RUNNING_KEYS.add(job_key)
    try:
        yield
    finally:
        with _RUNNING_KEYS_COND:
            _RUNNING_KEYS.discard(job_key)
            _RUNNING_KEYS_COND.notify_all()


class _ThreadFilter(logging.Filter):
    """Pass only log records written by the registered threads (one job's threads)."""

    def __init__(self):
        super().__init__()
        self._threads: set[int] = set()

    def add_current(self) -> None:
        self._threads.add(threading.get_ident())

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread in self._threads


class _Diarization:
    """State of one background speaker-detection run, shared between its thread and the job thread."""

    def __init__(self, key: str, cached: bool = False):
        self.key = key
        self.cached = cached
        self.thread: threading.Thread | None = None
        self.cancel = threading.Event()
        self.turns: list | None = None
        self.error: BaseException | None = None
        self.fraction = 0.0
        self.started: float | None = None

    def set_fraction(self, fraction: float) -> None:
        self.fraction = float(fraction)


class Pipeline:
    def __init__(
        self,
        config: JobConfig,
        jobs_dir: Path,
        asr_plans: list[EnginePlan],
        mt_plans: list[EnginePlan],
        asr_factory: AsrFactory,
        mt_factory: MtFactory,
        progress: ProgressFn | None = None,
        cancel: threading.Event | None = None,
        recorder: StageRecorder | None = None,
        downloader=None,
        providers: list[SubtitleProvider] | None = None,
        refiner_factory=None,
        asr_settings: dict | None = None,
        enhancer=None,
        diarizer=None,
        name_normalization: bool = True,
        video_quality: str = "best",
        snap_shots: bool = False,
        keep_cache: bool = True,
        resources: ResourceSet | None = None,
        priority=0,
    ):
        if not asr_plans or not mt_plans:
            raise ValueError("At least one ASR plan and one translation plan are required")
        if (config.input_path is None) == (config.source_url is None):
            raise ValueError("Exactly one of input_path and source_url must be set")
        self.config = config
        self._downloader = downloader
        self._refiner_factory = refiner_factory   # (source, target, media) -> LlmRefiner | None
        self._providers = providers or []
        # {"enhance": "auto" | "on" | "off"} (D-044)
        self._asr_settings = dict(asr_settings or {})
        self._enhancer = enhancer              # (audio, progress, cancel) -> voice-only audio
        self._diarizer = diarizer              # (audio, windows, partial, progress, cancel) -> [(start, end, id)]
        self._video_quality = str(video_quality or "best")   # "best" or a height, for link downloads (D-080)
        self._media_path: Path | None = config.input_path
        self._out_dir: Path | None = None
        self._media_info: dict | None = None
        self.series = SeriesInfo()
        self.jobs_dir = Path(jobs_dir)
        self._asr_plans = asr_plans
        self._mt_plans = mt_plans
        self._asr_factory = asr_factory
        self._mt_factory = mt_factory
        self._progress_fn = progress
        self.cancel = cancel or threading.Event()
        self._recorder = recorder
        self._params = MODE_PARAMS[config.mode]
        self._asr: AsrEngine | None = None
        self._asr_index = 0
        self._mt: TranslationBackend | None = None
        self._mt_index = 0
        self._done_weight = 0.0
        self.warnings: list[str] = []
        self.stats: dict = {"stages": {}}
        self.job_key: str | None = None
        self.job_dir: Path | None = None
        self._name_normalization = bool(name_normalization)
        self._snap_shots = bool(snap_shots)         # snap cue edges to shot changes (D-103), off by default
        self._keep_cache = bool(keep_cache)         # False: the job folder is removed after a successful export
        self._finished = False
        # Machine resources shared with the other jobs of the queue (D-120). A pipeline on its own gets private gates,
        # so it never waits; `priority` orders waiting jobs (the queue position: lower goes first).
        self._resources = resources if resources is not None else ResourceSet()
        self._priority = priority
        self._log_threads: _ThreadFilter | None = None
        self._waited = 0.0                  # seconds this job waited for resources other jobs held

    # -- helpers ---------------------------------------------------------------------------

    def _progress(self, stage: str, fraction: float, message: str = "") -> None:
        if self._progress_fn:
            fraction = min(max(fraction, 0.0), 1.0)
            self._progress_fn(stage, fraction, self._done_weight + _WEIGHTS[stage] * fraction, message)

    def _notify(self, message: str) -> None:
        """Side channel for the UI (media length for ETA maths): no stage progress, no output change."""
        if self._progress_fn:
            self._progress_fn("", 0.0, self._done_weight, message)

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise JobCancelled()

    def _sleep_unless_cancelled(self, seconds: float) -> None:
        """A wait that ends at once when the job is cancelled (rate-limit waits of the cloud AI)."""
        if self.cancel.wait(max(0.0, seconds)):
            raise JobCancelled()

    def _record(self, stage: str, status: str, key: str | None = None, error: str | None = None) -> None:
        if self._recorder:
            try:
                self._recorder(stage, status, key, error)
            except Exception:  # Bookkeeping must never break processing.
                log.exception("Stage recorder failed")

    def _warn(self, message: str) -> None:
        log.warning(message)
        self.warnings.append(message)

    @contextmanager
    def _using(self, resource: str, stage: str):
        """Hold one slot of a machine resource for the work inside the block. While another job holds it, the UI
        shows "waiting for ..." (a progress message without stage progress); a cancel ends the wait at once."""
        def waiting() -> None:
            log.info("Waiting for the %s (stage %s)", resource, stage)
            self._notify(f"waiting:{resource}:{stage}")

        asked = time.monotonic()
        with self._resources.gate(resource).hold(self._priority, self.cancel, waiting):
            self._waited += time.monotonic() - asked
            yield

    def _manifest_path(self, stage: str) -> Path:
        return self.job_dir / "stages" / f"{stage}.json"

    def _cached(self, stage: str, key: str) -> dict | None:
        path = self._manifest_path(stage)
        if not path.is_file():
            return None
        try:
            manifest = read_json(path)
        except ValueError:
            return None
        if manifest.get("key") != key or not manifest.get("completed"):
            return None
        if not all((self.job_dir / name).is_file() for name in manifest.get("outputs", [])):
            return None
        return manifest

    def _complete(self, stage: str, key: str, outputs: list[str], info: dict) -> None:
        atomic_write_json(self._manifest_path(stage), {
            "key": key, "completed": True, "outputs": outputs, "info": info,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    def _skip_stage(self, stage: str, reason: str) -> None:
        self.stats["stages"][stage] = {"skipped": reason}
        self._record(stage, "skipped")
        self._done_weight += _WEIGHTS[stage]

    def _finish_stage(self, stage: str, started: float, duration: float | None, cached: bool, **extra) -> None:
        elapsed = time.monotonic() - started
        entry = {"cached": cached, "elapsed_s": round(elapsed, 2), **extra}
        if duration and not cached:
            entry["real_time_factor"] = round(elapsed / duration, 4)
        self.stats["stages"][stage] = entry
        self._done_weight += _WEIGHTS[stage]
        if self._progress_fn:
            self._progress_fn(stage, 1.0, self._done_weight, "")
        log.info("Stage %s finished: %s", stage, entry)

    # -- engines ---------------------------------------------------------------------------

    def _download_cb(self, stage: str) -> DownloadProgress:
        return lambda fraction, text: self._progress(stage, fraction, f"download:{text}")

    def _ensure_asr(self) -> AsrEngine:
        if self._asr is not None:
            return self._asr
        options = self._asr_options()
        while self._asr_index < len(self._asr_plans):
            plan = self._asr_plans[self._asr_index]
            self._check_cancel()
            try:
                log.info("Loading ASR plan %s", plan)
                self._asr = self._asr_factory(plan, options, self._download_cb("transcribe"))
                return self._asr
            except JobCancelled:
                raise
            except Exception as exc:
                self._warn(f"ASR plan {plan.model}/{plan.device}/{plan.compute_type} unavailable: {exc}")
                self._asr_index += 1
        raise PipelineError("No speech recognition engine could be loaded", "error.asr_unavailable")

    def _release_asr(self) -> None:
        self._asr = None
        gc.collect()

    def _asr_options(self) -> AsrOptions:
        return AsrOptions(beam_size=self._params["asr_beam"])

    def _ensure_mt(self) -> TranslationBackend:
        if self._mt is not None:
            return self._mt
        while self._mt_index < len(self._mt_plans):
            plan = self._mt_plans[self._mt_index]
            self._check_cancel()
            try:
                log.info("Loading translation plan %s", plan)
                self._mt = self._mt_factory(plan, self._params["mt_beam"], self._download_cb("translate"))
                return self._mt
            except JobCancelled:
                raise
            except Exception as exc:
                self._warn(f"Translation plan {plan.model}/{plan.device}/{plan.compute_type} unavailable: {exc}")
                self._mt_index += 1
        raise PipelineError("No translation engine could be loaded", "error.mt_unavailable")

    def _release_mt(self) -> None:
        if self._mt is not None and hasattr(self._mt, "close"):
            try:
                self._mt.close()
            except Exception:
                log.debug("Translation backend close failed", exc_info=True)
        self._mt = None
        gc.collect()

    # -- stages ----------------------------------------------------------------------------

    def _download_key(self, identity: dict) -> str:
        """Cache key of the download stage. The requested quality and the format chain version are part of it, so
        a cached 360p file can never satisfy a later 1080p request, and a file fetched with an older chain is
        downloaded again (D-080, D-083)."""
        return stable_hash({"v": PIPELINE_VERSION, "quality": self._video_quality,
                            "chain": CHAIN_VERSION, **identity})

    def _stage_download(self, key: str) -> None:
        started = time.monotonic()
        cached = self._cached("download", key)
        if cached and Path(cached["info"]["media"]).is_file():
            self._media_info = read_json(self.job_dir / "media_info.json")
            self._media_path = Path(cached["info"]["media"])
            self._out_dir = Path(cached["info"]["out_dir"])
            self._record("download", "completed", key)
            self._finish_stage("download", started, None, cached=True)
            return
        if self._downloader is None:
            raise PipelineError("URL input requires a downloader", "error.download_failed")
        self._record("download", "running", key)
        with self._using(NET, "download"):
            self._download_now(key, time.monotonic())     # the stage's time starts when it got its slot

    def _download_now(self, key: str, started: float) -> None:
        self._progress("download", 0.0, "probing")
        info = self._downloader.probe(self.config.source_url)
        note = getattr(self._downloader, "access_note", None)
        if note:
            self._warn(note)
        keep = ("id", "title", "channel", "uploader", "duration", "description", "upload_date", "language",
                "webpage_url", "extractor_key", "series", "season_number", "episode_number", "episode",
                "release_year")
        self._media_info = {k: info.get(k) for k in keep}
        self._media_info["subtitles"] = {k: [f.get("ext") for f in v] for k, v in (info.get("subtitles") or {}).items()}
        self._media_info["automatic_captions"] = {
            k: [f.get("ext") for f in v] for k, v in (info.get("automatic_captions") or {}).items()}
        atomic_write_json(self.job_dir / "media_info.json", self._media_info)
        if self._media_info.get("duration"):
            self._notify(f"media_duration:{self._media_info['duration']}")   # ETA only; no stage progress
        self._check_cancel()
        base = safe_basename(self._media_info.get("title") or "Video", max_length=80)
        self._out_dir = self._output_root() / base
        # Another video with the same title may already be in this folder: never reuse its file (D-047).
        owner_file = self._out_dir / f"{base}.source.json"
        video_id = str(self._media_info.get("id") or self.config.source_url)
        try:
            owner = read_json(owner_file).get("id")
        except (OSError, ValueError):
            owner = None
        if owner not in (None, video_id) or (owner is None and any(self._out_dir.glob(f"{base}.mp4"))):
            self._out_dir = self._output_root() / safe_basename(f"{base} [{video_id}]", max_length=100)
            owner_file = self._out_dir / f"{base}.source.json"
        self._out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(owner_file, {"id": video_id, "url": self.config.source_url})
        path = self._downloader.download_video(self.config.source_url, self._out_dir / base, self.cancel,
                                               lambda f: self._progress("download", f),
                                               quality=self._video_quality, warn=self._warn,
                                               work_dir=self.job_dir / "download")
        self._media_path = path
        self._complete("download", key, ["media_info.json"], {"media": str(path), "out_dir": str(self._out_dir)})
        self._record("download", "completed", key)
        self._finish_stage("download", started, self._media_info.get("duration"), cached=False,
                           title=self._media_info.get("title"))

    def _output_root(self) -> Path:
        return Path(self.config.output_dir) if self.config.output_dir else default_output_dir()

    def _detect_series(self) -> None:
        info = self._media_info or {}
        filename = self.config.input_path.stem if self.config.input_path else None
        detected = metadata.detect(title=info.get("title"), filename=filename, ytdlp_info=info or None)
        if self.config.series_override:
            detected = detected.merged_with(self.config.series_override)
        self.series = detected
        self.stats["series"] = detected.to_dict()
        log.info("Series info: %s", detected.to_dict())

    def _diarize_wanted(self) -> bool:
        return self._diarizer is not None and bool(self._asr_settings.get("diarization", True))

    def _start_diarize(self, key: str, audio) -> "_Diarization | None":
        """Start "who speaks when" on the CPU in the background (D-098, D-120).

        It needs only the audio, so it runs while the GPU transcribes the same episode. The thread waits for a CPU
        slot itself; everything that touches the job database stays on the job's own thread (`_finish_diarize`).
        Returns None when the stage is skipped or its result is cached."""
        if not self._diarize_wanted():
            return None
        if self._cached("diarize", key):
            return _Diarization(key, cached=True)
        self._record("diarize", "running", key)
        job = _Diarization(key)
        options = self._asr_options()

        def work() -> None:
            if self._log_threads is not None:
                self._log_threads.add_current()
            try:
                with self._resources.gate(CPU).hold(self._priority, job.cancel):
                    job.started = time.monotonic()
                    windows = plan_windows(audio, options.vad_threshold)
                    job.turns = self._diarizer(audio, windows, self.job_dir / f"diarize.{key}.partial.jsonl",
                                               job.set_fraction, job.cancel)
            except BaseException as exc:          # noqa: BLE001 - handed to the job thread in _finish_diarize
                job.error = exc

        job.thread = threading.Thread(target=work, name=f"diarize-{self.job_key}", daemon=True)
        job.thread.start()
        return job

    def _stop_diarize(self, job: "_Diarization | None") -> None:
        """Stop a background diarization whose result is no longer wanted (the job failed or was cancelled)."""
        if job is not None and job.thread is not None:
            job.cancel.set()
            job.thread.join(30.0)

    def _finish_diarize(self, job: "_Diarization | None", doc: dict) -> None:
        """Wait for the background diarization, then put its speaker labels on the master segments. Any failure only
        costs the speaker labels (D-098)."""
        started = time.monotonic()
        if job is None:
            self._skip_stage("diarize", "disabled" if self._diarizer else "not available")
            return
        key = job.key
        out_name = f"diarize.{key}.json"
        if job.cached:
            turns = [tuple(t) for t in read_json(self.job_dir / out_name)["turns"]]
        else:
            try:
                told = False
                while job.thread.is_alive():
                    if self.cancel.wait(0.5):
                        job.cancel.set()
                        job.thread.join(30.0)
                        raise JobCancelled()
                    if job.started is None:          # still waiting for another job's speaker detection
                        if not told:
                            self._notify(f"waiting:{CPU}:diarize")
                            told = True
                    else:
                        self._progress("diarize", job.fraction)
                if job.error is not None:
                    raise job.error
                turns = job.turns or []
                atomic_write_json(self.job_dir / out_name, {"turns": [list(t) for t in turns]})
                self._complete("diarize", key, [out_name], {"turns": len(turns)})
            except JobCancelled:
                raise
            except Exception as exc:
                self._warn(f"Speaker detection failed ({type(exc).__name__}: {exc}); continuing without speaker labels")
                self._record("diarize", "failed", key, str(exc))
                self._finish_stage("diarize", started, None, cached=False, error=str(exc))
                return
            if job.started is not None:
                started = job.started             # report the stage's own run time, not only the wait for it
        names = diarize.apply_labels(doc, turns)
        self._record("diarize", "completed", key)
        self._finish_stage("diarize", started, doc["media"]["duration"], cached=job.cached, speakers=len(names))

    def _stage_subtitles(self, key: str, doc: dict) -> list[dict]:
        """Collect subtitle evidence from all providers and score same-language tracks against the ASR."""
        started = time.monotonic()
        out_name = f"subtitles.{key}.json"
        if self._cached("subtitles", key):
            self._record("subtitles", "completed", key)
            self._finish_stage("subtitles", started, None, cached=True)
            return read_json(self.job_dir / out_name)
        if not self._providers:
            self._skip_stage("subtitles", "no providers")
            return []
        self._record("subtitles", "running", key)
        source_language = doc["language"]["code"]
        request = SubtitleRequest(
            source_language=source_language, target_language=self.config.target_language, series=self.series,
            title=(self._media_info or {}).get("title") or (self.config.input_path.stem if self.config.input_path else None),
            media_info=self._media_info, url=self.config.source_url, work_dir=str(self.job_dir / "subtitles"),
        )
        evidence, failures = collect_evidence(self._providers, request)
        for message in failures:
            self._warn(message)
        records = []
        for item in evidence:
            record = item.to_dict()
            if item.language == source_language and item.cues:
                agreement = audio_agreement(doc["segments"], item.cues)
                record["audio_agreement"] = agreement
                if agreement is not None and agreement < EVIDENCE_REJECT_BELOW:
                    record["status"] = "rejected"
                    record["notes"].append("text does not match the audio transcription")
                    self._warn(f"Ignored {item.provider} {item.kind} subtitles ({item.reference}): "
                               f"agreement with the audio only {agreement:.2f}")
                else:
                    record["status"] = "matches_audio" if agreement and agreement >= EVIDENCE_MATCH_FROM else "partial"
            elif item.kind == AUTO:
                # The platform's own speech recognition ran in the wrong language (seen on real YouTube videos).
                record["audio_agreement"] = audio_agreement(doc["segments"], item.cues) if item.cues else None
                record["status"] = "rejected"
                record["notes"].append(f"platform transcribed the speech as '{item.language}', "
                                       f"but the audio is '{source_language}'")
                self._warn(f"Ignored {item.provider} auto captions: transcribed as {item.language}, "
                           f"speech is {source_language}")
            else:
                record["audio_agreement"] = None
                record["status"] = "reference" if item.machine_generated else "unverified"
            records.append(record)
        atomic_write_json(self.job_dir / out_name, records)
        summary = [{k: r[k] for k in ("provider", "language", "kind", "reference", "status", "audio_agreement")}
                   | {"cues": len(r["cues"])} for r in records]
        self.stats["subtitle_sources"] = summary
        self._complete("subtitles", key, [out_name], {"tracks": len(records)})
        self._record("subtitles", "completed", key)
        self._finish_stage("subtitles", started, None, cached=False, tracks=len(records))
        return records

    def _stage_audio(self, key: str) -> tuple[Path, float]:
        started = time.monotonic()
        wav = self.job_dir / "audio.wav"
        cached = self._cached("audio", key)
        if cached:
            self._record("audio", "completed", key)
            self._finish_stage("audio", started, None, cached=True)
            return wav, cached["info"]["duration"]
        self._record("audio", "running", key)
        duration = extract_audio(self._media_path, wav, self.cancel,
                                 lambda f: self._progress("audio", f))
        self._complete("audio", key, ["audio.wav"], {"duration": duration})
        self._record("audio", "completed", key)
        self._finish_stage("audio", started, duration, cached=False)
        return wav, duration

    def _resolve_language(self, audio_key: str, audio) -> tuple[str, float | None]:
        if self.config.source_language != AUTO_DETECT:
            return self.config.source_language, None
        path = self.job_dir / f"language.{audio_key}.json"
        if path.is_file():
            data = read_json(path)
            return data["language"], data["probability"]
        self._progress("transcribe", 0.0, "detecting_language")
        language, probability = self._ensure_asr().detect_language(audio)
        atomic_write_json(path, {"language": language, "probability": probability})
        log.info("Detected language %s (p=%.2f)", language, probability)
        if probability < 0.5:
            self._warn(f"Language detection is uncertain: {language} (p={probability:.2f})")
        return language, probability

    # -- speech recognition ----------------------------------------------------------------

    def _enhance_wanted(self) -> bool:
        if self._enhancer is None:
            return False
        setting = self._asr_settings.get("enhance") or "auto"
        return setting == "on" or (setting == "auto" and self.config.mode == Mode.MAXIMUM_ACCURACY)

    def _enhanced_audio(self, audio_key: str, audio):
        """Voice-only audio (music and noise removed, quiet speech raised); cached per job as float16."""
        path = self.job_dir / f"enhanced.{audio_key}.v{enhance.VERSION}.npy"
        if path.is_file():
            try:
                return np.load(path).astype(np.float32)
            except (OSError, ValueError):
                path.unlink(missing_ok=True)
        started = time.monotonic()
        self._progress("transcribe", 0.0, "enhancing_audio")
        result = self._enhancer(audio, lambda f: self._progress("transcribe", 0.1 * f, "enhancing_audio"),
                                self.cancel)
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, result.astype(np.float16))
        os.replace(tmp, path)
        log.info("Audio enhanced in %.1f s", time.monotonic() - started)
        return result

    def _transcribe_key(self, audio_key: str, language: str, enhanced_audio: bool) -> str:
        return stable_hash({
            "v": PIPELINE_VERSION, "audio": audio_key, "language": language,
            "model": self._asr_plans[0].model, "beam": self._params["asr_beam"],
            "engine": "whisper", "enhance": enhance.VERSION if enhanced_audio else 0,
            "filter": ASR_FILTER_VERSION, "vad": self._asr_options().vad_parameters(),
            "batch": self._asr_options().batch_size,
        })

    def _known_language(self, audio_key: str) -> str | None:
        """The source language when it is already known without the GPU (set by the user, or detected earlier)."""
        if self.config.source_language != AUTO_DETECT:
            return self.config.source_language
        path = self.job_dir / f"language.{audio_key}.json"
        try:
            return read_json(path)["language"] if path.is_file() else None
        except (OSError, ValueError, KeyError):
            return None

    def _speech_stages(self, audio_key: str, wav: Path, duration: float) -> tuple[dict, str]:
        """Voice separation (CPU), speech recognition (GPU) and speaker detection (CPU, in parallel with the GPU).

        Each part holds only the machine resource it uses, so other episodes of the queue can use the rest (D-120):
        the next episode's voice separation or speaker detection runs while this one is on the GPU."""
        audio = load_wav(wav)
        asr_audio = audio
        enhanced = self._enhance_wanted()
        if enhanced:
            known = self._known_language(audio_key)
            if known is not None and self._cached("transcribe", self._transcribe_key(audio_key, known, True)):
                asr_audio = None             # cached transcript: the voice-only audio is loaded only if still needed
            else:
                try:
                    with self._using(CPU, "transcribe"):
                        asr_audio = self._enhanced_audio(audio_key, audio)
                except (JobCancelled, PipelineError):
                    raise
                except Exception as exc:
                    self._warn(f"Audio enhancement failed ({exc}); transcribing the original audio")
                    enhanced = False
                    asr_audio = audio
        diarize_key = stable_hash({"v": PIPELINE_VERSION, "audio": audio_key, "diarize": diarize.VERSION,
                                   "enhance": enhance.VERSION if enhanced else 0,
                                   "vad": self._asr_options().vad_threshold})
        diarization = None
        try:
            if asr_audio is None and self._diarize_wanted() and not self._cached("diarize", diarize_key):
                with self._using(CPU, "transcribe"):
                    asr_audio = self._enhanced_audio(audio_key, audio)
            if asr_audio is not None:
                diarization = self._start_diarize(diarize_key, asr_audio)
            elif self._diarize_wanted():
                diarization = _Diarization(diarize_key, cached=True)
            with self._using(GPU, "transcribe"):
                try:
                    language, probability = self._resolve_language(audio_key, audio)
                    if language == self.config.target_language:
                        self._warn(f"Source language {language} equals the target language")
                    transcribe_key = self._transcribe_key(audio_key, language, enhanced)
                    if asr_audio is not audio:
                        del audio                # language known: only the voice-only audio is needed now
                    doc, transcribe_key = self._stage_transcribe(
                        transcribe_key, asr_audio, language, probability, duration,
                        alt_audio=(lambda: load_wav(wav)) if enhanced else None)
                finally:
                    self._release_asr()      # GPU memory is free before the next job gets the GPU
            del asr_audio
            self._finish_diarize(diarization, doc)
            diarization = None
        finally:
            self._stop_diarize(diarization)
        return doc, transcribe_key

    def _stage_transcribe(self, key: str, audio, language: str, language_probability: float | None,
                          duration: float, alt_audio: Callable | None = None) -> tuple[dict, str]:
        """Returns the master transcript and the cache key it is valid for: when a fallback plan (another
        model or the CPU) produced any segment, the key names the engines used, so the result is never reused
        as if the first plan had made it and the next run transcribes again (D-047)."""
        started = time.monotonic()
        doc_name = f"master.{key}.json"
        if self._cached("transcribe", key):
            self._record("transcribe", "completed", key)
            self._finish_stage("transcribe", started, None, cached=True)
            return read_json(self.job_dir / doc_name), key
        self._record("transcribe", "running", key)
        partial = self.job_dir / f"transcribe.{key}.partial.jsonl"
        records = load_jsonl(partial)
        if records:
            log.info("Resuming transcription after %.1f s (%d segments cached)",
                     records[-1]["segment"]["end"], len(records))
        span = 0.9                                      # first 10 %: audio enhancement
        windows = plan_windows(audio, self._asr_options().vad_threshold)
        while True:
            engine = self._ensure_asr()
            offset = records[-1]["segment"]["end"] if records else 0.0
            # The previous text conditions the next window; never seed it with a probable hallucination.
            prompt = next((r["segment"]["text"] for r in reversed(records)
                           if not hallucination.reason(r["segment"])), None)
            try:
                with JsonlWriter(partial) as writer:
                    for w_start, w_end in windows:
                        if w_end <= offset + 0.5:
                            continue
                        begin = max(w_start, offset)
                        piece = audio[int(begin * SAMPLE_RATE):int(w_end * SAMPLE_RATE)]
                        if len(piece) <= SAMPLE_RATE // 2:
                            continue
                        for segment in engine.transcribe(piece, language, begin, prompt):
                            self._check_cancel()
                            if segment.end <= offset:
                                continue
                            record = {"segment": segment.to_dict(), "engine": engine.name, "plan": self._asr_index}
                            writer.write(record)
                            records.append(record)
                            offset = segment.end
                            prompt = None if hallucination.reason(record["segment"]) else segment.text
                            self._progress("transcribe", 0.1 + span * (segment.end / duration if duration else 0.0))
                        self._check_cancel()
                break
            except (JobCancelled, PipelineError):
                raise
            except Exception as exc:
                plan = self._asr_plans[self._asr_index]
                if plan.device != "cuda" or self._asr_index + 1 >= len(self._asr_plans):
                    raise PipelineError(f"Transcription failed: {exc}", "error.asr_failed") from exc
                self._warn(f"Transcription failed on {plan.device} ({exc}); continuing with the next plan")
                self._release_asr()
                self._asr_index += 1

        redecoded = {"selected": 0, "replaced": 0, "spans": []}
        try:
            redecoded = redecode.redecode(
                engine, [audio] + ([alt_audio] if alt_audio else []), records, language, duration,
                self.job_dir / f"redecode.{key}.partial.jsonl", self._check_cancel)
        except (JobCancelled, PipelineError):
            raise
        except Exception as exc:
            self._warn(f"Re-decoding low-confidence segments failed ({exc}); keeping the first decode")
        kept = []
        removed = 0
        for record in records:
            why = hallucination.reason(record["segment"])
            if why:
                removed += 1
                seg = record["segment"]
                log.info("Removed probable ASR hallucination at %.1f-%.1f s (%s): %s",
                         seg["start"], seg["end"], why, seg["text"])
            else:
                kept.append(record)
        if removed:
            log.info("Hallucination filter removed %d of %d segments", removed, len(records))
        if not kept:
            raise PipelineError("No speech was detected in the audio", "error.no_speech")
        segments = mt.build_segments(
            [AsrSegment.from_dict(r["segment"]) for r in kept], language, [r["engine"] for r in kept])
        job_info = {
            "input": self.config.source_url or str(self.config.input_path),
            "mode": self.config.mode.value,
            "target_language": self.config.target_language,
            "pipeline_version": PIPELINE_VERSION,
        }
        doc = mt.make_document(segments, language=language, language_probability=language_probability,
                               duration=duration, job=job_info)
        doc["asr"] = {"engine": "whisper", "hallucinations_removed": removed,
                      "redecoded": {k: redecoded[k] for k in ("selected", "replaced")}}
        problems = mt.validate_document(doc)
        if problems:
            self._warn(f"Master transcript has structural problems: {problems[:5]}")
        engines = sorted({r["engine"] for r in records})
        if any(r.get("plan", 0) > 0 for r in records):
            self._warn(f"Transcribed with a fallback engine ({', '.join(engines)}); the result is not cached as "
                       f"{self._asr_plans[0].model} and the next run transcribes again")
            if partial.exists():
                partial.replace(partial.with_name(partial.name + ".fallback"))
            key = stable_hash({"transcribe": key, "engines": engines})
            doc_name = f"master.{key}.json"
        atomic_write_json(self.job_dir / doc_name, doc)
        info = {"segments": len(segments), "engines": engines, "hallucinations_removed": removed}
        self._complete("transcribe", key, [doc_name], info)
        self._record("transcribe", "completed", key)
        self._finish_stage("transcribe", started, duration, cached=False, segments=len(segments), engines=engines,
                           hallucinations_removed=removed)
        return doc, key

    def _stage_translate(self, key: str, doc: dict, duration: float) -> dict:
        started = time.monotonic()
        out_name = f"translation.{key}.json"
        if self._cached("translate", key):
            self._record("translate", "completed", key)
            self._finish_stage("translate", started, None, cached=True)
            return read_json(self.job_dir / out_name)
        self._record("translate", "running", key)
        source_language = doc["language"]["code"]
        target_language = self.config.target_language
        units = segment_cues(doc["segments"])
        partial = self.job_dir / f"translate.{key}.partial.jsonl"
        done = {r["unit"]: r for r in load_jsonl(partial)}
        if done:
            log.info("Resuming translation: %d of %d units cached", len(done), len(units))
        remaining = [u for u in units if u.id not in done]
        if remaining:
            # The local model needs the GPU: one job at a time, and its memory is freed before the slot is (D-120).
            with self._using(GPU, "translate"):
                started = time.monotonic()           # waiting for the GPU is not part of the stage's own time
                self._release_asr()
                try:
                    self._translate_units(units, done, remaining, partial, source_language, target_language)
                finally:
                    self._release_mt()

        result_units = []
        previous = None
        for unit in units:
            record = done[unit.id]
            flags = check_line(unit.text, record["text"], target_language, previous)
            previous = (unit.text, record["text"])
            result_units.append({**unit.to_dict(), "translation": record["text"], "engine": record["engine"],
                                 "flags": flags})

        result = {"source_language": source_language, "target_language": target_language, "units": result_units}
        atomic_write_json(self.job_dir / out_name, result)
        engines = sorted({u["engine"] for u in result_units})
        self._complete("translate", key, [out_name], {"units": len(result_units), "engines": engines})
        self._record("translate", "completed", key)
        self._finish_stage("translate", started, duration, cached=False, units=len(result_units), engines=engines)
        return result

    def _translate_units(self, units, done: dict, remaining: list, partial: Path, source_language: str,
                         target_language: str) -> None:
        """Translate `remaining` with the local engine, moving to the next plan when one fails at runtime."""
        while remaining:
            backend = self._ensure_mt()
            try:
                with JsonlWriter(partial) as writer:
                    while remaining:
                        self._check_cancel()
                        batch, remaining = remaining[:_TRANSLATION_BATCH], remaining[_TRANSLATION_BATCH:]
                        context = [(u.text, done[u.id]["text"]) for u in units
                                   if u.id < batch[0].id and u.id in done][-3:]
                        outputs = backend.translate([u.text for u in batch], source_language, target_language,
                                                    context=context)
                        if len(outputs) != len(batch):
                            raise PipelineError("Translation backend returned a wrong number of results")
                        for unit, text in zip(batch, outputs):
                            record = {"unit": unit.id, "text": text, "engine": backend.name}
                            writer.write(record)
                            done[unit.id] = record
                        self._progress("translate", len(done) / len(units))
            except (JobCancelled, PipelineError):
                raise
            except Exception as exc:
                plan = self._mt_plans[self._mt_index]
                remaining = [u for u in units if u.id not in done]
                if plan.device not in _FALLBACK_DEVICES or self._mt_index + 1 >= len(self._mt_plans):
                    raise PipelineError(f"Translation failed: {exc}", "error.mt_failed") from exc
                self._warn(f"Translation failed on {plan.device} ({exc}); continuing with the next plan")
                self._release_mt()
                self._mt_index += 1

    def _link_input_video(self, out_dir: Path, stem: str, paths: dict) -> None:
        """Put the local video next to its subtitle: a hard link (no extra disk space) when possible."""
        target = out_dir / f"{safe_basename(stem)}{self.config.input_path.suffix}"
        if target.exists():
            paths["video"] = target
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.link(self.config.input_path, target)
            paths["video"] = target
        except OSError as exc:
            log.info("Video not linked into the output folder (%s); the original stays at %s",
                     exc, self.config.input_path)

    def _link_job_audio(self, work: Path, paths: dict) -> None:
        """Keep the extracted 16 kHz WAV with the episode, so the review editor can re-transcribe a span
        without the job cache: a hard link (no extra disk space) when possible, otherwise a copy."""
        source = self.job_dir / "audio.wav"
        target = work / "audio.wav"
        if not source.is_file():
            return
        if not target.is_file():
            work.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError as exc:
                log.info("Audio not linked into the episode folder (%s); copying it", exc)
                try:
                    shutil.copyfile(source, target)
                except OSError as copy_exc:
                    log.warning("Could not place the episode audio at %s: %s", target, copy_exc)
                    return
        paths["audio"] = target

    def _empty_translation(self, doc: dict) -> dict:
        units = [{**u.to_dict(), "translation": "", "engine": None, "flags": []} for u in segment_cues(doc["segments"])]
        return {"source_language": doc["language"]["code"], "target_language": self.config.target_language,
                "units": units}

    def _fill_missing(self, translation: dict) -> dict:
        """MADLAD fallback for lines without any translation (AI unavailable or failed)."""
        missing = [u for u in translation["units"] if not u["translation"].strip()]
        if not missing:
            return translation
        log.info("Translating %d lines with the offline fallback engine", len(missing))
        with self._using(GPU, "export"):
            return self._fill_missing_now(translation, missing)

    def _fill_missing_now(self, translation: dict, missing: list[dict]) -> dict:
        # Runs after all AI work: an engine failure moves on to the next plan, and if every plan fails the job
        # still exports (those lines stay empty and are flagged) instead of losing the finished translation.
        while missing:
            try:
                backend = self._ensure_mt()
            except PipelineError as exc:
                self._warn(f"Offline translation unavailable ({exc}); {len(missing)} lines stay untranslated")
                break
            try:
                for i in range(0, len(missing), _TRANSLATION_BATCH):
                    self._check_cancel()
                    batch = missing[i:i + _TRANSLATION_BATCH]
                    outputs = backend.translate([u["text"] for u in batch], translation["source_language"],
                                                translation["target_language"])
                    for unit, text in zip(batch, outputs):
                        unit["translation"] = text
                        unit["engine"] = backend.name
                        unit["flags"] = (check_line(unit["text"], text, translation["target_language"])
                                         + ["not_refined_by_ai"])
                missing = []
            except (JobCancelled, PipelineError):
                raise
            except Exception as exc:
                missing = [u for u in missing if not u["translation"].strip()]
                self._warn(f"Offline translation failed on {self._mt_plans[self._mt_index].device} ({exc}); "
                           f"trying the next plan")
                self._mt_index += 1
            finally:
                self._release_mt()
        return translation

    def _normalize_names(self, translation: dict) -> dict:
        """Deterministic name normalisation (D-058, task 2.6)."""
        if not self._name_normalization:
            log.info("Name normalisation disabled by setting")
            return translation
        target_language = translation.get("target_language") or self.config.target_language
        if not names_supported(target_language):
            return translation
        source_language = translation.get("source_language") or "auto"
        series_name = self.series.to_dict().get("series_name")
        series_glossary = SeriesGlossary.for_series(self.jobs_dir.parent, series_name, target_language)
        glossary = dict(series_glossary.names)
        glossary.update(translation.get("summary", {}).get("glossary") or {})
        if not glossary:
            return translation
        normalized_units = names.normalize_units(
            translation["units"],
            glossary=glossary,
            user_names=series_glossary.user,
            source_language=source_language,
            target_language=target_language,
        )
        return {**translation, "units": normalized_units}

    def _media_summary(self) -> dict:
        info = self._media_info or {}
        series = {k: v for k, v in self.series.to_dict().items() if v not in (None, "") and k != "source"}
        summary = {"title": info.get("title") or (self.config.input_path.stem if self.config.input_path else None),
                   "url": self.config.source_url, "channel": info.get("channel"), **series}
        return {k: v for k, v in summary.items() if v}

    @staticmethod
    def _annotate_speakers(doc: dict, translation: dict) -> None:
        """Give every translation unit the voice id ("S1", ...) of its first segment (None without diarization)."""
        segments = {s["id"]: s for s in doc["segments"]}
        for unit in translation["units"]:
            unit["speaker"] = next((segments[i].get("speaker") for i in unit.get("segment_ids", [])
                                    if i in segments and segments[i].get("speaker")), None)
            unit["audio_confidence"] = finalize.unit_audio_confidence(unit, segments)
            unit.pop("speaker_split", None)
            split = finalize.speaker_split(unit, segments)
            if split:
                unit["speaker"] = split["first"]
                unit["speaker_split"] = split

    def _speaker_map(self, key: str, refiner, source_language: str, units: list[dict]) -> dict | None:
        """Voice id -> character and gender for the AI translation (D-099); cached; never fatal."""
        if not any(u.get("speaker") for u in units):
            return None
        path = self.job_dir / f"speakers.{key}.json"
        if path.is_file():
            try:
                return read_json(path)
            except (OSError, ValueError):
                pass
        try:
            result = brief.build_speaker_map(refiner.pool, source_language, refiner.brief, units,
                                             sleep=self._sleep_unless_cancelled)
        except JobCancelled:
            raise
        except Exception as exc:                  # never fail the stage for the speaker map
            log.warning("Speaker map failed: %s", exc)
            result = None
        if result is None:
            log.warning("AI translation: speaker map unavailable; lines carry the voice id only")
            return None
        atomic_write_json(path, result)
        log.info("Speaker map ready: %s", {k: (v["name"] or "?", v["gender"]) for k, v in result.items()})
        return result

    def _episode_brief(self, key: str, refiner, source_language: str, target_language: str,
                       units: list[dict]) -> dict | None:
        """Episode brief for the AI translation (D-055); cached only when it has content. Never fatal."""
        path = self.job_dir / f"brief.{key}.json"
        if path.is_file():
            try:
                return read_json(path)
            except (OSError, ValueError):
                pass
        started = time.monotonic()
        self._progress("refine", 0.0, "episode brief")
        try:
            result = brief.build_brief(refiner.pool, source_language, target_language, units, self._media_summary(),
                                       sleep=self._sleep_unless_cancelled)
        except JobCancelled:
            raise
        except Exception as exc:                  # never fail the stage for the brief
            log.warning("Episode brief failed: %s", exc)
            result = None
        if result is None:
            self._warn("AI translation: episode brief unavailable; translating without it")
            return None
        atomic_write_json(path, result)
        log.info("Episode brief ready in %.1f s: %d characters, %d relations, %d terms",
                 time.monotonic() - started, len(result["characters"]), len(result["relations"]),
                 len(result["terms"]))
        return result

    def _condense_after_timing(self, refiner, units: list[dict], target_language: str,
                               glossary: dict[str, str]) -> int:
        """Lines that are still read too fast after timing get one more condense request (task 5.2, step 4).
        Never fatal; returns the number of lines shortened."""
        try:
            probe = copy.deepcopy(units)
            cues = {c.uid: c for c in finalize.finalize_units(probe, [], target_language)}
            cps = style.cps(target_language)
            lines = []
            for unit in probe:
                cue = cues.get(unit["id"])
                if cue is not None and "reading_speed" in unit.get("qa_flags", []):
                    budget = max(12, int((cue.end - cue.start) * cps))
                    lines.append({"id": unit["id"], "source": unit["text"], "max_chars": budget})
            if not lines:
                return 0
            by_id = {u["id"]: u for u in units}
            changed = 0
            for first in range(0, len(lines), 40):
                self._check_cancel()
                chunk = lines[first:first + 40]
                result = BlockResult(translations={l["id"]: by_id[l["id"]]["translation"] for l in chunk},
                                     translator=None, changed=set())
                refiner._condense(chunk, result, glossary, factor=1.0)
                for line_id in result.changed or ():
                    unit = by_id[line_id]
                    unit["translation"] = unit["llm_translation"] = result.translations[line_id]
                    unit["ai_corrected"] = True
                    changed += 1
            log.info("Condensed %d of %d lines that were still too fast after timing", changed, len(lines))
            return changed
        except JobCancelled:
            raise
        except Exception as exc:
            log.warning("Condense after timing failed: %s", exc)
            return 0

    def _stage_refine(self, key: str, translation: dict, duration: float) -> dict:
        """Correct (or retranslate) the local translation with cloud LLMs block by block (resumable, never fatal)."""
        started = time.monotonic()
        out_name = f"refined.{key}.json"
        if self._cached("refine", key):
            refined = read_json(self.job_dir / out_name)
            self.stats["refine"] = refined.get("summary", {})
            self._record("refine", "completed", key)
            self._finish_stage("refine", started, None, cached=True)
            return refined
        if self._refiner_factory is None:
            self._skip_stage("refine", "disabled")
            return translation
        with self._using(AI, "refine"):
            return self._refine_now(key, translation, duration, time.monotonic())

    def _refine_now(self, key: str, translation: dict, duration: float, started: float) -> dict:
        out_name = f"refined.{key}.json"
        self._record("refine", "running", key)
        source_language = translation["source_language"]
        target_language = translation["target_language"]
        self._progress("refine", 0.0, "connecting")
        refiner = self._refiner_factory(source_language, target_language, self._media_summary())
        self._check_cancel()
        if refiner is not None and hasattr(refiner, "set_cancel"):
            refiner.set_cancel(self.cancel)       # a cancelled job must not wait for a slow provider
        if refiner is None:
            self._warn("No usable AI translation provider (missing API keys or none reachable); "
                       "the basic translation was kept")
            self._skip_stage("refine", "no providers")
            return translation
        units = translation["units"]
        refiner.brief = self._episode_brief(key, refiner, source_language, target_language, units)
        speakers = self._speaker_map(key, refiner, source_language, units)
        blocks = [units[i:i + REFINE_BLOCK] for i in range(0, len(units), REFINE_BLOCK)]
        block_of = {u["id"]: bi for bi, block in enumerate(blocks) for u in block}
        partial = self.job_dir / f"refine.{key}.partial.jsonl"
        done = {r["block"]: r for r in load_jsonl(partial)}
        check_names = names_supported(target_language)
        series_glossary = SeriesGlossary.for_series(self.jobs_dir.parent, self.series.to_dict().get("series_name"),
                                                    target_language)
        glossary: dict[str, str] = dict(series_glossary.names)
        for bi in sorted(done):
            for name, spelling in done[bi].get("names", {}).items():
                if not check_names or transliteration_ok(name, spelling):
                    glossary.setdefault(name, spelling)
        glossary.update(series_glossary.names)          # names fixed by the user always win
        refiner.known_names.update(glossary)
        incomplete: dict[int, dict] = {}                # blocks with missing lines: used now, retried next run
        position = {"block": 0}

        def block_progress(fraction: float) -> None:
            """Progress inside the block being translated (translate -> check names -> double-check -> condense)."""
            self._progress("refine", (len(done) + fraction * 0.98) / len(blocks), f"{position['block'] + 1}/{len(blocks)}")

        refiner.on_step = block_progress

        def current_text(unit: dict) -> str:
            record = done.get(block_of[unit["id"]])
            if record:
                return record["translations"].get(str(unit["id"]), unit["translation"])
            return unit["translation"]

        with JsonlWriter(partial) as writer:
            for bi, block in enumerate(blocks):
                if bi in done:
                    continue
                self._check_cancel()
                position["block"] = bi
                if not refiner.has_routes():
                    self._warn("All AI providers are rate limited or unavailable; remaining lines keep the "
                               "basic translation (run the job again later to finish them)")
                    break
                first = units.index(block[0])
                previous = [{"source": u["text"], "translation": current_text(u),
                             **brief.speaker_fields(u.get("speaker"), speakers)}
                            for u in units[max(0, first - REFINE_CONTEXT_BEFORE):first]]
                following = [{"source": u["text"], **brief.speaker_fields(u.get("speaker"), speakers)}
                             for u in units[first + len(block):first + len(block) + REFINE_CONTEXT_AFTER]]
                lines = [{"id": u["id"], "source": u["text"], "draft": u["translation"],
                          "max_chars": style.max_chars(max(0.0, float(u.get("end", 0.0)) - float(u.get("start", 0.0))),
                                                       target_language),
                          "audio_confidence": u.get("audio_confidence"),
                          **brief.speaker_fields(u.get("speaker"), speakers)}
                         for u in block]
                result = refiner.translate_block(lines, previous, following, glossary)
                for note in result.notes:
                    log.info("Block %d: %s", bi, note)
                if not result.translations:
                    self._warn(f"AI translation failed for lines {block[0]['id'] + 1}-{block[-1]['id'] + 1}; "
                               f"kept the basic translation")
                    continue
                record = {"block": bi, "translations": {str(k): v for k, v in result.translations.items()},
                          "changed": sorted(result.changed) if result.changed is not None else None,
                          "weak": sorted(result.weak),
                          "name_issues": {str(k): v for k, v in result.name_issues.items()},
                          "translator": result.translator, "reviewer": result.reviewer, "judge": result.judge,
                          "risky": {str(k): v for k, v in result.risky.items()},
                          "disagreement": {str(k): v for k, v in result.disagreement.items()},
                          "second": {str(k): v for k, v in result.second.items()},
                          "double_check_requests": result.double_check_requests,
                          "corrections": {str(k): v for k, v in result.corrections.items()},
                          "names": result.names, "notes": result.notes,
                          "condense_requests": result.condense_requests}
                for name, spelling in result.names.items():
                    glossary.setdefault(name, spelling)   # first spelling wins: names stay consistent
                missing = [u["id"] for u in block if str(u["id"]) not in record["translations"]]
                if missing:
                    self._warn(f"AI translation returned {len(block) - len(missing)} of {len(block)} lines for "
                               f"lines {block[0]['id'] + 1}-{block[-1]['id'] + 1}; the block is retried on the "
                               f"next run")
                    incomplete[bi] = record
                    continue
                writer.write(record)
                done[bi] = record
                self._progress("refine", len(done) / len(blocks))

        refined_units = []
        previous_pair = None
        providers: set[str] = set()
        corrected = refined_count = ai_changed = 0
        for unit in units:
            record = done.get(block_of[unit["id"]]) or incomplete.get(block_of[unit["id"]])
            llm = record["translations"].get(str(unit["id"])) if record else None
            fix = record["corrections"].get(str(unit["id"])) if record else None
            changed = record.get("changed") if record else None
            # Reviewer output is a suggestion for the human reviewer, never applied automatically: on the real
            # test episode 2 of 6 automatic corrections reversed correct translations (DECISIONS D-026).
            final = llm or unit["translation"]
            # "correct" mode: an accepted line keeps its local engine; a corrected line names the AI model.
            ai_wrote = bool(llm) and (changed is None or unit["id"] in changed)
            second_model = (record.get("second") or {}).get(str(unit["id"])) if record else None
            engine = (second_model or record["translator"]) if ai_wrote else unit["engine"]
            disagreement = (record.get("disagreement") or {}).get(str(unit["id"])) if record else None
            if llm:
                refined_count += 1
                providers.update(p.split(":")[0] for p in (record["translator"], record["reviewer"],
                                                           record.get("judge"), second_model) if p)
            if llm and changed is not None and unit["id"] in changed:
                ai_changed += 1
            corrected += 1 if fix else 0
            flags = check_line(unit["text"], final, target_language, previous_pair)
            if not llm:
                flags.append(confidence.NOT_REFINED_BY_AI)
            if fix:                                 # only records written before D-104 have these
                flags.append(confidence.REVIEWER_SUGGESTION)
            if disagreement:
                flags.append(confidence.AI_DISAGREEMENT)
            if llm and unit["id"] in (record.get("weak") or ()):
                flags.append(confidence.WEAK_AI_MODEL)
            if llm and str(unit["id"]) in (record.get("name_issues") or {}):
                flags.append(confidence.NAME_MISMATCH)
            previous_pair = (unit["text"], final)
            refined_units.append({**unit, "draft": unit["translation"], "llm_translation": llm, "review": fix,
                                  "disagreement": disagreement,
                                  "translation": final, "engine": engine, "flags": flags,
                                  "ai_corrected": bool(llm) and changed is not None and unit["id"] in changed})
        if len(done) == len(blocks) and refiner.has_routes():
            self._condense_after_timing(refiner, refined_units, target_language, glossary)
        if len(done) == len(blocks) and check_names and series_glossary.learn(glossary):
            series_glossary.save()
            log.info("Series glossary updated: %s", series_glossary.path)
        tokens = {c.name: dict(c.usage) for c in refiner.pool.clients if (getattr(c, "usage", None) or {}).get("requests")}
        for client in refiner.pool.clients:
            for model, used in (getattr(client, "usage_by_model", None) or {}).items():
                log.info("AI usage %s:%s: %d requests, %d input + %d output tokens", client.name, model,
                         used["requests"], used["prompt_tokens"], used["completion_tokens"])
        total_condense = sum(r.get("condense_requests", 0) for r in done.values())
        double_check = {"risky_lines": sum(len(r.get("risky") or {}) for r in done.values()),
                        "requests": sum(r.get("double_check_requests", 0) for r in done.values()),
                        "second_used": sum(len(r.get("second") or {}) for r in done.values()),
                        "disagreements": sum(len(r.get("disagreement") or {}) for r in done.values())}
        summary = {"lines": len(units), "refined": refined_count, "suggestions": corrected,
                   "mode": getattr(refiner, "mode", "translate"), "corrected": ai_changed,
                   "providers": sorted(providers), "glossary": glossary, "tokens": tokens,
                   "condense_requests": total_condense, "double_check": double_check}
        self.stats["refine"] = summary
        self.stats["condense_requests"] = total_condense
        result = {**translation, "units": refined_units, "summary": summary}
        if len(done) == len(blocks):
            atomic_write_json(self.job_dir / out_name, result)
            self._complete("refine", key, [out_name], summary)
            self._record("refine", "completed", key)
        else:
            self._record("refine", "pending", key, "incomplete; resumes on the next run")
        self._finish_stage("refine", started, duration, cached=False, refined=refined_count, lines=len(units))
        return result

    def _frame_timing(self) -> tuple[float | None, list[float] | None]:
        """Frame rate of the video and, when enabled, its shot changes; any failure only skips the snapping."""
        if self._media_path is None or not self._media_path.is_file():
            return None, None
        try:
            fps = frames.video_fps(self._media_path)
            shots = None
            if fps and self._snap_shots:
                with self._using(CPU, "export"):
                    self._progress("export", 0.0, "detecting_shots")
                    shots = frames.shot_changes(self._media_path, self.job_dir / "shots.json",
                                                cancel=self.cancel) or None
            return fps, shots
        except JobCancelled:
            raise
        except Exception as exc:
            self._warn(f"Frame timing skipped ({type(exc).__name__}: {exc})")
            return None, None

    def _stage_export(self, doc: dict, translation: dict, evidence: list[dict]) -> tuple[Path, dict[str, str]]:
        started = time.monotonic()
        self._record("export", "running")
        if self.config.input_path is not None:
            stem = self.config.input_path.stem
            out_dir = self._output_root() / safe_basename(stem)
        else:
            stem = (self._media_info or {}).get("title") or "Video"
            out_dir = self._out_dir or self._output_root() / safe_basename(stem)
        source_language = doc["language"]["code"]
        target_language = self.config.target_language
        paths = output_paths(out_dir, stem, target_language, source_language)
        if self.config.input_path is not None:
            self._link_input_video(out_dir, stem, paths)
        elif self._media_path is not None:
            paths["video"] = self._media_path
        work = out_dir / "work"
        self._link_job_audio(work, paths)

        fps, shots = self._frame_timing()
        target_cues = finalize.finalize_units(translation["units"], doc["segments"], target_language,
                                              fps=fps, shots=shots)
        source_cues = finalize.source_cues(translation["units"])
        if translation["source_language"].split("-")[0] in RTL_LANGUAGES:
            source_cues = [replace(c, text=mark_rtl(c.text)) for c in source_cues]
        write_srt(source_cues, paths["source_srt"])
        write_srt(target_cues, paths["target_srt"])
        to_review = finalize.write_review_file(paths["review_required"], translation["units"])
        counts = {c: sum(1 for u in translation["units"] if u["confidence"] == c) for c in ("HIGH", "MEDIUM", "LOW")}
        self.stats["confidence"] = {**counts, "needs_review": to_review}
        if to_review:
            self._warn(f"{to_review} of {len(translation['units'])} subtitle lines need review "
                       f"(see ReviewRequired.txt or the review window)")
        for index, record in enumerate(evidence):
            # Fetched subtitles are kept for comparison; they are not the final output (DECISIONS D-018).
            name = f"{safe_basename(stem)}_ref{index + 1}_{record['provider']}_{record['kind']}_{record['language']}.srt"
            ev = SubtitleEvidence.from_dict(record)
            if ev.cues:
                path = work / "sources" / name
                write_srt(ev.cues, path)
                paths[f"evidence_{index + 1}"] = path

        unit_of_segment: dict[int, int] = {}
        for u in translation["units"]:
            for sid in u["segment_ids"]:
                unit_of_segment.setdefault(sid, u["id"])
        export_doc = {
            **doc,
            "media": {**doc["media"], **({"fps": fps} if fps else {}), **({"shot_changes": shots} if shots else {})},
            "segments": [{**s, "translation_unit": unit_of_segment.get(s["id"])} for s in doc["segments"]],
            "translation_units": translation["units"],
            "series": self.series.to_dict(),
            "media_info": self._media_info,
            "subtitle_evidence": [{k: v for k, v in r.items() if k != "cues"} for r in evidence],
            "stats": self.stats,
            "warnings": self.warnings,
        }
        atomic_write_json(paths["master_transcript"], export_doc)
        job_log = self.job_dir / "job.log"
        if job_log.is_file():
            shutil.copyfile(job_log, paths["processing_log"])
        self._record("export", "completed")
        self._finish_stage("export", started, None, cached=False)
        return out_dir, {k: str(v) for k, v in paths.items()}

    # -- entry point -----------------------------------------------------------------------

    def run(self) -> JobResult:
        if self.config.source_url:
            key = stable_hash({"url": self.config.source_url.strip()})
        else:
            key = stable_hash({"input": file_fingerprint(self.config.input_path)}) \
                if self.config.input_path.is_file() else None
        if key is None:
            return self._run()
        with _exclusive_job(key, self.cancel):
            return self._run()

    def _run(self) -> JobResult:
        if self.config.source_url:
            identity = {"url": self.config.source_url.strip()}
        else:
            input_path = self.config.input_path
            if not input_path.is_file():
                raise PipelineError(f"Input file not found: {input_path}", "error.input_not_found", path=str(input_path))
            identity = {"input": file_fingerprint(input_path)}
        self.job_key = stable_hash(identity)
        self.job_dir = self.jobs_dir / self.job_key
        (self.job_dir / "stages").mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.job_dir / "input.json", identity)

        handler = logging.FileHandler(self.job_dir / "job.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        # Several jobs run at once (D-120): this job's log keeps only the lines of its own threads.
        self._log_threads = _ThreadFilter()
        self._log_threads.add_current()
        handler.addFilter(self._log_threads)
        root = logging.getLogger()
        root.addHandler(handler)
        started = time.monotonic()
        try:
            log.info("Job %s started: %s", self.job_key, self.config)
            if self.config.source_url:
                self._stage_download(self._download_key(identity))
                identity = {"input": file_fingerprint(self._media_path)}
            else:
                self._skip_stage("download", "local file")
            self._detect_series()
            audio_key = stable_hash({"v": PIPELINE_VERSION, **identity})
            wav, duration = self._stage_audio(audio_key)
            self._notify(f"media_duration:{duration}")   # ETA only; no stage progress
            doc, transcribe_key = self._speech_stages(audio_key, wav, duration)
            self.stats["asr"] = doc.get("asr", {"engine": "whisper"})
            subtitles_key = stable_hash({
                "v": PIPELINE_VERSION, "transcribe": transcribe_key, "target": self.config.target_language,
                "providers": sorted(p.name for p in self._providers),
                "series": self.series.to_dict(),
            })
            evidence = self._stage_subtitles(subtitles_key, doc)
            replaced = platform_text.apply(doc, evidence, EVIDENCE_MATCH_FROM)
            if replaced:
                log.info("Completed %d segment texts from the platform's manual subtitles", len(replaced))
                self.stats["platform_text"] = {"replaced": len(replaced), "changes": replaced[:200]}
                # The source text changed, so every later cache key must differ from a run without it.
                transcribe_key = stable_hash({"transcribe": transcribe_key, "platform_text": replaced})
            translate_key = stable_hash({
                "v": PIPELINE_VERSION, "transcribe": transcribe_key, "target": self.config.target_language,
                "model": self._mt_plans[0].model, "beam": self._params["mt_beam"],
            })
            options = getattr(self._refiner_factory, "options", None) or {}
            draft_used = self._refiner_factory is None or options.get("mode", "correct") == "correct"
            if draft_used:
                translation = self._stage_translate(translate_key, doc, duration)
                self._release_mt()
            else:
                # "translate" mode does not need a local draft: skip it and translate locally afterwards only
                # the lines the AI could not handle.
                translation = self._empty_translation(doc)
                self._skip_stage("translate", "not needed (AI translates from scratch)")
            self._annotate_speakers(doc, translation)
            refine_key = stable_hash({
                "speakers": stable_hash([[u.get("speaker"), u.get("speaker_split"), (u.get("audio_confidence") or 1.0) < 0.5]
                                     for u in translation["units"]]),
                # Translate mode does not use the local draft, so changing the local model keeps the AI result.
                "v": PIPELINE_VERSION, "translate": translate_key if draft_used else transcribe_key,
                "segmentation": SEGMENTATION_VERSION, "series": self.series.to_dict(),
                "options": getattr(self._refiner_factory, "options", None), "refine_v": REFINE_VERSION,
            })
            translation = self._stage_refine(refine_key, translation, duration)
            translation = self._fill_missing(translation)
            translation = self._normalize_names(translation)
            self.stats.update({
                "media_duration_s": round(duration, 2),
                "total_elapsed_s": round(time.monotonic() - started, 2),
                "waited_s": round(self._waited, 2),
                "language": doc["language"]["code"],
            })
            out_dir, outputs = self._stage_export(doc, translation, evidence)
            self._finished = True
            log.info("Job %s completed in %.1f s", self.job_key, time.monotonic() - started)
            return JobResult(self.job_key, self.job_dir, out_dir, outputs, self.stats, self.warnings,
                             self.series.to_dict())
        except JobCancelled:
            log.info("Job %s cancelled; completed work is cached", self.job_key)
            raise
        except PipelineError as exc:
            log.error("Job %s failed: %s", self.job_key, exc)
            raise
        except Exception as exc:
            log.exception("Job %s failed unexpectedly", self.job_key)
            raise PipelineError(f"Unexpected error: {type(exc).__name__}: {exc}") from exc
        finally:
            self._release_asr()
            self._release_mt()
            root.removeHandler(handler)
            handler.close()
            if self._finished and not self._keep_cache:
                self._remove_job_cache()

    def _remove_job_cache(self) -> None:
        """A finished episode keeps everything it needs in its output folder (subtitles, work/ with the audio, the
        master transcript, the processing log). The job cache is then only a leftover that would make a re-run
        skip every stage, so it is removed (D-116). Never fatal: a locked file only leaves the folder behind."""
        folder = self.job_dir
        if folder is None or not folder.is_dir() or folder.parent != self.jobs_dir:
            return                                  # never delete anything that is not this job's own folder
        for attempt in range(4):
            shutil.rmtree(folder, ignore_errors=True)
            if not folder.exists():
                log.info("Job cache removed: %s", folder.name)
                return
            time.sleep(0.3 * (attempt + 1))        # antivirus or an indexer may hold a file for a moment
        log.warning("Could not remove the job cache %s completely; it can be deleted by hand", folder)
