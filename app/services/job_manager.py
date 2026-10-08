"""Runs a pipeline job on a worker thread and reports to the GUI through Qt signals."""

from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Signal

from app.core.downloader import YtDlpAdapter, access_options
from app.core.errors import JobCancelled, PipelineError
from app.core.llm_providers import HealthStore, ModelRanking, ProviderCallError, ProviderPool, build_clients
from app.core.modes import Mode
from app.core.transcription import AsrEngine, AsrOptions
from app.utils.paths import default_data_root, model_ranking_files
from app.core.llm_translation import LlmRefiner
from app.core.pipeline import JobConfig, Pipeline
from app.database.database import Database
from app.database.jobs import JobsRepo
from app.models import engines
from app.models.model_manager import DEFAULT_LOCAL_MODEL, DEFAULT_TRANSLATION_MODEL, ModelManager
from app.providers.opensubtitles import OpenSubtitlesProvider
from app.providers.subdl import SubdlProvider
from app.providers.youtube import PlatformSubtitleProvider
from app.services import hardware_detection
from app.services.api_keys import load_keys
from app.services.hardware_detection import EnginePlan

log = logging.getLogger(__name__)

# (config, cancel_event, progress_fn, recorder, extras) -> Pipeline
PipelineFactory = Callable[..., Pipeline]


def build_providers(adapter: YtDlpAdapter, extras: dict) -> list:
    """Subtitle providers enabled by the user's settings (keys are passed outside JobConfig so they never get logged)."""
    providers = []
    if extras.get("fetch_platform_subtitles", True):
        providers.append(PlatformSubtitleProvider(adapter))
    if extras.get("opensubtitles_api_key"):
        providers.append(OpenSubtitlesProvider(extras["opensubtitles_api_key"]))
    if extras.get("subdl_api_key"):
        providers.append(SubdlProvider(extras["subdl_api_key"]))
    return providers


class RefinerFactory:
    """Builds the cloud LLM refiner for a job; returns None when no provider is usable."""

    def __init__(self, mode: str, review: bool, keys_loader=load_keys,
                 ranking_loader=lambda: ModelRanking.load(*model_ranking_files()), health_path=None):
        self.options = {"mode": mode, "review": review}
        self._health_path = health_path if health_path is not None else default_data_root() / "provider_health.json"
        self._load_keys = keys_loader
        self._load_ranking = ranking_loader

    def __call__(self, source_language: str, target_language: str, media: dict) -> LlmRefiner | None:
        clients = build_clients(self._load_keys())
        usable = []
        for client in clients:
            try:
                models = client.discover_models()
            except ProviderCallError as exc:
                log.warning("Provider %s unusable: %s", client.name, exc)
                continue
            if models:
                log.info("Provider %s models: %s", client.name, models)
                usable.append(client)
        if not usable:
            return None
        pool = ProviderPool(usable, self._load_ranking(), health=HealthStore(self._health_path))
        remembered = pool.apply_health()
        if remembered:
            log.info("Skipping %d model/provider entries that failed recently: %s", len(remembered),
                     ", ".join(sorted(remembered)[:12]))
        log.info("AI models, best first: %s", ", ".join(f"{r.key} ({r.score:g}{'' if r.measured else ' est.'})"
                                                         for r in pool.routes()))
        return LlmRefiner(pool, source_language, target_language, media,
                          mode=self.options["mode"], review=self.options["review"])


class AsrEngineFactory:
    """Builds and caches one ASR engine for the review editor's "re-transcribe this span".

    Built like the pipeline's transcribe stage: the hardware's plans are tried in order and the first
    engine that loads is kept. Only models already on disk are used, so editing never starts a
    download. Hardware detection runs on the first call, which the review window makes in its worker
    thread. release() drops the engine (and its VRAM) after a span: the design GPU has only 4 GB.
    """

    def __init__(self, models: ModelManager, installed: list[str], mode: Mode = Mode.BALANCED,
                 options: AsrOptions | None = None):
        self._models = models
        self._installed = list(installed)
        self._mode = mode
        self._options = options or AsrOptions()
        self._plans: list[EnginePlan] | None = None
        self._engine: AsrEngine | None = None

    def plans(self) -> list[EnginePlan]:
        if self._plans is None:
            hw = hardware_detection.detect(self._models.models_dir)
            plans = hardware_detection.recommend_asr(hw, self._mode)
            named = {plan.model for plan in plans}
            # A model the user already has but this mode does not recommend still works on the CPU.
            plans += [EnginePlan(name, "cpu", "int8", "installed model, not recommended for this mode")
                      for name in self._installed if name not in named]
            self._plans = [plan for plan in plans if plan.model in self._installed]
        return self._plans

    def __call__(self) -> AsrEngine | None:
        if self._engine is not None:
            return self._engine
        for plan in self.plans():
            try:
                log.info("Loading review ASR plan %s", plan)
                self._engine = engines.load_asr_engine(plan, self._models, self._options)
                return self._engine
            except Exception as exc:
                log.warning("Review ASR plan %s/%s/%s unavailable: %s",
                            plan.model, plan.device, plan.compute_type, exc)
        return None

    def release(self) -> None:
        if self._engine is not None:
            self._engine = None
            gc.collect()


def translation_plans(hw, extras: dict) -> list[EnginePlan]:
    """Built-in local TranslateGemma first by default (GPU build if CUDA works, then CPU build); MADLAD-400
    always remains as the fallback."""
    plans: list[EnginePlan] = []
    if extras.get("translation_engine", "local") == "local":
        model = extras.get("local_model") or DEFAULT_LOCAL_MODEL
        if hw.cuda_usable:
            plans.append(EnginePlan(model, "local", "cuda", "built-in local AI model on the NVIDIA GPU (llama.cpp)"))
        plans.append(EnginePlan(model, "local", "cpu", "built-in local AI model on the CPU (llama.cpp)"))
    return plans + hardware_detection.recommend_translation(hw, DEFAULT_TRANSLATION_MODEL)


def default_pipeline_factory(jobs_dir: Path, models_dir: Path, data_dir: Path) -> PipelineFactory:
    def factory(config: JobConfig, cancel, progress, recorder, extras: dict | None = None) -> Pipeline:
        hw = hardware_detection.detect(data_dir)
        extras = extras or {}
        asr_plans = hardware_detection.recommend_asr(hw, config.mode)
        mt_plans = translation_plans(hw, extras)
        log.info("ASR plans: %s", asr_plans)
        log.info("Translation plans: %s", mt_plans)
        models = ModelManager(models_dir)
        adapter = YtDlpAdapter(extra_options=access_options(extras))
        return Pipeline(
            config, jobs_dir, asr_plans, mt_plans,
            asr_factory=lambda plan, options, cb: engines.load_asr_engine(plan, models, options, cb),
            mt_factory=lambda plan, beam, cb: engines.load_translation_backend(plan, models, beam, cb),
            progress=progress, cancel=cancel, recorder=recorder,
            downloader=adapter, providers=build_providers(adapter, extras),
            refiner_factory=RefinerFactory("correct" if extras.get("llm_correct_only", False) else "translate",
                                           extras.get("llm_review", True))
            if extras.get("llm_refine", False) else None,
            asr_settings={"enhance": extras.get("audio_enhance", "off"), "diarization": extras.get("diarization", True)},
            enhancer=engines.audio_enhancer(models_dir),
            diarizer=engines.diarizer(models_dir),
            name_normalization=extras.get("name_normalization", True),
            snap_shots=bool(extras.get("snap_to_shots", False)),
            video_quality=str(extras.get("video_quality") or "best"),
            keep_cache=bool(extras.get("keep_job_cache", False)),
        )
    return factory


class JobRunner(QThread):
    progress = Signal(str, float, float, str)     # stage, stage fraction, overall fraction, message
    succeeded = Signal(object)                    # JobResult
    failed = Signal(str, str, object)             # technical message, UI key, UI args
    cancelled = Signal()

    def __init__(self, config: JobConfig, db_path: Path, pipeline_factory: PipelineFactory,
                 extras: dict | None = None, parent=None, job_id: int | None = None):
        super().__init__(parent)
        self.config = config
        self._extras = extras or {}
        self._db_path = db_path
        self._factory = pipeline_factory
        self._cancel = threading.Event()
        self.job_id = job_id

    def request_cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        # SQLite connections are per thread: open a dedicated one here.
        try:
            db = Database(self._db_path)
            repo = JobsRepo(db)
            if self.job_id is None:
                job_id = repo.create(
                    input_type="url" if self.config.source_url else "file",
                    input_value=self.config.source_url or str(self.config.input_path),
                    source_language=self.config.source_language, target_language=self.config.target_language,
                    mode=self.config.mode.value,
                    output_dir=str(self.config.output_dir) if self.config.output_dir else None,
                )
                self.job_id = job_id
            else:
                job_id = self.job_id
        except Exception as exc:          # database locked, disk full: report instead of stopping silently
            log.exception("Could not record the job")
            self.failed.emit(f"{type(exc).__name__}: {exc}", "error.pipeline_failed", {})
            return
        try:
            repo.update(job_id, status="running")
            pipeline = self._factory(
                self.config, self._cancel,
                lambda *args: self.progress.emit(*args),
                lambda stage, status, key, error: repo.record_stage(job_id, stage, status, key, error),
                self._extras,
            )
            try:
                result = pipeline.run()
            finally:
                if pipeline.job_key:
                    repo.update(job_id, input_hash=pipeline.job_key)
            repo.update(job_id, status="completed", output_dir=str(result.output_dir),
                        outputs=result.outputs, warnings=result.warnings, stats=result.stats)
            repo.set_series(job_id, result.series)
            self.succeeded.emit(result)
        except JobCancelled:
            repo.update(job_id, status="cancelled")
            self.cancelled.emit()
        except PipelineError as exc:
            repo.update(job_id, status="failed", error=str(exc))
            self.failed.emit(str(exc), exc.ui_key, exc.ui_args)
        except Exception as exc:
            log.exception("Job runner failed")
            repo.update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            self.failed.emit(f"{type(exc).__name__}: {exc}", "error.pipeline_failed", {})
        finally:
            db.close()
