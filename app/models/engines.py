"""Production factories that turn an EnginePlan into a loaded engine (downloading if needed)."""

from __future__ import annotations

import logging
from typing import Callable

from app.core.errors import PipelineError
from app.core.local_llm import LocalModelError, LocalTranslationBackend
from app.models.model_manager import LOCAL_LLM_FILES
from app.services import local_runtime
from app.core.transcription import AsrEngine, AsrOptions, FasterWhisperEngine
from app.core.translation import MadladBackend, TranslationBackend
from app.models.model_manager import ModelManager
from app.services import gpu_probe
from app.services.hardware_detection import EnginePlan
from app.utils.cuda_setup import nvidia_bin_dirs, register_nvidia_dll_dirs

log = logging.getLogger(__name__)

DownloadProgress = Callable[[float, str], None]


def _check_gpu(kind: str, plan: EnginePlan, model_dir) -> None:
    if plan.device != "cuda":
        return
    ok, detail = gpu_probe.probe(kind, model_dir, plan.compute_type)
    if not ok:
        raise PipelineError(f"GPU check failed for {plan.model} ({plan.compute_type}): {detail}")


def load_asr_engine(plan: EnginePlan, models: ModelManager, options: AsrOptions,
                    progress: DownloadProgress | None = None) -> AsrEngine:
    register_nvidia_dll_dirs()
    model_dir = models.ensure("whisper", plan.model, progress)
    _check_gpu("whisper", plan, model_dir)
    return FasterWhisperEngine(model_dir, plan.device, plan.compute_type, options)


def audio_enhancer(models_dir):
    """(audio, progress, cancel) -> voice-only, levelled audio; the separation model is downloaded on first use."""
    import os

    from app.core import enhance

    def run(audio, progress=None, cancel=None):
        model_dir = enhance.ensure_model(models_dir)
        threads = max(1, min(4, (os.cpu_count() or 2) // 2))
        voice = enhance.separate_speech(audio, model_dir, threads=threads, progress=progress, cancel=cancel)
        return enhance.level_speech(voice)
    return run


def diarizer(models_dir):
    """(audio, windows, partial, progress, cancel) -> [(start, end, speaker)]; models are downloaded on first use."""
    from app.core import diarize

    def run(audio, windows, partial, progress=None, cancel=None):
        from app.services import diarize_worker

        diarize.ensure_models(models_dir)
        # In a child process: the sherpa-onnx call holds Python's GIL for minutes and froze the window (D-120).
        return diarize_worker.run(audio, models_dir, windows, partial, progress=progress, cancel=cancel)
    return run


def load_translation_backend(plan: EnginePlan, models: ModelManager, beam_size: int,
                             progress: DownloadProgress | None = None,
                             runtime_builder=local_runtime.ensure_runtime) -> TranslationBackend:
    if plan.device == "local":
        if plan.model not in LOCAL_LLM_FILES:
            raise LocalModelError(f"Unknown local model {plan.model}")
        build = local_runtime.WINDOWS_CUDA if plan.compute_type == "cuda" else local_runtime.WINDOWS_CPU
        exe = runtime_builder(models.models_dir, build, progress=progress)
        model_dir = models.ensure("translation", plan.model, progress)
        gpu = build.variant == "cuda"
        server = local_runtime.LlamaServer(exe, model_dir / LOCAL_LLM_FILES[plan.model], gpu=gpu,
                                           log_path=model_dir / "server.log",
                                           dll_dirs=nvidia_bin_dirs() if gpu else None)
        server.start()
        return LocalTranslationBackend(server, plan.model)
    register_nvidia_dll_dirs()
    model_dir = models.ensure("translation", plan.model, progress)
    _check_gpu("translation", plan, model_dir)
    return MadladBackend(model_dir, plan.device, plan.compute_type, beam_size=beam_size)
