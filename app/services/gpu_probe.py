"""Out-of-process GPU check.

A missing or mismatched cuBLAS/cuDNN DLL can terminate the whole process from native code instead
of raising a Python exception. Loading the model once in a child process first means such a failure
only costs the probe, and the job falls back to the next plan.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

PROBE_OK = "GPU_PROBE_OK"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_cache: dict[tuple[str, str, str], tuple[bool, str]] = {}


def _command(kind: str, model_dir: str, compute_type: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--gpu-probe", kind, model_dir, compute_type]
    return [sys.executable, "-m", "app.services.gpu_probe", kind, model_dir, compute_type]


def probe(kind: str, model_dir: Path, compute_type: str, timeout_s: int = 600) -> tuple[bool, str]:
    """Return (ok, detail). Results are cached for the lifetime of the process."""
    key = (kind, str(model_dir), compute_type)
    if key in _cache:
        return _cache[key]
    try:
        proc = subprocess.run(
            _command(kind, str(model_dir), compute_type), cwd=_PROJECT_ROOT, capture_output=True,
            text=True, timeout=timeout_s, creationflags=_NO_WINDOW,
        )
        ok = proc.returncode == 0 and PROBE_OK in proc.stdout
        detail = (proc.stderr or proc.stdout)[-2000:].strip() or f"exit code {proc.returncode}"
    except (OSError, subprocess.SubprocessError) as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    log.log(logging.INFO if ok else logging.WARNING, "GPU probe %s %s %s: %s",
            kind, Path(model_dir).name, compute_type, "ok" if ok else detail)
    _cache[key] = (ok, "" if ok else detail)
    return _cache[key]


def run_probe(kind: str, model_dir: str, compute_type: str) -> int:
    """Child-process entry point: load the model on CUDA and run one tiny inference."""
    from app.utils.cuda_setup import register_nvidia_dll_dirs

    register_nvidia_dll_dirs()
    if kind == "whisper":
        import numpy as np
        from faster_whisper import WhisperModel

        model = WhisperModel(model_dir, device="cuda", compute_type=compute_type)
        segments, _ = model.transcribe(np.zeros(32000, dtype=np.float32), language="en", vad_filter=False)
        list(segments)
    elif kind == "translation":
        import ctranslate2

        translator = ctranslate2.Translator(model_dir, device="cuda", compute_type=compute_type)
        translator.translate_batch([["</s>"]], max_decoding_length=2)
    else:
        print(f"unknown probe kind {kind}", file=sys.stderr)
        return 2
    print(PROBE_OK)
    return 0


if __name__ == "__main__":
    sys.exit(run_probe(*sys.argv[1:4]))
