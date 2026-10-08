"""Live local translation with the built-in llama.cpp runtime and TranslateGemma 4B (Windows; downloads
the llama.cpp CUDA build ~263 MB + ~2.5 GB model once; needs requirements-gpu.txt).

    .venv\\Scripts\\python.exe -m pytest tests/integration/test_real_local_llm.py -m integration -s
"""

import time

import pytest

from app.core.verification import check_line
from app.models import engines
from app.models.model_manager import DEFAULT_LOCAL_MODEL, ModelManager
from app.services.hardware_detection import EnginePlan
from app.utils.paths import AppPaths, default_data_root

pytestmark = pytest.mark.integration
# Real Turkish spelling (what Whisper produces); an earlier ASCII-only version made the input artificially hard.
LINES = ["Kamyon hazır mı abi?", "Bana bulamadım deme, Nevzat.", "Edebini tak, burası anaokulu değil.",
         "Saat 5'te limanda buluşalım.", "Helikopter mi? Bu iş çok riskli.", "Tamam abi, hallederim."]


def test_local_translation():
    paths = AppPaths.from_root(default_data_root()).ensure()
    started = time.monotonic()
    last = {"step": -1}

    def progress(fraction, text):
        # One line per 5 %, flushed: a carriage-return line would stay buffered and invisible.
        step = int(fraction * 20)
        if step != last["step"]:
            last["step"] = step
            print(f"  downloading: {text}", flush=True)

    print("\npreparing the local model (first run downloads ~263 MB runtime + ~2.5 GB model)...", flush=True)
    backend = engines.load_translation_backend(
        EnginePlan(DEFAULT_LOCAL_MODEL, "local", "cuda", "test"), ModelManager(paths.models_dir), 4, progress)
    print(f"\nready in {time.monotonic() - started:.1f} s on the {'GPU' if backend.client.gpu else 'CPU'}")
    try:
        started = time.monotonic()
        out = backend.translate(LINES, "tr", "ar")
        print(f"{len(LINES)} lines in {time.monotonic() - started:.1f} s")
    finally:
        backend.close()
    for src, tgt in zip(LINES, out):
        print(f"{src}\n  -> {tgt}  {check_line(src, tgt, 'ar') or ''}")
    assert all(out)
