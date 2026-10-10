"""Live check of D-120: the real sherpa-onnx speaker detection holds Python's GIL for a whole window when it runs in the
application process, and does not when it runs in the child process. Downloads the two models (about 35 MB) on first
use into a temporary folder.

    .venv\\Scripts\\python.exe -m pytest tests/integration/test_real_diarize_worker.py -m integration -s
"""

import threading
import time

import numpy as np
import pytest

from app.core import diarize
from app.services import diarize_worker

pytestmark = pytest.mark.integration


def _signal(seconds: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(16000 * seconds) / 16000
    voice = 0.2 * np.sin(2 * np.pi * (150 + 50 * np.sin(t / 3)) * t) * ((t % 7) < 4)
    return (voice + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def _longest_stall(work) -> tuple[float, object]:
    gaps, stop = [], threading.Event()

    def ticker():
        last = time.monotonic()
        while not stop.is_set():
            time.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    thread = threading.Thread(target=ticker, daemon=True)
    thread.start()
    try:
        result = work()
    finally:
        stop.set()
        thread.join(1)
    return max(gaps), result


def test_worker_keeps_the_process_responsive(tmp_path_factory):
    models = tmp_path_factory.mktemp("models")
    diarize.ensure_models(models)
    audio = _signal(300)
    windows = [(0.0, 300.0)]
    work = tmp_path_factory.mktemp("work")

    stall_in, turns_in = _longest_stall(lambda: diarize.diarize(audio, models, windows, work / "a.partial.jsonl",
                                                                threads=2))
    stall_out, turns_out = _longest_stall(lambda: diarize_worker.run(audio, models, windows, work / "b.partial.jsonl",
                                                                     cancel=threading.Event(), threads=2))
    print(f"longest stall in process {stall_in:.2f} s, with the worker {stall_out:.2f} s")
    assert turns_in == turns_out
    assert stall_out < 0.5 < stall_in
