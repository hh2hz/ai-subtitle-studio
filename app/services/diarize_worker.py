"""Speaker detection in a child process (D-120).

sherpa-onnx's `OfflineSpeakerDiarization.process` does not release Python's global interpreter lock while it runs
(its pybind11 binding has no `gil_scoped_release`, unlike the voice separation and the speaker embedding calls). One
call works on a whole 10-minute window, 80-100 s on the design laptop, and for that time no other Python thread of
the process can run: the window stopped repainting and Windows marked it "Not responding". In a child process the
lock it holds is its own.

The parent writes the audio to a .npy file next to the job's partial results and starts this module; the child
appends each finished window to the same partial file the in-process code used (so a cancelled or killed run
resumes), reports its progress through a small text file and writes the turns as JSON. The child runs at below
normal priority, so the window and the GPU job stay responsive.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from app.core.errors import JobCancelled

log = logging.getLogger(__name__)

ARG = "--diarize-worker"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
_POLL_S = 0.5


def _command(spec: Path) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, ARG, str(spec)]
    return [sys.executable, "-m", "app.services.diarize_worker", str(spec)]


def _files(partial: Path) -> dict[str, Path]:
    stem = partial.name.removesuffix(".partial.jsonl")
    return {name: partial.with_name(f"{stem}.worker.{name}")
            for name in ("audio.npy", "spec.json", "progress.txt", "result.json", "log.txt")}


def _read_progress(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):         # not written yet, or being rewritten right now
        return None


def run(audio: np.ndarray, models_dir: Path, windows, partial: Path, progress: Callable[[float], None] | None = None,
        cancel: threading.Event | None = None, threads: int | None = None,
        command: Callable[[Path], list[str]] = _command) -> list[tuple[float, float, int]]:
    """Run `app.core.diarize.diarize` in a child process and return its turns [(start_s, end_s, speaker_id)]."""
    files = _files(Path(partial))
    for name in ("progress.txt", "result.json"):
        files[name].unlink(missing_ok=True)
    tmp = files["audio.npy"].with_name(files["audio.npy"].name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, np.ascontiguousarray(audio, dtype=np.float32))
    os.replace(tmp, files["audio.npy"])
    spec = {"audio": str(files["audio.npy"]), "models_dir": str(models_dir), "partial": str(partial),
            "windows": [[float(a), float(b)] for a, b in windows], "threads": threads,
            "progress": str(files["progress.txt"]), "result": str(files["result.json"])}
    files["spec.json"].write_text(json.dumps(spec), encoding="utf-8")
    started = time.monotonic()
    try:
        with open(files["log.txt"], "wb") as errors:
            proc = subprocess.Popen(command(files["spec.json"]), cwd=_PROJECT_ROOT, stdin=subprocess.DEVNULL,
                                    stdout=errors, stderr=errors, creationflags=_FLAGS)
            try:
                while proc.poll() is None:
                    if cancel is not None and cancel.wait(_POLL_S):
                        raise JobCancelled()
                    if cancel is None:
                        time.sleep(_POLL_S)
                    fraction = _read_progress(files["progress.txt"])
                    if progress is not None and fraction is not None:
                        progress(fraction)
            except BaseException:
                proc.kill()
                proc.wait()
                raise
        try:
            result = json.loads(files["result.json"].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result = {}
        if "turns" in result and proc.returncode == 0:
            log.info("Speaker detection process finished in %.1f s", time.monotonic() - started)
            files["log.txt"].unlink(missing_ok=True)
            return [(float(a), float(b), int(c)) for a, b, c in result["turns"]]
        try:
            tail = files["log.txt"].read_text(encoding="utf-8", errors="replace")[-1500:].strip()
        except OSError:
            tail = ""
        raise RuntimeError(result.get("error") or f"speaker detection process exited with code {proc.returncode}"
                           + (f": {tail}" if tail else ""))
    finally:
        for name in ("audio.npy", "spec.json", "progress.txt", "result.json"):
            files[name].unlink(missing_ok=True)


def main(spec_path: str) -> int:
    """Child-process entry point."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    result_path = Path(spec["result"])
    progress_path = Path(spec["progress"])

    def progress(fraction: float) -> None:
        try:
            progress_path.write_text(f"{fraction:.4f}", encoding="utf-8")
        except OSError:                   # the parent may be reading it; the next window writes again
            pass

    try:
        from app.core import diarize

        audio = np.load(spec["audio"], mmap_mode="r")
        turns = diarize.diarize(audio, Path(spec["models_dir"]), [tuple(w) for w in spec["windows"]],
                                Path(spec["partial"]), progress=progress, threads=spec.get("threads"))
        payload, code = {"turns": [list(t) for t in turns]}, 0
    except Exception as exc:              # noqa: BLE001 - reported to the parent, which keeps the job going
        log.exception("Speaker detection failed")
        payload, code = {"error": f"{type(exc).__name__}: {exc}"}, 1
    tmp = result_path.with_name(result_path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, result_path)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
