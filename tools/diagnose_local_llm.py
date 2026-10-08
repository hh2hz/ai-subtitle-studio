"""Diagnose the built-in llama.cpp server (CPU and CUDA builds). Writes tools/diagnose_report.txt.

    .venv\\Scripts\\python.exe tools\\diagnose_local_llm.py
Every step has a time limit, so the script always finishes.
"""

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services import local_runtime  # noqa: E402
from app.utils.cuda_setup import nvidia_bin_dirs  # noqa: E402
from app.utils.paths import AppPaths, default_data_root  # noqa: E402

REPORT = ROOT / "tools" / "diagnose_report.txt"
lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def run(title: str, cmd: list[str], cwd: Path, timeout: int = 40, env: dict | None = None) -> None:
    say(f"\n=== {title}\n$ {' '.join(cmd)}")
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                              errors="replace", env={**os.environ, **(env or {})})
        say(f"exit code {proc.returncode} (0x{proc.returncode & 0xFFFFFFFF:08X}) after {time.monotonic() - started:.1f} s")
        output = (proc.stdout + proc.stderr).strip().splitlines()
        for line in output[:60] + (["..."] + output[-25:] if len(output) > 85 else output[60:]):
            say("  " + line)
        if not output:
            say("  (no output at all)")
    except subprocess.TimeoutExpired as exc:
        say(f"TIMED OUT after {timeout} s (process hung)")
        def text(value):
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else (value or "")
        partial = (text(exc.stdout) + text(exc.stderr)).strip().splitlines()
        for line in partial[:40]:
            say("  " + line)
        if not partial:
            say("  (no output before hanging)")
    except OSError as exc:
        say(f"COULD NOT START: {exc} (winerror={getattr(exc, 'winerror', None)})")


def main() -> None:
    say(f"Windows: {platform.platform()} | Python {platform.python_version()}")
    models = AppPaths.from_root(default_data_root()).models_dir
    dll_dirs = nvidia_bin_dirs()
    say("NVIDIA DLL folders: " + (", ".join(map(str, dll_dirs)) or "none"))
    missing = local_runtime.missing_cuda_dlls(dll_dirs)
    say(f"CUDA DLLs missing: {missing or 'none'}")
    gguf = next((models / "translation").glob("*/*.gguf"), None)
    say(f"model: {gguf} size={gguf.stat().st_size if gguf else None}")
    env = {"PATH": os.pathsep.join([str(d) for d in dll_dirs] + [os.environ.get("PATH", "")])}
    local_runtime._suppress_error_dialogs()
    builds = [local_runtime.WINDOWS_CPU] + ([] if missing else [local_runtime.WINDOWS_CUDA])
    for build in builds:
        say(f"\n##### {build.asset}")
        try:
            exe = local_runtime.ensure_runtime(models, build,
                                               lambda f, t: print(f"  downloading {t}", end="\r", flush=True))
        except Exception as exc:      # noqa: BLE001 - report every failure
            say(f"runtime not available: {exc}")
            continue
        runtime = exe.parent
        run("version", [str(exe), "--version"], runtime, env=env)
        run("list devices", [str(exe), "--list-devices"], runtime, timeout=60, env=env)
        if not gguf:
            continue
        common = [str(exe), "-m", str(gguf), "-c", "2048", "-np", "1", "--no-jinja"]
        if build.variant == "cuda":
            run("load model on GPU (stops after 120 s)", common + ["--fit", "on", "--port", "8098"], runtime,
                timeout=120, env=env)
        else:
            run("load model on CPU (stops after 60 s)", common + ["-ngl", "0", "--device", "none", "--port", "8097"],
                runtime, timeout=60, env=env)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    say(f"\nReport written to {REPORT}")


if __name__ == "__main__":
    main()
