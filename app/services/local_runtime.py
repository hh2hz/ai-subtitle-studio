"""Built-in local LLM runtime: the official llama.cpp server, downloaded and started by the app.

Nothing has to be installed by the user. On first use the app downloads (1) the official llama.cpp Windows
build - the CUDA build when an NVIDIA GPU is usable (it reuses the pip-installed CUDA libraries of
requirements-gpu.txt), otherwise the CPU build - and (2) the GGUF model file, then runs llama-server.exe on
127.0.0.1 only. If the GPU cannot be used, the server is restarted on the CPU. See DECISIONS D-030 and D-032.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core.local_llm import LocalModelError
from app.utils.cuda_setup import nvidia_bin_dirs

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
COMPLETE_MARKER = ".runtime-complete"


@dataclass(frozen=True)
class RuntimeBuild:
    tag: str
    asset: str
    sha256: str
    size: int
    variant: str = "cpu"        # "cuda" or "cpu"

    @property
    def url(self) -> str:
        return f"https://github.com/ggml-org/llama.cpp/releases/download/{self.tag}/{self.asset}"

    @property
    def folder(self) -> str:
        return f"{self.tag}-{self.variant}"


# Pinned official release b11368; sha256 computed from the downloaded release assets (2026-10-03).
WINDOWS_CUDA = RuntimeBuild(
    "b11368", "llama-b11368-bin-win-cuda-12.4-x64.zip",
    "c38e19d812429334875a3792f4e0c171d3e3958f476276bcace2423c6058ab55", 263352051, "cuda")
WINDOWS_CPU = RuntimeBuild(
    "b11368", "llama-b11368-bin-win-cpu-x64.zip",
    "8d5548f8ef5dbaecbc818b83261bf59fbf4fe4ab2aba47b06721e5d1356f6c0d", 19333447, "cpu")
_WINDOWS_BUILDS = (WINDOWS_CUDA, WINDOWS_CPU)
# Vulkan build used before D-032: its device enumeration hangs on the user's machine, so it is removed.
_LEGACY_FOLDERS = ("b11368",)
# ggml-cuda.dll of the CUDA 12.4 build imports these (cublasLt through cublas); they come from the pip packages
# nvidia-cuda-runtime-cu12 and nvidia-cublas-cu12 instead of the separate 391 MB cudart archive.
CUDA_DLLS = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")


def missing_cuda_dlls(dirs: list[Path]) -> list[str]:
    return [name for name in CUDA_DLLS if not any((d / name).is_file() for d in dirs)]


def server_executable_name() -> str:
    return "llama-server.exe" if sys.platform == "win32" else "llama-server"


def _download(url: str, target: Path, expected_size: int | None, progress: Progress | None) -> None:
    tmp = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
            total = expected_size or int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(min(done / total, 0.99), f"{done >> 20} / {total >> 20} MB")
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise LocalModelError(f"Download failed: {url}: {exc}") from exc
    tmp.replace(target)


MSVC_RUNTIME_DLLS = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_codecvt_ids.dll",
                     "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll")
_STATUS_DLL_NOT_FOUND = 0xC0000135


def _msvc_sources() -> list[Path]:
    """Folders that ship the Microsoft Visual C++ runtime with this application (PySide6 wheel, frozen bundle)."""
    sources = []
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        sources.append(Path(frozen))
    for package in ("PySide6", "shiboken6"):
        try:
            module = __import__(package)
            sources.append(Path(module.__file__).parent)
        except ImportError:
            continue
    sources.append(Path(sys.executable).parent)
    return sources


def provide_msvc_runtime(target: Path) -> list[str]:
    """llama.cpp's Windows build needs the Visual C++ 2015-2022 runtime (MSVCP140.dll, VCRUNTIME140_1.dll).
    Machines without the Microsoft redistributable cannot start it, so the copies that ship with the application
    are placed next to llama-server.exe (the DLL search order prefers the executable's folder)."""
    copied = []
    if sys.platform != "win32":
        return copied
    for name in MSVC_RUNTIME_DLLS:
        if (target / name).is_file():
            continue
        for source in _msvc_sources():
            candidate = source / name
            if candidate.is_file():
                shutil.copyfile(candidate, target / name)
                copied.append(name)
                break
    if copied:
        log.info("Placed Visual C++ runtime next to llama-server: %s", ", ".join(copied))
    return copied


def ensure_runtime(base_dir: Path, build: RuntimeBuild = WINDOWS_CPU, progress: Progress | None = None) -> Path:
    """Return the path of llama-server, downloading and verifying the pinned build on first use."""
    root = Path(base_dir) / "llama.cpp"
    target = root / build.folder
    exe = target / server_executable_name()
    if sys.platform != "win32" and build in _WINDOWS_BUILDS:
        raise LocalModelError("The built-in local model runtime is provided for Windows only")
    if build.variant == "cuda":
        missing = missing_cuda_dlls(nvidia_bin_dirs())
        if missing:
            raise LocalModelError("CUDA libraries for the GPU runtime are missing (" + ", ".join(missing)
                                  + "); install requirements-gpu.txt")
    for name in _LEGACY_FOLDERS:
        if (root / name).is_dir():
            shutil.rmtree(root / name, ignore_errors=True)
    if (target / COMPLETE_MARKER).is_file() and exe.is_file():
        provide_msvc_runtime(target)
        return exe
    target.mkdir(parents=True, exist_ok=True)
    archive = target / build.asset
    log.info("Downloading the local AI runtime %s (%d MB)", build.asset, build.size >> 20)
    _download(build.url, archive, build.size, progress)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != build.sha256:
        archive.unlink(missing_ok=True)
        raise LocalModelError(f"Runtime archive checksum mismatch ({digest})")
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name          # flatten; never extract outside target
            if member.is_dir() or not name:
                continue
            with zf.open(member) as src, open(target / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    archive.unlink(missing_ok=True)
    if not exe.is_file():
        raise LocalModelError(f"{exe.name} not found in {build.asset}")
    provide_msvc_runtime(target)
    (target / COMPLETE_MARKER).write_text(build.sha256, encoding="utf-8")
    return exe


def _suppress_error_dialogs() -> None:
    """A missing DLL makes Windows show a modal error box that blocks the child process forever (nothing is
    logged and the start just times out). With these error-mode flags, inherited by the child, it exits at once
    with STATUS_DLL_NOT_FOUND instead."""
    if sys.platform == "win32":
        import ctypes

        sem_failcriticalerrors, sem_nogpfaulterrorbox, sem_noopenfileerrorbox = 0x0001, 0x0002, 0x8000
        ctypes.windll.kernel32.SetErrorMode(sem_failcriticalerrors | sem_nogpfaulterrorbox | sem_noopenfileerrorbox)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


GEMMA_TURN = "<start_of_turn>{role}\n{content}<end_of_turn>\n"


class LlamaServer:
    """Runs llama-server for one model; implements the CompletionClient protocol used by the backend."""

    GPU_START_TIMEOUT_S = 180
    CPU_START_TIMEOUT_S = 300

    def __init__(self, executable: Path, model_path: Path, gpu: bool = False, ctx_size: int = 4096,
                 log_path: Path | None = None, extra_args: list[str] | None = None,
                 dll_dirs: list[Path] | None = None, mmproj: Path | None = None, jinja: bool = False):
        self.executable = Path(executable)
        # mmproj: multimodal projector (image/audio models); jinja: use the model's own chat template.
        self.mmproj = Path(mmproj) if mmproj else None
        self.jinja = jinja
        self.model_path = Path(model_path)
        self.gpu = gpu
        self.dll_dirs = list(dll_dirs or [])
        self.ctx_size = ctx_size
        self.log_path = log_path or self.executable.parent / "server.log"
        self.extra_args = extra_args or []
        self.port: int | None = None
        self._process: subprocess.Popen | None = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _command(self) -> list[str]:
        # --no-jinja (default): the TranslateGemma GGUF chat template cannot be parsed by llama.cpp and aborts the start;
        # it is not needed because requests use /completion with the Gemma turn format.
        cmd = [str(self.executable), "-m", str(self.model_path), "--host", "127.0.0.1", "--port", str(self.port),
               "-c", str(self.ctx_size), "-np", "1"]
        cmd += ["--jinja"] if self.jinja else ["--no-jinja"]
        if self.mmproj is not None:
            cmd += ["--mmproj", str(self.mmproj)] + ([] if self.gpu else ["--no-mmproj-offload"])
        # GPU: the layer count is left unset so "--fit on" places as many layers as the free VRAM allows.
        cmd += ["--fit", "on"] if self.gpu else ["-ngl", "0", "--device", "none"]
        if self.executable.suffix == ".py":
            cmd = [sys.executable] + cmd
        return cmd + self.extra_args

    def start(self, timeout_s: float | None = None) -> None:
        """Start the server; a failed GPU start is retried once on the CPU."""
        try:
            self._start_once(timeout_s or (self.GPU_START_TIMEOUT_S if self.gpu else self.CPU_START_TIMEOUT_S))
        except LocalModelError as exc:
            if not self.gpu:
                raise
            log.warning("Local model did not start on the GPU (%s); retrying on the CPU", exc)
            self.gpu = False
            self._start_once(timeout_s or self.CPU_START_TIMEOUT_S)

    def _environment(self) -> dict:
        env = dict(os.environ)
        if self.dll_dirs:
            env["PATH"] = os.pathsep.join([str(d) for d in self.dll_dirs] + [env.get("PATH", "")])
        # GPUs without native kernels in the build (e.g. Turing) JIT-compile them on first use; the driver's
        # default 256 MB compute cache may be too small to keep them, so it is raised to the 4 GB maximum.
        env.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024 ** 3))
        return env

    def _start_once(self, timeout_s: float) -> None:
        self.stop()
        self.port = _free_port()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "ab")
        log.info("Starting local model server: %s", " ".join(self._command()))
        _suppress_error_dialogs()
        self._process = subprocess.Popen(self._command(), stdout=self._log_file, stderr=subprocess.STDOUT,
                                         cwd=str(self.executable.parent), env=self._environment(),
                                         creationflags=_NO_WINDOW)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                code = self._process.returncode
                self.stop()
                if code & 0xFFFFFFFF == _STATUS_DLL_NOT_FOUND:
                    raise LocalModelError("llama-server could not start: a required DLL is missing "
                                          "(Microsoft Visual C++ runtime or CUDA libraries)")
                raise LocalModelError(f"llama-server exited with code {code}; see {self.log_path}")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=2) as response:
                    if json.loads(response.read()).get("status") == "ok":
                        log.info("Local model server ready on port %d (%s)", self.port, "GPU" if self.gpu else "CPU")
                        return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.5)
        self.stop()
        raise LocalModelError(f"llama-server did not become ready within {timeout_s:.0f} s")

    def complete(self, prompt: str, n_predict: int = 1024, timeout_s: float = 600) -> str:
        if self._process is None or self._process.poll() is not None:
            raise LocalModelError("The local model server is not running")
        body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": True,
                           "stop": ["<end_of_turn>", "<start_of_turn>"]}).encode("utf-8")
        request = urllib.request.Request(self.base_url + "/completion", data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read()).get("content", "")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LocalModelError(f"Local model request failed: {exc}") from exc

    # CompletionClient protocol: Gemma chat format applied directly (avoids chat-template differences).
    def chat(self, model: str, messages: list[dict], fmt: dict | None = None, num_ctx: int = 4096) -> str:
        prompt = "".join(GEMMA_TURN.format(role="user" if m["role"] != "assistant" else "model", content=m["content"])
                         for m in messages)
        return self.complete(prompt + "<start_of_turn>model\n").strip()

    def unload(self, model: str) -> None:
        self.stop()

    def stop(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            self._process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
