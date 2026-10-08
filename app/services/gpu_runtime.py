"""NVIDIA GPU libraries downloaded on first use (user decision D-043: keep the installer small).

faster-whisper (CTranslate2) needs cuBLAS and cuDNN, the llama.cpp CUDA build needs the CUDA runtime. The
installer does not contain them (about 1.2 GB); on a PC with an NVIDIA GPU the app offers to download the same
pinned wheels as requirements-gpu.txt from PyPI, verifies each against the SHA-256 that PyPI publishes, and
extracts only their DLLs into <data>/cuda/nvidia/<lib>/bin, which app.utils.cuda_setup adds to the DLL search
path. From source with requirements-gpu.txt installed nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from app.core.errors import JobCancelled
from app.utils import cuda_setup
from app.utils.paths import default_data_root

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]
# Same pins as requirements-gpu.txt.
PACKAGES = (("nvidia-cublas-cu12", "12.9.2.10"), ("nvidia-cudnn-cu12", "9.27.0.42"),
            ("nvidia-cuda-runtime-cu12", "12.9.79"))
REQUIRED_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll", "cudart64_12.dll")
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"
MARKER = ".complete"


def install_dir() -> Path:
    return default_data_root() / "cuda"


def missing_dlls() -> list[str]:
    dirs = cuda_setup.nvidia_bin_dirs()
    return [name for name in REQUIRED_DLLS if not any((d / name).is_file() for d in dirs)]


def should_offer(gpus) -> bool:
    """An NVIDIA GPU is present (nvidia-smi) but the GPU libraries are not available."""
    return sys.platform == "win32" and bool(gpus) and bool(missing_dlls())


def _wheel_info(name: str, version: str) -> dict:
    with urllib.request.urlopen(PYPI_JSON.format(name=name, version=version), timeout=60) as response:
        data = json.loads(response.read())
    for item in data.get("urls", []):
        if item.get("packagetype") == "bdist_wheel" and item.get("filename", "").endswith("win_amd64.whl"):
            return {"url": item["url"], "sha256": item["digests"]["sha256"], "size": item.get("size") or 0,
                    "filename": item["filename"]}
    raise RuntimeError(f"No Windows wheel for {name} {version} on PyPI")


def download_all(progress: Progress | None = None, cancel: threading.Event | None = None,
                 target: Path | None = None) -> Path:
    """Download, verify and extract all packages; returns the install directory."""
    target = target or install_dir()
    target.mkdir(parents=True, exist_ok=True)
    infos = [(name, _wheel_info(name, version)) for name, version in PACKAGES]
    total = sum(info["size"] for _, info in infos) or 1
    done = 0
    for name, info in infos:
        marker = target / f"{name}{MARKER}"
        if marker.is_file() and marker.read_text(encoding="utf-8") == info["sha256"]:
            done += info["size"]
            continue
        wheel = target / (info["filename"] + ".part")
        digest = hashlib.sha256()
        log.info("Downloading %s (%d MB)", info["filename"], info["size"] >> 20)
        with urllib.request.urlopen(info["url"], timeout=120) as response, open(wheel, "wb") as out:
            while chunk := response.read(1 << 20):
                if cancel is not None and cancel.is_set():
                    out.close()
                    wheel.unlink(missing_ok=True)
                    raise JobCancelled()
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(min(done / total, 0.99), f"{done >> 20} / {total >> 20} MB")
        if digest.hexdigest() != info["sha256"]:
            wheel.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {info['filename']}")
        _extract_dlls(wheel, target)
        wheel.unlink(missing_ok=True)
        marker.write_text(info["sha256"], encoding="utf-8")
    cuda_setup.refresh()
    if progress:
        progress(1.0, "")
    return target


def _extract_dlls(wheel: Path, target: Path) -> None:
    """Only nvidia/<lib>/bin/*.dll; paths are rebuilt from their parts so nothing lands outside `target`."""
    with zipfile.ZipFile(wheel) as zf:
        for member in zf.infolist():
            parts = member.filename.split("/")
            if (len(parts) == 4 and parts[0] == "nvidia" and parts[2] == "bin" and parts[3].lower().endswith(".dll")
                    and ".." not in parts):
                dest = target / "nvidia" / parts[1] / "bin" / parts[3]
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
