"""Make pip-installed NVIDIA runtime DLLs visible to CTranslate2 on Windows.

The optional packages nvidia-cublas-cu12 and nvidia-cudnn-cu12 install DLLs under
site-packages/nvidia/<lib>/bin, which is not on the DLL search path by default.
Must run before ctranslate2 is imported.
"""

from __future__ import annotations

import logging
import os
import site
import sys
from pathlib import Path

from app.utils.paths import default_data_root

log = logging.getLogger(__name__)
_done = False
_handles: list = []


def _candidate_roots() -> list[Path]:
    roots = [Path(p) for p in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        roots.append(Path(frozen))
    # GPU libraries downloaded by the installed app (app.services.gpu_runtime).
    roots.append(default_data_root() / "cuda")
    return roots


def nvidia_bin_dirs() -> list[Path]:
    """All site-packages/nvidia/<lib>/bin folders (pip-installed NVIDIA runtime DLLs)."""
    dirs: list[Path] = []
    for root in _candidate_roots():
        nvidia = root / "nvidia"
        if nvidia.is_dir():
            dirs.extend(sorted(nvidia.glob("*/bin")))
    return dirs


_registered: set[str] = set()


def refresh() -> list[Path]:
    """Register folders that appeared after startup (GPU libraries downloaded while the app runs)."""
    global _done
    _done = False
    return register_nvidia_dll_dirs()


def register_nvidia_dll_dirs() -> list[Path]:
    """Add nvidia/*/bin directories to the DLL search path. No-op outside Windows."""
    global _done
    if _done or sys.platform != "win32":
        return []
    _done = True
    added: list[Path] = []
    for bin_dir in nvidia_bin_dirs():
        if str(bin_dir) in _registered:
            continue
        _registered.add(str(bin_dir))
        try:
            _handles.append(os.add_dll_directory(str(bin_dir)))
        except OSError as exc:
            log.warning("Could not add DLL directory %s: %s", bin_dir, exc)
            continue
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        added.append(bin_dir)
    if added:
        log.info("Registered NVIDIA DLL directories: %s", ", ".join(map(str, added)))
    return added
