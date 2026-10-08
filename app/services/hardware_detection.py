"""Hardware detection and model/device recommendations."""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.core.modes import Mode
from app.utils.cuda_setup import register_nvidia_dll_dirs

log = logging.getLogger(__name__)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class GpuInfo:
    name: str
    vram_mb: int | None


@dataclass
class HardwareInfo:
    os: str
    cpu_name: str
    logical_cores: int
    ram_mb: int | None
    disk_free_mb: int | None
    gpus: list[GpuInfo] = field(default_factory=list)
    cuda_device_count: int = 0
    cuda_compute_types: list[str] = field(default_factory=list)
    cuda_error: str | None = None

    @property
    def max_vram_mb(self) -> int:
        return max((g.vram_mb or 0 for g in self.gpus), default=0)

    @property
    def cuda_usable(self) -> bool:
        return self.cuda_device_count > 0 and bool(self.cuda_compute_types)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EnginePlan:
    """One way to run a model. Plans are tried in order until one loads."""

    model: str
    device: str          # "cuda" or "cpu"
    compute_type: str
    reason: str


# -- detection ---------------------------------------------------------------------------------

def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_mb() -> int | None:
    if sys.platform == "win32":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys // (1024 * 1024))
        return None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


def _nvidia_smi_gpus() -> list[GpuInfo]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, creationflags=_NO_WINDOW, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("nvidia-smi failed: %s", exc)
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, _, vram = (part.strip() for part in line.rpartition(","))
        try:
            gpus.append(GpuInfo(name=name, vram_mb=int(float(vram))))
        except ValueError:
            gpus.append(GpuInfo(name=line.strip(), vram_mb=None))
    return gpus


def detect(data_dir: Path) -> HardwareInfo:
    """Collect hardware facts. Never raises; unknown values stay None/empty."""
    try:
        disk_free = shutil.disk_usage(data_dir).free // (1024 * 1024)
    except OSError:
        disk_free = None
    info = HardwareInfo(
        os=platform.platform(),
        cpu_name=_cpu_name(),
        logical_cores=os.cpu_count() or 1,
        ram_mb=_ram_mb(),
        disk_free_mb=disk_free,
        gpus=_nvidia_smi_gpus(),
    )
    register_nvidia_dll_dirs()
    try:
        import ctranslate2

        info.cuda_device_count = ctranslate2.get_cuda_device_count()
        if info.cuda_device_count:
            info.cuda_compute_types = sorted(ctranslate2.get_supported_compute_types("cuda"))
    except Exception as exc:  # CUDA probing must never break startup.
        info.cuda_error = f"{type(exc).__name__}: {exc}"
    log.info("Hardware: %s", info.to_dict())
    return info


# -- recommendations ---------------------------------------------------------------------------

# VRAM thresholds include headroom for the desktop and CUDA context. Model memory figures are
# estimates derived from the faster-whisper README benchmark (large-v2: fp16 ~4.5 GB, int8 ~2.9 GB).
_VRAM_FP16_LARGE_MB = 6000
_VRAM_INT8_LARGE_MB = 3500
_VRAM_INT8_MEDIUM_MB = 2000
_VRAM_MT_GPU_MB = 6000


def _gpu_compute_type(hw: HardwareInfo, prefer_fp16: bool) -> str:
    types = set(hw.cuda_compute_types)
    if prefer_fp16 and "float16" in types:
        return "float16"
    if "int8_float16" in types:
        return "int8_float16"
    return "int8"


def recommend_asr(hw: HardwareInfo, mode: Mode) -> list[EnginePlan]:
    """Ordered ASR plans: the preferred one first, safer fallbacks after it."""
    plans: list[EnginePlan] = []
    vram = hw.max_vram_mb
    if hw.cuda_usable and vram >= _VRAM_INT8_MEDIUM_MB:
        if vram >= _VRAM_FP16_LARGE_MB:
            ct = _gpu_compute_type(hw, prefer_fp16=True)
            model = "large-v3-turbo" if mode == Mode.FAST else "large-v3"
            plans.append(EnginePlan(model, "cuda", ct, f"GPU with {vram} MB VRAM"))
        elif vram >= _VRAM_INT8_LARGE_MB:
            ct = _gpu_compute_type(hw, prefer_fp16=False)
            model = "large-v3" if mode == Mode.MAXIMUM_ACCURACY else "large-v3-turbo"
            plans.append(EnginePlan(model, "cuda", ct, f"GPU with {vram} MB VRAM, 8-bit weights"))
            if model == "large-v3":
                plans.append(EnginePlan("large-v3-turbo", "cuda", ct, "fallback: smaller model on GPU"))
        else:
            ct = _gpu_compute_type(hw, prefer_fp16=False)
            plans.append(EnginePlan("medium", "cuda", ct, f"GPU with only {vram} MB VRAM"))
    ram = hw.ram_mb or 0
    if ram >= 12000:
        cpu_model = {Mode.FAST: "small", Mode.BALANCED: "medium", Mode.MAXIMUM_ACCURACY: "large-v3"}[mode]
    elif ram >= 6000:
        cpu_model = {Mode.FAST: "small", Mode.BALANCED: "small", Mode.MAXIMUM_ACCURACY: "medium"}[mode]
    else:
        cpu_model = "base" if mode == Mode.FAST else "small"
    plans.append(EnginePlan(cpu_model, "cpu", "int8", f"CPU with {ram or 'unknown'} MB RAM"))
    return plans


def recommend_translation(hw: HardwareInfo, model_key: str) -> list[EnginePlan]:
    """Ordered translation plans.

    T5-family models (MADLAD-400) are prone to float16 overflow, so only int8 (float32 compute)
    is used. The int8 3B model has ~3 GB of weights alone, so a 4 GB card (minus desktop usage)
    is not tried; the GPU is used only from _VRAM_MT_GPU_MB upwards.
    """
    plans = []
    if hw.cuda_usable and "int8" in hw.cuda_compute_types and hw.max_vram_mb >= _VRAM_MT_GPU_MB:
        plans.append(EnginePlan(model_key, "cuda", "int8", f"GPU with {hw.max_vram_mb} MB VRAM"))
    plans.append(EnginePlan(model_key, "cpu", "int8", "CPU"))
    return plans
