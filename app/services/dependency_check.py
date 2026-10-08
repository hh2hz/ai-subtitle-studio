"""Automatic library check and repair before the app starts (D-114, supersedes D-113).

The user never types a command:

* pinned libraries (`requirements.txt`) that are missing or differ from the pin are re-installed with pip (only when
  the app runs from its sources - a frozen build cannot pip);
* `yt-dlp` and `yt-dlp-ejs` are the only floating libraries: a newer release is downloaded as a verified wheel into
  the data folder (`app/services/overlay.py`), which also works in the installed build;
* anything that is still wrong afterwards is returned as a plain-language error for the dialog to show.

A failed network lookup is never an error by itself: the installed libraries keep working.

Everything here is offline-testable: pip, the HTTP calls and the import probe are injected.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.services import overlay

log = logging.getLogger(__name__)

PYPI_URL = "https://pypi.org/pypi/{name}/json"
LOOKUP_BUDGET_S = 8.0               # all PyPI lookups together
DOWNLOAD_TIMEOUT_S = 60.0
PIP_TIMEOUT_S = 900.0
PROBE_TIMEOUT_S = 60.0
CHECK_INTERVAL_S = 6 * 3600.0       # the PyPI lookup runs at most this often; repairs always run
MIN_PYTHON = (3, 12)
PYTHON_MAX = (3, 14)
RESTART_ENV = "AISS_RESTARTED"

#: Tools the app shells out to; a missing one is a real problem for downloading or burning.
REQUIRED_TOOLS = (("ffmpeg", "winget install Gyan.FFmpeg"),
                  ("ffprobe", "winget install Gyan.FFmpeg"),
                  ("deno", "winget install DenoLand.Deno"))

#: Distribution name -> import name, for the few packages whose names differ.
IMPORT_NAMES = {"pyside6": "PySide6", "faster-whisper": "faster_whisper", "yt-dlp": "yt_dlp",
                "huggingface-hub": "huggingface_hub", "sherpa-onnx": "sherpa_onnx",
                "pyinstaller": "PyInstaller", "pytest-qt": "pytestqt"}

_LINE = re.compile(r"^(?P<name>[A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*(?P<op>==|>=|<=|~=|!=|>|<)?\s*(?P<version>[^\s;#]*)")


@dataclass(frozen=True)
class Requirement:
    """One line of a requirements file."""

    name: str
    version: str = ""
    pinned: bool = False

    @property
    def module(self) -> str:
        return IMPORT_NAMES.get(self.name.lower(), self.name.replace("-", "_"))

    @property
    def floating(self) -> bool:
        """yt-dlp is kept up to date by the overlay, so its pin is only a minimum."""
        return self.name.lower() in overlay.PACKAGES


@dataclass(frozen=True)
class Problem:
    """Something that is wrong with the environment."""

    level: str        # "error" (the app may not work) or "warning" (worth knowing)
    title: str
    detail: str = ""
    fix: str = ""

    def text(self) -> str:
        out = self.title
        if self.detail:
            out += f" ({self.detail})"
        return out


@dataclass
class Result:
    """What the start-up check did and what is still wrong."""

    errors: list[str] = field(default_factory=list)       # still wrong after the repair attempt
    notes: list[str] = field(default_factory=list)        # informational, never shown as an error
    updated: list[str] = field(default_factory=list)      # "yt-dlp 2026.8.19 -> 2026.9.2"
    repaired: bool = False                                # pip re-installed the pinned libraries
    restart: bool = False                                 # loaded modules are stale: start the app again
    looked_up: bool = False                               # the PyPI lookup was attempted (throttle stamp)
    checked: int = 0                                      # how many things were really verified

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_requirements(text: str) -> list[Requirement]:
    """Parse a requirements file: comments, blank lines, `-r other.txt` and environment markers are skipped."""
    requirements: list[Requirement] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        version = match.group("version").strip()
        requirements.append(Requirement(name=match.group("name"), version=version,
                                        pinned=match.group("op") == "==" and bool(version)))
    return requirements


def requirements_path() -> Path | None:
    """The runtime requirements next to the sources, or None in a frozen build."""
    if getattr(sys, "frozen", False):
        return None
    path = Path(__file__).resolve().parents[2] / "requirements.txt"
    return path if path.is_file() else None


def parse_text_or_file() -> list[Requirement]:
    path = requirements_path()
    if path is None:
        return []
    try:
        return parse_requirements(path.read_text(encoding="utf-8"))
    except OSError as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return []


# -- offline checks ----------------------------------------------------------------------------

def check_environment(requirements: list[Requirement] | None = None, tools: bool = True,
                      modules: tuple[str, ...] = ()) -> list[Problem]:
    """Problems visible without any network: Python version, installed packages, importable modules, tools.

    `modules` are import names that must be importable (used by the frozen build, which has no requirements file).
    """
    problems: list[Problem] = []
    if sys.version_info[:2] < MIN_PYTHON:
        problems.append(Problem("error", f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required",
                                detail=f"running Python {sys.version.split()[0]}"))
    elif sys.version_info[:2] > PYTHON_MAX:
        problems.append(Problem("warning", f"Python {sys.version.split()[0]} is newer than the tested range",
                                detail=f"tested with {MIN_PYTHON[0]}.{MIN_PYTHON[1]} - {PYTHON_MAX[0]}.{PYTHON_MAX[1]}"))

    for requirement in requirements if requirements is not None else ():
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(Problem("error", f"{requirement.name} is not installed",
                                    detail=f"required version {requirement.version}" if requirement.version else ""))
            continue
        if requirement.pinned and not requirement.floating and installed != requirement.version:
            problems.append(Problem("error", f"{requirement.name} {installed} differs from the pinned "
                                             f"{requirement.version}",
                                    detail="the pinned versions are the ones this release is tested with"))
        if _find_spec(requirement.module) is None:
            problems.append(Problem("error", f"{requirement.name} is installed but cannot be imported",
                                    detail=f"module {requirement.module!r} was not found"))

    for module in modules:
        if _find_spec(module) is None:
            problems.append(Problem("error", f"The component {module} is missing from this installation",
                                    detail="reinstall the application"))

    if tools:
        for tool, hint in REQUIRED_TOOLS:
            if shutil.which(tool) is None:
                problems.append(Problem("error" if getattr(sys, "frozen", False) else "warning",
                                        f"{tool} was not found",
                                        detail="needed to download or process video"
                                               + (" (deno: to get past YouTube's challenge)" if tool == "deno" else ""),
                                        fix=hint))
    return problems


def _find_spec(module: str):
    try:
        return importlib.util.find_spec(module)
    except (ImportError, ValueError, AttributeError):
        return None


# -- network helpers -----------------------------------------------------------------------------

def http_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "AI-Subtitle-Studio"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_bytes(url: str, timeout: float, limit: int = overlay.MAX_WHEEL_BYTES) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "AI-Subtitle-Studio"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise OSError("download is larger than expected")
    return data


def pick_wheel(payload: dict) -> tuple[str, str, str]:
    """(version, url, sha256) of the pure-Python wheel in a PyPI JSON document. Raises ValueError if there is none."""
    version = str(payload.get("info", {}).get("version", "")).strip()
    for item in payload.get("urls", []):
        filename = str(item.get("filename", ""))
        if item.get("packagetype") == "bdist_wheel" and filename.endswith("-py3-none-any.whl") \
                and not item.get("yanked"):
            return version, str(item.get("url", "")), str(item.get("digests", {}).get("sha256", ""))
    raise ValueError("no pure-Python wheel published")


def installed_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


# -- repair --------------------------------------------------------------------------------------

def pip_command(path: Path) -> list[str]:
    return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(path)]


def run_pip(command: list[str], cwd: str, progress: Callable[[str], None]) -> int:
    """Run pip, forwarding its output lines to `progress`. Returns the exit code (124 = timed out)."""
    started = time.monotonic()
    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", creationflags=creation)
    for line in process.stdout or ():
        progress(line.rstrip())
        if time.monotonic() - started > PIP_TIMEOUT_S:
            process.kill()
            return 124
    return process.wait()


def probe_overlay(folder: Path, name: str, timeout: float = PROBE_TIMEOUT_S) -> tuple[bool, str]:
    """Import the new release in a child process, through the same finder the app uses, and check that the
    module really comes from `folder`. Works from sources and in the frozen build."""
    frozen = bool(getattr(sys, "frozen", False))
    command = [sys.executable] if frozen else [sys.executable, "-m", "app.main"]
    command += ["--overlay-probe", str(folder), overlay.PACKAGES[name]]
    env = dict(os.environ)
    if not frozen:
        env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")])
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return done.returncode == 0, (done.stdout + done.stderr).strip()[-300:]


def overlay_probe_main(folder: str, module: str) -> int:
    """Child-process entry (`--overlay-probe <folder> <module>`): 0 only if `module` imports from `folder`."""
    overlay.activate_dirs({module: Path(folder)})
    try:
        imported = importlib.import_module(module)
    except Exception as exc:                             # noqa: BLE001
        print(f"import failed: {exc}")
        return 1
    origin = Path(getattr(imported, "__file__", "") or "").resolve()
    if not origin.is_relative_to(Path(folder).resolve()):
        print(f"{module} was loaded from {origin}, not from the new release")
        return 1
    print(f"{module} ok")
    return 0


def update_floating(progress: Callable[[str], None], result: Result, *, root: Path | None,
                    fetch_json=http_json, fetch_bytes=http_bytes, probe=probe_overlay,
                    clock=time.monotonic) -> None:
    """Install newer yt-dlp / yt-dlp-ejs releases. Network or archive problems become notes, never errors."""
    deadline = clock() + LOOKUP_BUDGET_S
    for name in overlay.PACKAGES:
        remaining = deadline - clock()
        if remaining <= 0.5:
            result.notes.append("The update lookup ran out of time; it will be tried again later.")
            break
        try:
            progress(f"Checking {name}...")
            version, url, sha256 = pick_wheel(fetch_json(PYPI_URL.format(name=name), min(remaining, 4.0)))
            result.looked_up = True                       # only a successful lookup starts the 6-hour pause
        except Exception as exc:                         # noqa: BLE001 - offline is normal
            log.info("Update lookup for %s failed: %s", name, exc)
            result.notes.append(f"Could not look up {name} ({exc}); the installed version is used.")
            if isinstance(exc, OSError):
                break                                     # offline: do not wait for the next package too
            continue
        current = installed_version(name)
        result.checked += 1
        if current and overlay.version_key(version) <= overlay.version_key(current):
            continue
        if overlay.is_known_bad(name, version, root):
            continue
        if not overlay.allowed_url(url):
            result.notes.append(f"{name}: unexpected download address; skipped.")
            continue
        progress(f"Updating {name} {current or '-'} -> {version}...")
        folder = None
        try:
            data = fetch_bytes(url, DOWNLOAD_TIMEOUT_S)
            folder = overlay.install_wheel(name, version, data, sha256, root)
            good, detail = probe(folder, name)
        except Exception as exc:                         # noqa: BLE001 - keep the installed version
            log.warning("Update of %s failed: %s", name, exc)
            result.notes.append(f"{name} {version} could not be installed ({exc}); the installed version is used.")
            if folder is not None:
                overlay.remove(folder)
            continue
        if not good:
            log.warning("%s %s failed the import probe: %s", name, version, detail)
            overlay.remove(folder)
            overlay.mark_bad(name, version, root)
            result.notes.append(f"{name} {version} did not pass the self-check; the installed version is used.")
            continue
        overlay.mark_ok(folder)
        result.updated.append(f"{name} {current or '-'} -> {version}")
        if overlay.PACKAGES[name] in sys.modules:
            result.restart = True                         # the old copy is already loaded in this process
    overlay.prune(root)
    overlay.activate(root)


def perform_check(*, requirements: list[Requirement] | None = None, modules: tuple[str, ...] = (),
                  frozen: bool | None = None, root: Path | None = None, last_lookup: float = 0.0,
                  now: float | None = None, progress: Callable[[str], None] = lambda text: None,
                  runner=run_pip, fetch_json=http_json, fetch_bytes=http_bytes, probe=probe_overlay,
                  tools: bool = True, restarted: bool | None = None) -> Result:
    """The whole start-up job: verify, repair what can be repaired, update yt-dlp, report what is left."""
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    now = time.time() if now is None else now
    restarted = bool(os.environ.get(RESTART_ENV)) if restarted is None else restarted
    requirements = [] if frozen else (parse_text_or_file() if requirements is None else requirements)
    result = Result()

    progress("Checking the libraries...")
    problems = check_environment(requirements, tools=tools, modules=modules if frozen else ())
    result.checked = len(requirements) + (len(modules) if frozen else 0)
    fixable = [p for p in problems if p.level == "error" and ("not installed" in p.title or "differs" in p.title
                                                            or "cannot be imported" in p.title)]
    path = requirements_path()
    if fixable and not frozen and path is not None:
        progress("Repairing the libraries (this can take a few minutes)...")
        try:
            code = runner(pip_command(path), str(path.parent), progress)
        except OSError as exc:
            code = -1
            result.notes.append(f"pip could not be started: {exc}")
        importlib.invalidate_caches()
        if code == 0:
            result.repaired = True
            result.notes.append("The libraries were repaired.")
            problems = check_environment(requirements, tools=tools)
            if not restarted:
                result.restart = True                    # modules loaded before the repair may be stale
        else:
            result.notes.append(f"pip exited with code {code}.")
    for problem in problems:
        if problem.level == "error":
            result.errors.append(problem.text() + (f" - {problem.fix}" if problem.fix else ""))
        elif problem.fix:
            result.notes.append(f"{problem.text()} - {problem.fix}")
        else:
            result.notes.append(problem.text())

    if now - last_lookup >= CHECK_INTERVAL_S:
        update_floating(progress, result, root=root, fetch_json=fetch_json, fetch_bytes=fetch_bytes, probe=probe)
    else:
        overlay.activate(root)
    if _find_spec("yt_dlp") is None:
        result.errors.append("yt-dlp (the video downloader) cannot be loaded")
    return result


def restart_command() -> list[str]:
    """The command that starts this application again with the same arguments."""
    if getattr(sys, "frozen", False):
        return [sys.executable, *sys.argv[1:]]
    return [sys.executable, "-m", "app.main", *sys.argv[1:]]
