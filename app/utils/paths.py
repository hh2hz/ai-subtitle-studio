"""Filesystem locations for user data and bundled resources."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

APP_DIR_NAME = "AISubtitleStudio"
ENV_DATA_DIR = "AISS_DATA_DIR"


def default_data_root() -> Path:
    """Return the per-user data root. ENV_DATA_DIR overrides the platform default."""
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_DIR_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_DIR_NAME


def resources_dir() -> Path:
    """Return the bundled resources directory (source tree or PyInstaller bundle)."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root) / "app" / "resources"
    return Path(__file__).resolve().parents[1] / "resources"


def project_root() -> Path | None:
    """The source tree root, or None when running from a frozen build."""
    if getattr(sys, "frozen", False):
        return None
    return Path(__file__).resolve().parents[2]


def can_write(directory: Path) -> bool:
    """True when a file can really be created in `directory` (an ACL can deny writes while the folder exists)."""
    probe = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".write-test-{os.getpid()}"
        probe.write_bytes(b"")
        return True
    except OSError:
        return False
    finally:
        if probe is not None:
            try:
                probe.unlink()
            except OSError:
                pass


def resolve_data_root(preferred: Path) -> tuple[Path, str | None]:
    """Return a data root that can really be written to, with a note when it is not `preferred`.

    Started from a restricted shell (a sandbox, a service, an ACL-limited folder) the per-user data folder can be
    read-only, and then the settings database cannot be created at all. Instead of refusing to start, the app falls
    back to a folder it may write: `<project>/.appdata` from source, otherwise `<temp>/AISubtitleStudio`.
    """
    if can_write(preferred):
        return preferred, None
    candidates = [path for path in (project_root() and project_root() / ".appdata",
                                    Path(os.environ.get("TEMP") or "/tmp") / APP_DIR_NAME) if path]
    for candidate in candidates:
        if candidate != preferred and can_write(candidate):
            return candidate, (f"{preferred} is not writable here; using {candidate} instead "
                               f"(settings, jobs and logs stay separate from a normal start)")
    raise OSError(f"No writable data folder found (tried {preferred} and {len(candidates)} fallback(s))")


def default_output_dir() -> Path:
    """Installed app: Videos/AI Subtitle Studio (the install folder is replaced on updates). From source: the
    "output" folder in the project root."""
    if getattr(sys, "frozen", False):
        return Path.home() / "Videos" / "AI Subtitle Studio"
    return Path(__file__).resolve().parents[2] / "output"


def bundled_bin_dir() -> Path | None:
    """bin/ next to the installed executable: FFmpeg, FFprobe and Deno shipped with the installer."""
    if getattr(sys, "frozen", False):
        path = Path(sys.executable).resolve().parent / "bin"
        return path if path.is_dir() else None
    return None


def translations_dir() -> Path:
    return resources_dir() / "translations"


def model_ranking_files() -> list[Path]:
    """Name-based estimates first, then the shipped benchmark scores, then the user's own run
    (later files override earlier ones, so a real measurement always wins over an estimate)."""
    return [resources_dir() / "model_ranking_estimates.json",
            resources_dir() / "model_ranking.json",
            default_data_root() / "model_ranking.json"]


@dataclass(frozen=True)
class AppPaths:
    """All writable locations used by the application."""

    root: Path
    db_file: Path
    logs_dir: Path
    models_dir: Path
    cache_dir: Path
    jobs_dir: Path

    @classmethod
    def from_root(cls, root: Path | str) -> "AppPaths":
        root = Path(root)
        return cls(
            root=root,
            db_file=root / "studio.db",
            logs_dir=root / "logs",
            models_dir=root / "models",
            cache_dir=root / "cache",
            jobs_dir=root / "jobs",
        )

    def ensure(self) -> "AppPaths":
        """Create all directories. Safe to call repeatedly."""
        for directory in (self.root, self.logs_dir, self.models_dir, self.cache_dir, self.jobs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def with_models_from(self, other: Path) -> "AppPaths":
        """Keep using an existing models folder after a fallback: models are only read while the app runs, and a
        multi-gigabyte download should not repeat just because the shell that started the app is restricted."""
        from dataclasses import replace

        if (other / "models").is_dir():
            return replace(self, models_dir=other / "models")
        return self
