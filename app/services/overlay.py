"""Newer yt-dlp / yt-dlp-ejs releases that live in the user's data folder (D-114).

YouTube breaks yt-dlp every few weeks, and an installed (frozen) build cannot run pip. Both packages are pure
Python, so a newer release is downloaded as a wheel, verified, unpacked into `<data>/site/<name>-<version>/` and put
in front of the bundled copy at start-up. Nothing outside the data folder is ever written.

This module imports only the standard library and `app.utils.paths`, so it can run before anything heavy is loaded.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import logging
import os
import re
import shutil
import sys
import uuid
import zipfile
from pathlib import Path

from app.utils.paths import default_data_root

log = logging.getLogger(__name__)

#: distribution name -> import name. Both are floating by design (everything else stays pinned).
PACKAGES = {"yt-dlp": "yt_dlp", "yt-dlp-ejs": "yt_dlp_ejs"}
OK_MARKER = ".overlay-ok"
KEEP_VERSIONS = 2
MAX_WHEEL_BYTES = 40 * 1024 * 1024
_ALLOWED_HOST = "https://files.pythonhosted.org/"
_VERSION_DIR = re.compile(r"^(?P<name>[a-z0-9\-]+?)-(?P<version>[0-9][0-9A-Za-z.+]*)$")


def site_root(data_root: Path | None = None) -> Path:
    return (data_root or default_data_root()) / "site"


def version_key(text: str) -> tuple:
    """Sortable form of a release number such as 2026.9.1 or 0.8.0 (non-numeric parts count as 0)."""
    return tuple(int(p) if p.isdigit() else 0 for p in re.split(r"[.\-+]", text)[:6])


def valid_overlays(root: Path | None = None) -> dict[str, list[tuple[str, Path]]]:
    """name -> [(version, folder)] newest first, only folders that passed the import probe."""
    found: dict[str, list[tuple[str, Path]]] = {}
    base = site_root(root)
    try:
        entries = list(base.iterdir()) if base.is_dir() else []
    except OSError:
        return found
    for entry in entries:
        match = _VERSION_DIR.match(entry.name)
        if not match or match["name"] not in PACKAGES or not (entry / OK_MARKER).is_file():
            continue
        found.setdefault(match["name"], []).append((match["version"], entry))
    for items in found.values():
        items.sort(key=lambda item: version_key(item[0]), reverse=True)
    return found


class _OverlayFinder:
    """Meta-path finder that resolves yt_dlp / yt_dlp_ejs from the overlay folders before anything else.

    A plain `sys.path.insert` is not enough in a PyInstaller build, where the bundled copy can be found first.
    """

    def __init__(self, folders: dict[str, str]):
        self.folders = folders                      # import name -> folder that holds it

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".", 1)[0]
        folder = self.folders.get(top)
        if folder is None:
            return None
        return importlib.machinery.PathFinder.find_spec(fullname, [folder] if path is None else path, target)


def activate_dirs(folders: dict[str, Path]) -> None:
    """Make `folders` ({import name: overlay folder}) win over any other copy. Replaces an earlier activation."""
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _OverlayFinder)]
    if not folders:
        return
    mapping = {name: str(path) for name, path in folders.items()}
    sys.meta_path.insert(0, _OverlayFinder(mapping))
    for path in reversed(list(mapping.values())):
        if path not in sys.path:
            sys.path.insert(0, path)
    importlib.invalidate_caches()


def activate(root: Path | None = None) -> dict[str, str]:
    """Activate the newest valid overlay of every package. Never raises; returns {name: version} that is active."""
    try:
        chosen = {name: items[0] for name, items in valid_overlays(root).items() if items}
        activate_dirs({PACKAGES[name]: folder for name, (_, folder) in chosen.items()})
        return {name: version for name, (version, _) in chosen.items()}
    except Exception as exc:                        # noqa: BLE001 - a broken overlay must never stop the app
        log.warning("Overlay activation failed: %s", exc)
        return {}


def bad_marker(name: str, version: str, root: Path | None = None) -> Path:
    return site_root(root) / f"{name}-{version}.bad"


def is_known_bad(name: str, version: str, root: Path | None = None) -> bool:
    return bad_marker(name, version, root).exists()


def mark_bad(name: str, version: str, root: Path | None = None) -> None:
    try:
        marker = bad_marker(name, version, root)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("this release failed the import probe; it is not tried again\n", encoding="utf-8")
    except OSError as exc:
        log.warning("Cannot write %s: %s", bad_marker(name, version, root), exc)


def install_wheel(name: str, version: str, data: bytes, sha256: str, root: Path | None = None) -> Path:
    """Verify `data` against `sha256`, unpack it atomically into `<site>/<name>-<version>/` and return that folder.

    Raises ValueError on a checksum mismatch or an unsafe archive, OSError when the folder cannot be written.
    The folder is NOT active yet: it needs `mark_ok` after the import probe.
    """
    digest = hashlib.sha256(data).hexdigest()
    if not sha256 or digest.lower() != sha256.lower():
        raise ValueError(f"SHA-256 mismatch for {name} {version}")
    base = site_root(root)
    base.mkdir(parents=True, exist_ok=True)
    final = base / f"{name}-{version}"
    staging = base / f".tmp-{uuid.uuid4().hex}"
    wheel = base / f".tmp-{uuid.uuid4().hex}.whl"
    try:
        wheel.write_bytes(data)
        with zipfile.ZipFile(wheel) as archive:
            staging.mkdir()
            top = staging.resolve()
            for member in archive.infolist():
                target = (staging / member.filename).resolve()
                if member.filename.startswith(("/", "\\")) or not target.is_relative_to(top):
                    raise ValueError(f"unsafe path in the {name} archive: {member.filename}")
            archive.extractall(staging)
        if not (staging / PACKAGES[name]).is_dir():
            raise ValueError(f"the {name} archive does not contain {PACKAGES[name]}/")
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        os.replace(staging, final)
        return final
    finally:
        wheel.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def mark_ok(folder: Path) -> None:
    (folder / OK_MARKER).write_text("ok\n", encoding="utf-8")


def remove(folder: Path) -> None:
    shutil.rmtree(folder, ignore_errors=True)


def prune(root: Path | None = None, keep: int = KEEP_VERSIONS) -> None:
    """Keep only the newest `keep` valid versions of each package; broken or half-written folders are removed."""
    base = site_root(root)
    if not base.is_dir():
        return
    keep_dirs = {folder for items in valid_overlays(root).values() for _, folder in items[:keep]}
    for entry in base.iterdir():
        if entry.name.endswith(".bad") or entry in keep_dirs:
            continue
        if entry.is_dir() and (entry.name.startswith(".tmp-") or _VERSION_DIR.match(entry.name)):
            remove(entry)


def allowed_url(url: str) -> bool:
    return url.startswith(_ALLOWED_HOST)
