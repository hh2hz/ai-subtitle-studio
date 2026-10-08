"""Update check and installer download from the project's GitHub releases (D-118).

Only one host is trusted: the repository below. The check reads `releases/latest`, the installer asset must be
served from this repository's release downloads over HTTPS, and its SHA-256 (the digest GitHub publishes for every
asset) is verified before the installer is started. No personal data is sent: a plain GET with a User-Agent.
Nothing here raises into the UI except UpdateError, which carries a message that is safe to show.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core.errors import JobCancelled

log = logging.getLogger(__name__)

REPO = "hh2hz/ai-subtitle-studio"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
DOWNLOAD_PREFIX = f"https://github.com/{REPO}/releases/download/"
ASSET_NAME = "AI-Subtitle-Studio-Setup.exe"
TIMEOUT_S = 10.0
CHUNK = 256 * 1024
MAX_API_BYTES = 1_000_000
USER_AGENT = "AISubtitleStudio-update-check"
_VERSION = re.compile(r"^v?(\d+(?:\.\d+){0,3})$")
_SHA256 = re.compile(r"^sha256:([0-9a-fA-F]{64})$")


class UpdateError(Exception):
    """A problem with checking, downloading or verifying an update (the message can be shown to the user)."""


@dataclass(frozen=True)
class UpdateInfo:
    version: str                 # "1.1.0" (no leading v)
    page_url: str                # the release page
    download_url: str            # the installer asset
    size: int                    # bytes, 0 when unknown
    sha256: str                  # lowercase hex of the installer
    notes: str = ""              # release notes (plain text, shown truncated)


def parse_version(text: str) -> tuple[int, ...] | None:
    """'v1.2.3' -> (1, 2, 3); anything that is not a plain numeric version (e.g. '1.0.0-beta') -> None."""
    match = _VERSION.match(str(text).strip())
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def is_newer(latest: str, current: str) -> bool:
    new, old = parse_version(latest), parse_version(current)
    if new is None or old is None:
        return False
    width = max(len(new), len(old))
    pad = lambda parts: parts + (0,) * (width - len(parts))      # noqa: E731 - 1.1 == 1.1.0
    return pad(new) > pad(old)


class _HttpsOnly(urllib.request.HTTPRedirectHandler):
    """Redirects (GitHub sends downloads to a CDN) must stay on HTTPS."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.lower().startswith("https://"):
            raise urllib.error.URLError(f"refusing a non-HTTPS redirect: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url: str):
    if not url.lower().startswith("https://"):
        raise UpdateError("Refusing a non-HTTPS address")
    opener = urllib.request.build_opener(_HttpsOnly)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream, "
                                                   "application/vnd.github+json"})
    return opener.open(request, timeout=TIMEOUT_S)               # noqa: S310 - https checked above


def http_get(url: str) -> bytes:
    with _open(url) as response:
        data = response.read(MAX_API_BYTES + 1)
    if len(data) > MAX_API_BYTES:
        raise UpdateError("Unexpectedly large response")
    return data


def parse_release(payload: bytes) -> UpdateInfo:
    """Turn a `releases/latest` response into an UpdateInfo, or raise UpdateError when it is not usable."""
    try:
        data = json.loads(payload.decode("utf-8", "replace"))
        tag = str(data["tag_name"])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise UpdateError("The release information could not be read") from exc
    version = parse_version(tag)
    if version is None:
        raise UpdateError(f"Unrecognised release version: {tag}")
    for asset in data.get("assets") or []:
        if not isinstance(asset, dict) or asset.get("name") != ASSET_NAME:
            continue
        url = str(asset.get("browser_download_url") or "")
        digest = _SHA256.match(str(asset.get("digest") or ""))
        if not url.startswith(DOWNLOAD_PREFIX):
            raise UpdateError("The installer is not hosted by the project's releases")
        if not digest:
            raise UpdateError("The release has no SHA-256 for the installer")
        size = asset.get("size")
        return UpdateInfo(version=".".join(str(p) for p in version), page_url=RELEASES_PAGE, download_url=url,
                          size=size if isinstance(size, int) and size > 0 else 0, sha256=digest.group(1).lower(),
                          notes=str(data.get("body") or "")[:2000])
    raise UpdateError("The release has no installer file")


def check_latest(fetch: Callable[[str], bytes] = http_get) -> UpdateInfo:
    """The newest published release (the caller decides with is_newer whether it is an update)."""
    try:
        return parse_release(fetch(LATEST_URL))
    except UpdateError:
        raise
    except Exception as exc:                  # noqa: BLE001 - network errors of every kind
        raise UpdateError(f"{type(exc).__name__}: {exc}") from exc


def can_self_update() -> bool:
    """Only the installed Windows program can replace itself; a source checkout just opens the release page."""
    return sys.platform == "win32" and bool(getattr(sys, "frozen", False))


def download_installer(info: UpdateInfo, progress: Callable[[float], None] | None = None,
                       cancel: threading.Event | None = None, folder: Path | None = None,
                       opener: Callable[[str], object] = _open) -> Path:
    """Download the installer, verify its SHA-256 and return the finished file. Raises UpdateError/JobCancelled."""
    if not info.download_url.startswith(DOWNLOAD_PREFIX) or not info.sha256:
        raise UpdateError("The update has no verifiable installer")
    folder = folder or Path(tempfile.gettempdir()) / "AISubtitleStudio-update"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*"):                              # earlier downloads are not needed any more
        try:
            old.unlink()
        except OSError:
            pass
    target = folder / f"AI-Subtitle-Studio-Setup-{info.version}.exe"
    part = target.with_suffix(".exe.part")
    digest = hashlib.sha256()
    done = 0
    try:
        with opener(info.download_url) as response, part.open("wb") as out:
            total = info.size or int(getattr(response, "headers", {}).get("Content-Length") or 0)
            while True:
                if cancel is not None and cancel.is_set():
                    raise JobCancelled()
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress is not None and total:
                    progress(min(done / total, 1.0))
        if info.size and done != info.size:
            raise UpdateError(f"Incomplete download ({done} of {info.size} bytes)")
        if digest.hexdigest() != info.sha256:
            raise UpdateError("The downloaded file does not match the published SHA-256; it was discarded")
        os.replace(part, target)
    except (OSError, urllib.error.URLError) as exc:
        raise UpdateError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        part.unlink(missing_ok=True)
    return target


def launch_installer(path: Path) -> None:
    """Start the verified installer detached. /SILENT shows only the progress window; the installer closes any
    running copy, replaces the program files (settings, keys and history live outside them) and starts it again."""
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        subprocess.Popen([str(path), "/SILENT", "/SP-", "/CLOSEAPPLICATIONS"], close_fds=True,   # noqa: S603
                         creationflags=flags)
    except OSError as exc:
        raise UpdateError(f"The installer could not be started: {exc}") from exc
    log.info("Update installer started: %s", path)
