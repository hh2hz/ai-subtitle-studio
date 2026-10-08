"""Series lookup for the picker: suggestions and posters from TVMaze (free, no API key, D-076).

Nothing here raises into the UI: a failed lookup simply means no suggestions. HTTP is injected so tests can run
without the network, and posters are cached on disk so a kept series shows its picture offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

SEARCH_URL = "https://api.tvmaze.com/search/shows?q={query}"
TIMEOUT_S = 8.0
USER_AGENT = "AISubtitleStudio/1.0 (series picker)"
_ALLOWED_SCHEMES = ("http://", "https://")


@dataclass(frozen=True)
class SeriesCandidate:
    source_id: str
    name: str
    year: int | None = None
    image_url: str | None = None
    image_path: Path | None = None      # set when the poster is already on disk

    @property
    def label(self) -> str:
        return f"{self.name} ({self.year})" if self.year else self.name


def http_get(url: str) -> bytes:
    _check_scheme(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:   # noqa: S310 - scheme checked above
        return response.read()


def _check_scheme(url: str) -> None:
    """Only http(s): a hostile payload must not turn a poster URL into a local file read."""
    if not str(url).lower().startswith(_ALLOWED_SCHEMES):
        raise ValueError(f"Refusing a non-HTTP URL: {url}")


def _year(premiered: object) -> int | None:
    text = str(premiered or "")
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else None


def parse_search(payload: bytes, limit: int = 8) -> list[SeriesCandidate]:
    """Turn a TVMaze `/search/shows` response into candidates (separate from HTTP, so it is easy to test)."""
    try:
        data = json.loads(payload.decode("utf-8", "replace"))
    except (ValueError, AttributeError):
        return []
    if not isinstance(data, list):
        return []
    found: list[SeriesCandidate] = []
    for row in data:
        show = (row or {}).get("show") or {}
        name = str(show.get("name") or "").strip()
        if not name:
            continue
        image = show.get("image") or {}
        found.append(SeriesCandidate(
            source_id=str(show.get("id") or ""),
            name=name,
            year=_year(show.get("premiered")),
            image_url=image.get("medium") or image.get("original"),
        ))
        if len(found) >= limit:
            break
    return found


def search(query: str, limit: int = 8, fetch: Callable[[str], bytes] = http_get) -> list[SeriesCandidate]:
    """Look up series by name. Returns [] on any failure (offline, rate limit, bad payload)."""
    query = query.strip()
    if len(query) < 3:
        return []
    try:
        return parse_search(fetch(SEARCH_URL.format(query=urllib.parse.quote(query))), limit)
    except Exception as exc:      # noqa: BLE001 - a lookup must never break the window
        log.warning("Series lookup failed: %s", exc)
        return []


def poster_name(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16] + ".img"


def poster_path(url: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / poster_name(url)


def cache_poster(url: str, cache_dir: Path, fetch: Callable[[str], bytes] = http_get) -> Path | None:
    """Download a poster once, then reuse the file. Returns None when it cannot be fetched."""
    target = poster_path(url, cache_dir)
    if target.is_file():
        return target
    try:
        _check_scheme(url)
        data = fetch(url)
        if not data:
            return None
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(data)
        partial.replace(target)
        return target
    except Exception as exc:      # noqa: BLE001 - a missing poster must not break the picker
        log.warning("Could not cache a series poster: %s", exc)
        return None
