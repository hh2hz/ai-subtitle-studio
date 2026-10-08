"""SubDL API (optional, needs a free API key).

Implemented from public documentation (api.subdl.com/api/v1/subtitles, downloads from dl.subdl.com as ZIP);
not tested against the live service.
"""

from __future__ import annotations

import io
import logging
import zipfile

from app.core.subtitle_formats import decode_subtitle_bytes, parse_subtitle_text
from app.providers.base import COMMUNITY, ProviderError, ProviderUnavailable, SubtitleEvidence, SubtitleRequest
from app.providers.http import HttpClient

log = logging.getLogger(__name__)
SEARCH_URL = "https://api.subdl.com/api/v1/subtitles"
DOWNLOAD_BASE = "https://dl.subdl.com/subtitle/"
_MAX_ZIP_BYTES = 20 * 1024 * 1024


class SubdlProvider:
    name = "subdl"

    def __init__(self, api_key: str, http: HttpClient | None = None, max_downloads: int = 2):
        self._key = api_key.strip()
        self._http = http or HttpClient()
        self._max_downloads = max_downloads

    def fetch(self, request: SubtitleRequest) -> list[SubtitleEvidence]:
        if not self._key:
            raise ProviderUnavailable("no API key configured")
        name = request.series.series_name or request.title
        if not name:
            raise ProviderUnavailable("no title to search for")
        languages = [l.upper() for l in (request.source_language, request.target_language) if l != "auto"]
        params = {
            "api_key": self._key,
            "film_name": name,
            "type": "tv" if request.series.episode is not None else None,
            "season_number": request.series.season,
            "episode_number": request.series.episode,
            "languages": ",".join(dict.fromkeys(languages)),
            "subs_per_page": 10,
        }
        response = self._http.get_json(SEARCH_URL, params=params)
        if not response.get("status"):
            raise ProviderError(f"SubDL search failed: {response.get('error', 'unknown error')}")
        evidence = []
        seen_languages = set()
        for item in response.get("subtitles", []):
            language = str(item.get("lang") or item.get("language") or "").lower()
            if language in seen_languages or len(evidence) >= self._max_downloads:
                continue
            seen_languages.add(language)
            data = self._http.get_bytes(DOWNLOAD_BASE + str(item["url"]).lstrip("/"))
            text = _first_subtitle_in_zip(data, language)
            if text is None:
                continue
            evidence.append(SubtitleEvidence(
                provider=self.name, language=language, kind=COMMUNITY, cues=parse_subtitle_text(text),
                reference=f"release:{item.get('release_name', '')}", machine_generated=False,
                notes=["release may not match this video's timing"]))
        return evidence


def _first_subtitle_in_zip(data: bytes, language: str) -> str | None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return None
    for entry in archive.infolist():
        if entry.filename.lower().endswith((".srt", ".vtt")) and entry.file_size <= _MAX_ZIP_BYTES:
            return decode_subtitle_bytes(archive.read(entry), language)
    return None
