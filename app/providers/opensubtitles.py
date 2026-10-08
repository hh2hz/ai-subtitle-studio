"""OpenSubtitles.com REST API (optional, needs a free API key).

Implemented from public documentation; not tested against the live service (no key, and the API is not
reachable from the development environment). Anonymous downloads are limited (documented: 5 per 24 h per IP).
"""

from __future__ import annotations

import logging

from app.core.subtitle_formats import decode_subtitle_bytes, parse_subtitle_text
from app.providers.base import COMMUNITY, ProviderUnavailable, SubtitleEvidence, SubtitleRequest
from app.providers.http import HttpClient

log = logging.getLogger(__name__)
BASE_URL = "https://api.opensubtitles.com/api/v1"


class OpenSubtitlesProvider:
    name = "opensubtitles"

    def __init__(self, api_key: str, http: HttpClient | None = None, max_downloads: int = 2):
        self._key = api_key.strip()
        self._http = http or HttpClient()
        self._max_downloads = max_downloads

    def fetch(self, request: SubtitleRequest) -> list[SubtitleEvidence]:
        if not self._key:
            raise ProviderUnavailable("no API key configured")
        query = request.series.series_name or request.title
        if not query:
            raise ProviderUnavailable("no title to search for")
        languages = sorted({l for l in (request.source_language, request.target_language) if l != "auto"})
        headers = {"Api-Key": self._key}
        params = {
            "query": query,
            "languages": ",".join(languages),
            "season_number": request.series.season,
            "episode_number": request.series.episode,
            "type": "episode" if request.series.episode is not None else None,
        }
        data = self._http.get_json(f"{BASE_URL}/subtitles", params=params, headers=headers).get("data", [])
        best: dict[str, dict] = {}
        for item in data:
            attrs = item.get("attributes", {})
            if not attrs.get("files"):
                continue
            language = attrs.get("language")
            current = best.get(language)
            if current is None or attrs.get("download_count", 0) > current.get("download_count", 0):
                best[language] = attrs
        evidence = []
        for language, attrs in list(best.items())[: self._max_downloads]:
            file_id = attrs["files"][0]["file_id"]
            link = self._http.post_json(f"{BASE_URL}/download", {"file_id": file_id}, headers=headers)["link"]
            text = decode_subtitle_bytes(self._http.get_bytes(link), language)
            machine = bool(attrs.get("machine_translated") or attrs.get("ai_translated"))
            evidence.append(SubtitleEvidence(
                provider=self.name, language=language, kind=COMMUNITY, cues=parse_subtitle_text(text),
                reference=f"file_id:{file_id} release:{attrs.get('release', '')}", machine_generated=machine,
                notes=["release may not match this video's timing"]))
        return evidence
