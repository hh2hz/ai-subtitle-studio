"""Minimal HTTP client for subtitle APIs (stdlib only, injectable for tests)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from app import __version__
from app.providers.base import ProviderError

USER_AGENT = f"AISubtitleStudio v{__version__}"
TIMEOUT_S = 20


class HttpClient:
    def _request(self, url: str, data: bytes | None = None, headers: dict | None = None, method: str = "GET") -> bytes:
        all_headers = {"User-Agent": USER_AGENT, "Accept": "*/*", **(headers or {})}
        request = urllib.request.Request(url, data=data, headers=all_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise ProviderError(f"HTTP 429 from {urllib.parse.urlsplit(url).netloc}", "error.rate_limited",
                                    retryable=True) from exc
            if exc.code in (401, 403):
                raise ProviderError(f"HTTP {exc.code}: access denied (check the API key)",
                                    "error.provider_auth") from exc
            raise ProviderError(f"HTTP {exc.code} from {urllib.parse.urlsplit(url).netloc}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"Network error: {exc}", "error.provider_network", retryable=True) from exc

    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None) -> dict:
        if params:
            url = f"{url}?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}"
        return json.loads(self._request(url, headers=headers))

    def post_json(self, url: str, body: dict, headers: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8")
        return json.loads(self._request(url, data=data, method="POST",
                                        headers={"Content-Type": "application/json", **(headers or {})}))

    def get_bytes(self, url: str, headers: dict | None = None) -> bytes:
        return self._request(url, headers=headers)
