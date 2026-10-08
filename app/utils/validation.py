"""Small input validation helpers."""

from __future__ import annotations

from urllib.parse import urlparse


def is_url(text: str) -> bool:
    """True for absolute http(s) URLs with a host."""
    try:
        parsed = urlparse(text.strip())
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)
