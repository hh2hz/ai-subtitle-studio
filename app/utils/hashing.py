"""Stable hashes for cache keys."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_SAMPLE_BYTES = 4 * 1024 * 1024


def file_fingerprint(path: Path | str) -> str:
    """Fast content fingerprint: size plus the first and last 4 MiB.

    Hashing whole multi-GB videos would take too long. Two different files with identical size,
    head and tail would collide; for video files this is accepted (see DECISIONS.md).
    """
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode("ascii"))
    with open(path, "rb") as handle:
        digest.update(handle.read(_SAMPLE_BYTES))
        if size > 2 * _SAMPLE_BYTES:
            handle.seek(size - _SAMPLE_BYTES)
            digest.update(handle.read(_SAMPLE_BYTES))
        elif size > _SAMPLE_BYTES:
            digest.update(handle.read())
    return digest.hexdigest()


def stable_hash(obj: Any, length: int = 16) -> str:
    """Hash of a JSON-serialisable object, independent of dict ordering."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:length]
