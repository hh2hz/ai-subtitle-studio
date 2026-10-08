"""Crash-safe file writes and append-only JSON Lines checkpoints."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

_REPLACE_RETRIES = 10
_REPLACE_DELAY_S = 0.05


def _replace_with_retry(src: str, dst: Path) -> None:
    # On Windows, antivirus or indexers can briefly lock the destination file.
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_DELAY_S * (attempt + 1))


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Write data so that path holds either the old or the complete new content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_text(path: Path | str, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path | str, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: Path | str) -> list[dict]:
    """Read records up to the first damaged line and drop the damaged tail from the file.

    A process killed mid-write can leave a partial last line; truncating it keeps later appends valid.
    """
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    valid_bytes = 0
    with open(path, "rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                break
            try:
                records.append(json.loads(raw))
            except ValueError:
                break
            valid_bytes += len(raw)
    if valid_bytes != path.stat().st_size:
        with open(path, "r+b") as handle:
            handle.truncate(valid_bytes)
    return records


class JsonlWriter:
    """Append-only JSON Lines writer that fsyncs every record."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "ab")

    def write(self, record: dict) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def write_many(self, records: Iterable[dict]) -> None:
        for record in records:
            self.write(record)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
