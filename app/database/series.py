"""Kept series: what the picker remembers about a series the user chose to keep (D-076).

The `series` table is shared with the job history: `ensure_series` in `JobsRepo` inserts a name per job, and this
repository only marks rows as kept and stores what the lookup found (external id, poster URL, cached poster file).
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.database.database import Database

log = logging.getLogger(__name__)


class SeriesRepo:
    def __init__(self, db: Database):
        self._db = db

    def save(self, name: str, source_id: str | None = None, image_url: str | None = None,
             image_path: str | None = None) -> int:
        """Keep a series and remember the poster. An existing row keeps its id and its job links."""
        name = name.strip()
        if not name:
            raise ValueError("A series needs a name")
        with self._db.conn:
            self._db.conn.execute(
                "INSERT INTO series (name, saved, source_id, image_url, image_path) VALUES (?, 1, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET saved = 1, source_id = excluded.source_id, "
                "image_url = excluded.image_url, image_path = excluded.image_path",
                (name, source_id, image_url, image_path))
        row = self._db.conn.execute("SELECT id FROM series WHERE name = ?", (name,)).fetchone()
        return int(row["id"])

    def saved(self) -> list[dict]:
        """Kept series, newest first, for the suggestions list and the settings dialog."""
        rows = self._db.conn.execute(
            "SELECT id, name, source_id, image_url, image_path FROM series WHERE saved = 1 "
            "ORDER BY updated_at DESC, name COLLATE NOCASE").fetchall()
        return [dict(r) for r in rows]

    def find(self, name: str) -> dict | None:
        row = self._db.conn.execute(
            "SELECT id, name, source_id, image_url, image_path FROM series WHERE name = ?",
            (name.strip(),)).fetchone()
        return dict(row) if row else None

    def forget(self, series_id: int, cache_dir: Path | None = None) -> None:
        """Stop keeping a series and drop its cached poster. The row itself stays: jobs may still reference it."""
        row = self._db.conn.execute("SELECT image_path FROM series WHERE id = ?", (series_id,)).fetchone()
        if row is None:
            return
        image_path = row["image_path"]
        with self._db.conn:
            self._db.conn.execute(
                "UPDATE series SET saved = 0, source_id = NULL, image_url = NULL, image_path = NULL, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?", (series_id,))
        if image_path and cache_dir is not None:
            try:
                cached = Path(image_path)
                if cached.parent == Path(cache_dir) and cached.is_file():
                    cached.unlink()
            except OSError as exc:
                log.warning("Could not remove the cached poster %s: %s", image_path, exc)
