"""SQLite connection and versioned schema migrations.

Media files never go into the database; only paths and metadata are stored.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

_NOW = "(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"

# Each entry migrates from version N to N+1. Never edit a released migration; append a new one.
_MIGRATIONS: tuple[str, ...] = (
    f"""
    CREATE TABLE settings (
        key         TEXT PRIMARY KEY,
        value       TEXT NOT NULL,                  -- JSON
        updated_at  TEXT NOT NULL DEFAULT {_NOW}
    );

    CREATE TABLE series (
        id          INTEGER PRIMARY KEY,
        name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
        notes       TEXT NOT NULL DEFAULT '',
        created_at  TEXT NOT NULL DEFAULT {_NOW},
        updated_at  TEXT NOT NULL DEFAULT {_NOW}
    );

    CREATE TABLE jobs (
        id               INTEGER PRIMARY KEY,
        input_type       TEXT NOT NULL CHECK (input_type IN ('file', 'url')),
        input_value      TEXT NOT NULL,
        input_hash       TEXT,
        config_hash      TEXT,
        source_language  TEXT NOT NULL,
        target_language  TEXT NOT NULL,
        mode             TEXT NOT NULL,
        status           TEXT NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending', 'running', 'paused', 'cancelled', 'failed', 'completed')),
        series_id        INTEGER REFERENCES series(id) ON DELETE SET NULL,
        season           INTEGER,
        episode          INTEGER,
        episode_title    TEXT,
        year             INTEGER,
        output_dir       TEXT,
        output_paths     TEXT NOT NULL DEFAULT '{{}}',  -- JSON: kind -> path
        warnings         TEXT NOT NULL DEFAULT '[]',    -- JSON list
        error            TEXT,
        stats            TEXT NOT NULL DEFAULT '{{}}',  -- JSON: confidence statistics, timings
        created_at       TEXT NOT NULL DEFAULT {_NOW},
        updated_at       TEXT NOT NULL DEFAULT {_NOW}
    );
    CREATE INDEX idx_jobs_created_at ON jobs (created_at);

    CREATE TABLE job_stages (
        job_id         INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        stage          TEXT NOT NULL,
        status         TEXT NOT NULL
                       CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
        cache_key      TEXT,
        artifact_path  TEXT,
        error          TEXT,
        started_at     TEXT,
        finished_at    TEXT,
        PRIMARY KEY (job_id, stage)
    );

    CREATE TABLE characters (
        id          INTEGER PRIMARY KEY,
        series_id   INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
        name        TEXT NOT NULL,
        spellings   TEXT NOT NULL DEFAULT '{{}}',  -- JSON: language code -> preferred spelling
        notes       TEXT NOT NULL DEFAULT '',
        UNIQUE (series_id, name)
    );

    CREATE TABLE speaker_mappings (
        id            INTEGER PRIMARY KEY,
        job_id        INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        speaker_label TEXT NOT NULL,
        character_id  INTEGER REFERENCES characters(id) ON DELETE SET NULL,
        confirmed     INTEGER NOT NULL DEFAULT 0 CHECK (confirmed IN (0, 1)),
        UNIQUE (job_id, speaker_label)
    );

    CREATE TABLE translation_memory (
        id               INTEGER PRIMARY KEY,
        series_id        INTEGER REFERENCES series(id) ON DELETE CASCADE,  -- NULL = global
        source_language  TEXT NOT NULL,
        target_language  TEXT NOT NULL,
        source_term      TEXT NOT NULL,
        target_term      TEXT NOT NULL,
        confidence       REAL NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
        usage_count      INTEGER NOT NULL DEFAULT 0 CHECK (usage_count >= 0),
        context_examples TEXT NOT NULL DEFAULT '[]',  -- JSON list
        created_at       TEXT NOT NULL DEFAULT {_NOW},
        updated_at       TEXT NOT NULL DEFAULT {_NOW}
    );
    -- COALESCE makes global (NULL series) entries unique as well.
    CREATE UNIQUE INDEX idx_tm_unique ON translation_memory
        (COALESCE(series_id, 0), source_language, target_language, source_term);

    CREATE TABLE episode_summaries (
        id          INTEGER PRIMARY KEY,
        series_id   INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
        season      INTEGER,
        episode     INTEGER,
        summary     TEXT NOT NULL,
        job_id      INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
        created_at  TEXT NOT NULL DEFAULT {_NOW}
    );
    """,
    # v2: batch queue (Phase 6.1). A queued job is a 'pending' row with queue_config set.
    """
    ALTER TABLE jobs ADD COLUMN queue_config TEXT;       -- JSON: series override; NULL = not a queue entry
    ALTER TABLE jobs ADD COLUMN queue_order  INTEGER;    -- position in the queue (ascending)
    CREATE INDEX idx_jobs_queue ON jobs (queue_order) WHERE queue_config IS NOT NULL;
    """,
    # v3: a series the user keeps (series picker, D-076). `saved` marks it as kept on purpose; the other columns
    # hold what the lookup found, so the poster can be shown offline from the cache.
    """
    ALTER TABLE series ADD COLUMN saved      INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE series ADD COLUMN source_id  TEXT;       -- TVMaze show id
    ALTER TABLE series ADD COLUMN image_url  TEXT;       -- poster URL
    ALTER TABLE series ADD COLUMN image_path TEXT;       -- cached poster file
    CREATE INDEX idx_series_saved ON series (saved) WHERE saved = 1;
    """,
    # v4: drop the three tables that no code ever wrote to or read from (Phase 6.8, D-106). Wiring them would need
    # a database handle inside the pipeline, which the discovery/brief stage does not have; leaving empty tables
    # around is worse than removing them. `translation_memory` is NOT dropped: the review editor now records
    # learned corrections there.
    """
    DROP TABLE IF EXISTS speaker_mappings;
    DROP TABLE IF EXISTS characters;
    DROP TABLE IF EXISTS episode_summaries;
    """,
)

SCHEMA_VERSION = len(_MIGRATIONS)


class DatabaseError(RuntimeError):
    pass


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Turn on WAL when the folder allows it; a network share or a restricted folder must not stop the app (D-112)."""
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError as exc:
        log.warning("WAL journal mode is unavailable (%s); using the default journal mode", exc)


class Database:
    """One SQLite connection. Worker threads must open their own Database instance."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        in_memory = self.path == ":memory:"
        if not in_memory:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if not in_memory:
            _enable_wal(self._conn)
        self._migrate()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @property
    def schema_version(self) -> int:
        return self._conn.execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        current = self.schema_version
        if current > SCHEMA_VERSION:
            raise DatabaseError(
                f"Database schema version {current} is newer than supported version {SCHEMA_VERSION}"
            )
        for version in range(current, SCHEMA_VERSION):
            log.info("Migrating database %s: v%d -> v%d", self.path, version, version + 1)
            script = f"BEGIN;\n{_MIGRATIONS[version]}\nPRAGMA user_version = {version + 1};\nCOMMIT;"
            try:
                self._conn.executescript(script)
            except sqlite3.Error as exc:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise DatabaseError(f"Migration to v{version + 1} failed: {exc}") from exc

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
