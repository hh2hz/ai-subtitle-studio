"""Schema migration v3 -> v4: the unused tables are dropped, existing data survives (D-106, Phase 6.8)."""

import sqlite3

from app.database.database import _MIGRATIONS, SCHEMA_VERSION, Database

DROPPED = ("speaker_mappings", "characters", "episode_summaries")


def _tables(db_path) -> set:
    con = sqlite3.connect(db_path)
    try:
        return {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        con.close()


def _database_at_version(db_path, version: int) -> None:
    """Build a database that stops at `version`, exactly like an installation from that release."""
    con = sqlite3.connect(db_path)
    try:
        for script in _MIGRATIONS[:version]:
            con.executescript(script)
        con.execute(f"PRAGMA user_version = {version}")
        con.commit()
    finally:
        con.close()


def test_migration_from_v3_drops_the_unused_tables_and_keeps_data(tmp_path):
    db_path = tmp_path / "studio.db"
    _database_at_version(db_path, 3)
    assert set(DROPPED) <= _tables(db_path)
    con = sqlite3.connect(db_path)
    con.execute("INSERT INTO series (name) VALUES ('Kurtlar Vadisi Pusu')")
    con.execute("INSERT INTO translation_memory (series_id, source_language, target_language, source_term, "
                "target_term) VALUES (1, 'tr', 'ar', 'Polat', 'kept')")
    con.commit()
    con.close()

    db = Database(db_path)
    try:
        assert db.schema_version == SCHEMA_VERSION == 4
        remaining = _tables(db_path)
        assert not (set(DROPPED) & remaining)
        assert "translation_memory" in remaining and "settings" in remaining and "jobs" in remaining
        rows = db.conn.execute("SELECT name FROM series").fetchall()
        assert [row["name"] for row in rows] == ["Kurtlar Vadisi Pusu"]
        kept = db.conn.execute("SELECT source_term, target_term FROM translation_memory").fetchall()
        assert [tuple(row) for row in kept] == [("Polat", "kept")]
    finally:
        db.close()


def test_fresh_database_has_no_unused_table(tmp_path):
    db = Database(tmp_path / "fresh.db")
    try:
        assert db.schema_version == SCHEMA_VERSION
        assert not (set(DROPPED) & _tables(tmp_path / "fresh.db"))
        assert "translation_memory" in _tables(tmp_path / "fresh.db")
    finally:
        db.close()


def test_reopening_a_migrated_database_is_a_no_op(tmp_path):
    db_path = tmp_path / "studio.db"
    _database_at_version(db_path, 3)
    Database(db_path).close()
    db = Database(db_path)                 # the second open must not try to migrate again
    try:
        assert db.schema_version == SCHEMA_VERSION
    finally:
        db.close()
