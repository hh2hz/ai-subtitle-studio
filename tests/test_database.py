import sqlite3

import pytest

from app.database.database import SCHEMA_VERSION, Database, DatabaseError

EXPECTED_TABLES = {
    "settings", "series", "jobs", "job_stages", "translation_memory",
}


def _tables(db):
    rows = db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {r["name"] for r in rows}


def test_creates_schema(tmp_path):
    path = tmp_path / "sub" / "studio.db"
    with Database(path) as db:
        assert path.is_file()
        assert db.schema_version == SCHEMA_VERSION
        assert EXPECTED_TABLES <= _tables(db)
        assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_reopen_is_idempotent(tmp_path):
    path = tmp_path / "studio.db"
    Database(path).close()
    with Database(path) as db:
        assert db.schema_version == SCHEMA_VERSION


def test_rejects_newer_schema(tmp_path):
    path = tmp_path / "studio.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(DatabaseError):
        Database(path)


def test_foreign_key_cascade(db):
    c = db.conn
    with c:
        c.execute("INSERT INTO jobs (input_type, input_value, source_language, target_language, mode) "
                  "VALUES ('file', 'x.mp4', 'tr', 'ar', 'balanced')")
        c.execute("INSERT INTO job_stages (job_id, stage, status) VALUES (1, 'audio', 'completed')")
        c.execute("DELETE FROM jobs WHERE id = 1")
    assert c.execute("SELECT COUNT(*) FROM job_stages").fetchone()[0] == 0


def test_check_constraints(db):
    with pytest.raises(sqlite3.IntegrityError):
        with db.conn:
            db.conn.execute("INSERT INTO jobs (input_type, input_value, source_language, target_language, mode) "
                            "VALUES ('ftp', 'x', 'tr', 'ar', 'fast')")


def test_global_translation_memory_unique(db):
    sql = ("INSERT INTO translation_memory (series_id, source_language, target_language, source_term, target_term) "
           "VALUES (NULL, 'tr', 'ar', 'abi', 'x')")
    with db.conn:
        db.conn.execute(sql)
    with pytest.raises(sqlite3.IntegrityError):
        with db.conn:
            db.conn.execute(sql)
