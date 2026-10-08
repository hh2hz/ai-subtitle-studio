"""A read-only data folder must not stop the app (D-112): started from a restricted shell, `python -m app.main`
used to die with PermissionError on app.log and "unable to open database file"."""

import sqlite3

import pytest

from app.utils import paths as paths_module
from app.utils.paths import AppPaths, can_write, resolve_data_root


def test_can_write_reports_a_real_write(tmp_path):
    assert can_write(tmp_path / "fresh")
    assert can_write(tmp_path)                       # already exists


def test_can_write_is_false_when_the_probe_fails(tmp_path, monkeypatch):
    def boom(self, data):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("pathlib.Path.write_bytes", boom)
    assert can_write(tmp_path) is False


def test_the_preferred_root_is_used_when_writable(tmp_path):
    root, note = resolve_data_root(tmp_path / "data")

    assert root == tmp_path / "data" and note is None


def test_a_read_only_root_falls_back_to_the_project_folder(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked"
    fallback = tmp_path / "project"
    monkeypatch.setattr(paths_module, "can_write", lambda directory: directory != blocked)
    monkeypatch.setattr(paths_module, "project_root", lambda: fallback)

    root, note = resolve_data_root(blocked)

    assert root == fallback / ".appdata"
    assert note and str(blocked) in note and str(root) in note


def test_the_temp_folder_is_the_last_resort(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked"
    temp_root = tmp_path / "temp" / paths_module.APP_DIR_NAME
    monkeypatch.setattr(paths_module, "can_write", lambda directory: directory == temp_root)
    monkeypatch.setattr(paths_module, "project_root", lambda: None)
    monkeypatch.setenv("TEMP", str(tmp_path / "temp"))

    root, note = resolve_data_root(blocked)

    assert root == temp_root and note


def test_nothing_writable_reports_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_module, "can_write", lambda directory: False)
    monkeypatch.setattr(paths_module, "project_root", lambda: None)

    with pytest.raises(OSError) as excinfo:
        resolve_data_root(tmp_path / "blocked")

    assert "No writable data folder" in str(excinfo.value)


def test_models_are_reused_from_the_preferred_root(tmp_path):
    preferred = tmp_path / "preferred"
    (preferred / "models" / "whisper").mkdir(parents=True)
    fallback = AppPaths.from_root(tmp_path / "fallback")

    reused = fallback.with_models_from(preferred)

    assert reused.models_dir == preferred / "models"
    assert reused.root == fallback.root                       # everything writable stays in the fallback
    assert reused.jobs_dir == fallback.jobs_dir and reused.logs_dir == fallback.logs_dir


def test_a_root_without_models_keeps_its_own_folder(tmp_path):
    empty_preferred = tmp_path / "preferred"
    empty_preferred.mkdir()
    fallback = AppPaths.from_root(tmp_path / "fallback")

    assert fallback.with_models_from(empty_preferred).models_dir == tmp_path / "fallback" / "models"


def test_the_database_opens_when_wal_is_unavailable(tmp_path, monkeypatch, caplog):
    """A network share can refuse the WAL pragma; the app must still start (D-112)."""
    from app.database import database as database_module

    real_connect = sqlite3.connect

    class RefusingConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if "journal_mode" in sql:
                raise sqlite3.OperationalError("unable to open database file")
            return super().execute(sql, *args, **kwargs)

    def connect(path, *args, **kwargs):
        kwargs["factory"] = RefusingConnection
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(database_module.sqlite3, "connect", connect)
    with caplog.at_level("WARNING"):
        db = database_module.Database(tmp_path / "studio.db")
    try:
        assert db.schema_version == database_module.SCHEMA_VERSION     # migrations still ran
    finally:
        db.close()
    assert "WAL journal mode is unavailable" in caplog.text
