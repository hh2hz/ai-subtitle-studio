"""Tests for Phase 6.1: batch queue (files, folders, playlists, order, failure, cancel, restart)."""

from pathlib import Path
from unittest.mock import patch

from app.core.errors import PipelineError
from app.core.pipeline import Pipeline
from app.database.database import Database
from app.database.jobs import JobsRepo
from app.ui.main_window import MainWindow, RuntimeContext
from tests.fakes import CPU_ASR, CPU_MT, Factory, FakeAsrEngine, FakeDownloader, FakeTranslator
from tests.media import make_tone_file


def _factory(jobs_dir, delay=0.0, fail_paths=()):
    def factory(config, cancel, progress, recorder, extras=None):
        if config.input_path and any(fp in str(config.input_path) for fp in fail_paths):
            raise PipelineError("Simulated pipeline failure", "error.pipeline_failed")
        return Pipeline(
            config,
            jobs_dir,
            [CPU_ASR],
            [CPU_MT],
            Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine(delay=delay)}),
            Factory({(CPU_MT.model, "cpu"): FakeTranslator()}),
            progress=progress,
            cancel=cancel,
            recorder=recorder,
        )
    return factory


def test_batch_queue_add_files_folder_playlist(qtbot, translator, settings, db, tmp_path):
    media1 = make_tone_file(tmp_path / "ep01.mp4", seconds=2.0)
    media2 = make_tone_file(tmp_path / "ep02.mp4", seconds=2.0)

    sub = tmp_path / "season1"
    sub.mkdir()
    media3 = make_tone_file(sub / "ep03.mkv", seconds=2.0)
    media4 = make_tone_file(sub / "ep04.avi", seconds=2.0)
    (sub / "ignore.txt").write_text("not media", encoding="utf-8")

    playlist_entries = [
        {"url": "https://www.youtube.com/watch?v=pl1", "title": "Online Ep 1"},
        {"url": "https://www.youtube.com/watch?v=pl2", "title": "Online Ep 2"},
    ]
    fake_dl = FakeDownloader(media1, playlist_entries=playlist_entries)

    runtime = RuntimeContext(
        db_path=db.path,
        pipeline_factory=_factory(tmp_path / "jobs"),
        downloader=fake_dl,
    )
    w = MainWindow(translator, settings, runtime)
    w._show_error = lambda *a: None
    qtbot.addWidget(w)

    # 1. Add files
    ids1 = w.add_files_to_queue([media1, media2])
    assert len(ids1) == 2
    assert w.queue_table.rowCount() == 2

    # 2. Add folder (should find ep03.mkv and ep04.avi, ignoring ignore.txt)
    ids2 = w.add_folder_to_queue(sub)
    assert len(ids2) == 2
    assert w.queue_table.rowCount() == 4

    # 3. Add playlist
    ids3 = w.add_playlist_to_queue("https://www.youtube.com/playlist?list=PL_TEST")
    assert len(ids3) == 2
    assert w.queue_table.rowCount() == 6

    # Verify database state
    repo = JobsRepo(db)
    queue = repo.get_queue()
    assert len(queue) == 6
    assert [j["status"] for j in queue] == ["pending"] * 6
    assert [j["queue_order"] for j in queue] == [1, 2, 3, 4, 5, 6]


def test_batch_queue_order(qtbot, translator, settings, db, tmp_path):
    media1 = make_tone_file(tmp_path / "first.mp4", seconds=2.0)
    media2 = make_tone_file(tmp_path / "second.mp4", seconds=2.0)

    runtime = RuntimeContext(
        db_path=db.path,
        pipeline_factory=_factory(tmp_path / "jobs"),
    )
    w = MainWindow(translator, settings, runtime)
    w._show_error = lambda *a: None
    qtbot.addWidget(w)
    w.output_edit.setText(str(tmp_path / "out"))

    w.add_files_to_queue([media1, media2])
    assert w.queue_table.rowCount() == 2

    assert w.start_queue()
    # Wait until all jobs in the queue finish
    qtbot.waitUntil(lambda: not w.is_running() and w.queue_table.item(1, 2).text() == "Completed", timeout=60000)

    repo = JobsRepo(db)
    queue = repo.get_queue()
    assert [j["status"] for j in queue] == ["completed", "completed"]
    # Check execution order: first finished before or at same time as second
    assert queue[0]["updated_at"] <= queue[1]["updated_at"]
    assert w.queue_table.item(0, 2).text() == "Completed"
    assert w.queue_table.item(1, 2).text() == "Completed"


def test_batch_queue_failure_continues(qtbot, translator, settings, db, tmp_path):
    media1 = make_tone_file(tmp_path / "broken.mp4", seconds=2.0)
    media2 = make_tone_file(tmp_path / "working.mp4", seconds=2.0)

    runtime = RuntimeContext(
        db_path=db.path,
        pipeline_factory=_factory(tmp_path / "jobs", fail_paths=("broken",)),
    )
    w = MainWindow(translator, settings, runtime)
    w._show_error = lambda *a: None
    qtbot.addWidget(w)
    w.output_edit.setText(str(tmp_path / "out"))

    w.add_files_to_queue([media1, media2])
    assert w.start_queue()

    # The queue must finish: job 1 failed, but job 2 ran and completed!
    qtbot.waitUntil(lambda: not w.is_running() and w.queue_table.item(1, 2).text() == "Completed", timeout=60000)

    repo = JobsRepo(db)
    queue = repo.get_queue()
    assert queue[0]["status"] == "failed"
    assert queue[1]["status"] == "completed"
    assert w.queue_table.item(0, 2).text() == "Failed"
    assert w.queue_table.item(1, 2).text() == "Completed"


def test_batch_queue_cancel(qtbot, translator, settings, db, tmp_path):
    media1 = make_tone_file(tmp_path / "long1.mp4", seconds=4.0)
    media2 = make_tone_file(tmp_path / "queued2.mp4", seconds=2.0)
    media3 = make_tone_file(tmp_path / "queued3.mp4", seconds=2.0)

    runtime = RuntimeContext(
        db_path=db.path,
        pipeline_factory=_factory(tmp_path / "jobs", delay=0.5),
    )
    w = MainWindow(translator, settings, runtime)
    w._show_error = lambda *a: None
    qtbot.addWidget(w)
    w.output_edit.setText(str(tmp_path / "out"))

    w.add_files_to_queue([media1, media2, media3])
    assert w.start_queue()

    # Wait until job 1 is running
    qtbot.waitUntil(lambda: w.is_running() and "Running" in w.queue_table.item(0, 2).text(), timeout=30000)

    # Cancel the entire queue
    w.cancel_queue()
    # Wait for the outcome itself, not for the intermediate "running" flag: cancelling the queued jobs happens
    # right after the running one stops, and under load that had already been observed as finished (D-106).
    qtbot.waitUntil(lambda: not w.is_running() and all(j["status"] == "cancelled" for j in JobsRepo(db).get_queue()),
                    timeout=30000)

    repo = JobsRepo(db)
    queue = repo.get_queue()
    # Job 1 was cancelled during execution, remaining queued jobs cancelled
    assert all(j["status"] == "cancelled" for j in queue)
    assert w.queue_table.item(0, 2).text() == "Cancelled"
    assert w.queue_table.item(1, 2).text() == "Cancelled"
    assert w.queue_table.item(2, 2).text() == "Cancelled"


def test_batch_queue_restart_restore(qtbot, translator, settings, db, tmp_path):
    repo = JobsRepo(db)
    jid1 = repo.enqueue(
        input_type="file",
        input_value=str(tmp_path / "file1.mp4"),
        source_language="tr",
        target_language="ar",
        mode="fast",
        title="Episode 1",
    )
    jid2 = repo.enqueue(
        input_type="file",
        input_value=str(tmp_path / "file2.mp4"),
        source_language="tr",
        target_language="ar",
        mode="fast",
        title="Episode 2",
    )

    # Simulate app start
    runtime = RuntimeContext(
        db_path=db.path,
        pipeline_factory=_factory(tmp_path / "jobs"),
    )
    w = MainWindow(translator, settings, runtime)
    qtbot.addWidget(w)

    # Verify table restored from DB
    assert w.queue_table.rowCount() == 2
    assert w.queue_table.item(0, 0).text() == "1"
    assert w.queue_table.item(0, 1).text() == "Episode 1"
    assert w.queue_table.item(0, 2).text() == "Pending"
    assert w.queue_table.item(1, 0).text() == "2"
    assert w.queue_table.item(1, 1).text() == "Episode 2"
    assert w.queue_table.item(1, 2).text() == "Pending"


def test_batch_queue_open_output_and_clear(qtbot, translator, settings, db, tmp_path):
    media = make_tone_file(tmp_path / "done.mp4", seconds=2.0)
    out_dir = tmp_path / "out"

    runtime = RuntimeContext(
        db_path=db.path,
        pipeline_factory=_factory(tmp_path / "jobs"),
    )
    w = MainWindow(translator, settings, runtime)
    w._show_error = lambda *a: None
    qtbot.addWidget(w)
    w.output_edit.setText(str(out_dir))

    w.add_files_to_queue([media])
    assert w.start_queue()
    qtbot.waitUntil(lambda: not w.is_running() and w.queue_table.item(0, 2).text() == "Completed", timeout=60000)

    # Check that open output button in table is enabled
    btn = w.queue_table.cellWidget(0, 4)
    assert btn is not None
    assert btn.isEnabled()

    with patch("PySide6.QtGui.QDesktopServices.openUrl") as mock_open:
        btn.click()
        assert mock_open.called

    # Clear completed queue
    w.clear_completed_queue()
    assert w.queue_table.rowCount() == 0
    assert len(JobsRepo(db).get_queue()) == 0
    assert len(JobsRepo(db).get_queue()) == 0


def test_queue_never_stores_api_keys(tmp_path):
    from app.database.database import Database
    from app.database.jobs import JobsRepo

    repo = JobsRepo(Database(tmp_path / "q.db"))
    jid = repo.enqueue(input_type="file", input_value="a.mp4", source_language="tr", target_language="ar",
                       mode="balanced", extras={"llm_refine": True, "subdl_api_key": "SECRET-123"})
    row = next(r for r in repo.get_queue() if r["id"] == jid)
    assert "SECRET-123" not in row["queue_config"] and '"llm_refine": true' in row["queue_config"]
