"""Tests for Phase 6.2: job history and resume."""

from pathlib import Path
from unittest.mock import MagicMock

from app.core.errors import PipelineError
from app.core.pipeline import Pipeline
from app.database.database import Database
from app.database.jobs import JobsRepo
from app.ui.history_window import JobHistoryDialog
from app.ui.main_window import MainWindow, RuntimeContext
from tests.fakes import CPU_ASR, CPU_MT, Factory, FakeAsrEngine, FakeTranslator
from tests.media import make_tone_file


def _history_factory(jobs_dir, fail_on_first_run=None):
    """Pipeline factory that tracks call counts to verify stage caching on resume."""
    asr_runs = {"count": 0}

    class CountingAsrEngine(FakeAsrEngine):
        def transcribe(self, *args, **kwargs):
            asr_runs["count"] += 1
            return super().transcribe(*args, **kwargs)

    def factory(config, cancel, progress, recorder, extras=None):
        if fail_on_first_run and fail_on_first_run.get("fail"):
            fail_on_first_run["fail"] = False
            raise PipelineError("Simulated first run failure", "error.pipeline_failed")
        return Pipeline(
            config,
            jobs_dir,
            [CPU_ASR],
            [CPU_MT],
            Factory({(CPU_ASR.model, "cpu"): CountingAsrEngine()}),
            Factory({(CPU_MT.model, "cpu"): FakeTranslator()}),
            progress=progress,
            cancel=cancel,
            recorder=recorder,
        )

    return factory, asr_runs


def test_jobs_repo_recent_and_prepare_resume(db):
    repo = JobsRepo(db)
    s_id = repo.ensure_series("Test Series")
    j1 = repo.create(
        input_type="file", input_value="C:/test1.mp4", source_language="tr",
        target_language="ar", mode="balanced", output_dir="C:/out1"
    )
    repo.set_series(j1, {"series_name": "Test Series", "season": 1, "episode": 5, "episode_title": "Pilot"})
    repo.update(j1, status="failed", error="Error details")

    recent = repo.recent()
    assert len(recent) >= 1
    found = next(j for j in recent if j["id"] == j1)
    assert found["series_name"] == "Test Series"
    assert found["status"] == "failed"

    # Test prepare_resume
    resumed = repo.prepare_resume(j1, extras={"llm_refine": True})
    assert resumed["id"] == j1
    assert resumed["status"] == "pending"
    assert resumed["error"] is None
    assert resumed["queue_config"] is not None
    assert resumed["queue_order"] == 1


def test_job_history_dialog_listing_and_open_folder(qtbot, translator, settings, db, tmp_path):
    repo = JobsRepo(db)
    out_dir = tmp_path / "completed_out"
    out_dir.mkdir()

    j_completed = repo.create(
        input_type="file", input_value=str(tmp_path / "ep01.mp4"),
        source_language="tr", target_language="ar", mode="balanced", output_dir=str(out_dir)
    )
    repo.update(j_completed, status="completed")

    j_failed = repo.create(
        input_type="file", input_value=str(tmp_path / "ep02.mp4"),
        source_language="en", target_language="ar", mode="fast", output_dir=None
    )
    repo.update(j_failed, status="failed", error="Failed")

    runtime = RuntimeContext(
        db_path=db.path,
        pipeline_factory=lambda *a, **k: None,
    )
    w = MainWindow(translator, settings, runtime)
    w._show_error = lambda *a: None
    qtbot.addWidget(w)

    dlg = w.open_history()
    qtbot.addWidget(dlg)

    assert dlg.table.rowCount() == 2
    # Row 0 is newest (j_failed)
    assert dlg.table.item(0, 0).text() == str(j_failed)
    assert dlg.table.item(0, 4).text() == "Fast"
    assert dlg.table.item(0, 5).text() == "Failed"

    # Row 1 is j_completed
    assert dlg.table.item(1, 0).text() == str(j_completed)
    assert dlg.table.item(1, 4).text() == "Balanced"
    assert dlg.table.item(1, 5).text() == "Completed"

    # Actions widget on row 1: open folder button should be enabled
    actions_row1 = dlg.table.cellWidget(1, 6)
    buttons_row1 = actions_row1.findChildren(type(w.start_button))
    resume_btn1, open_btn1 = buttons_row1[0], buttons_row1[1]
    assert resume_btn1.isEnabled()
    assert open_btn1.isEnabled()

    # Click open folder triggers _open_dir
    w._open_dir = MagicMock()
    open_btn1.click()
    w._open_dir.assert_called_once_with(str(out_dir))

    # Actions widget on row 0: open folder button should be disabled (output_dir is None)
    actions_row0 = dlg.table.cellWidget(0, 6)
    buttons_row0 = actions_row0.findChildren(type(w.start_button))
    assert not buttons_row0[1].isEnabled()


def test_job_history_resume_execution_and_cache_continuation(qtbot, translator, settings, db, tmp_path):
    media = make_tone_file(tmp_path / "resume_test.mp4", seconds=2.0)
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    out_dir = tmp_path / "out"

    fail_control = {"fail": True}
    factory, asr_runs = _history_factory(jobs_dir, fail_on_first_run=fail_control)

    runtime = RuntimeContext(
        db_path=db.path,
        pipeline_factory=factory,
    )
    w = MainWindow(translator, settings, runtime)
    w._show_error = lambda *a: None
    qtbot.addWidget(w)
    w.output_edit.setText(str(out_dir))

    # Add file and run: it should fail due to fail_control
    job_ids = w.add_files_to_queue([media])
    assert len(job_ids) == 1
    job_id = job_ids[0]

    assert w.start_queue()
    qtbot.waitUntil(lambda: w._runner is None and w.queue_table.item(0, 2).text() == "Failed", timeout=30000)

    repo = JobsRepo(db)
    assert repo.get(job_id)["status"] == "failed"
    assert asr_runs["count"] == 0

    # Open history dialog and click Resume
    dlg = w.open_history()
    qtbot.addWidget(dlg)

    actions = dlg.table.cellWidget(0, 6)
    resume_btn = actions.findChildren(type(w.start_button))[0]
    resume_btn.click()

    # Wait until resumed job finishes successfully
    qtbot.waitUntil(lambda: w._runner is None and w.queue_table.item(0, 2).text() == "Completed", timeout=60000)

    # Database status should now be completed
    assert repo.get(job_id)["status"] == "completed"
    assert asr_runs["count"] == 1

    # Resume the completed job again: it must finish using cached results!
    dlg.reload()
    actions_completed = dlg.table.cellWidget(0, 6)
    resume_completed_btn = actions_completed.findChildren(type(w.start_button))[0]
    resume_completed_btn.click()

    qtbot.waitUntil(lambda: w._runner is None and w.queue_table.item(0, 2).text() == "Completed", timeout=60000)
    assert repo.get(job_id)["status"] == "completed"
    # ASR should have hit cache, so ASR engine was not re-run (count stays 1)!
    assert asr_runs["count"] == 1


def test_job_history_menu_and_header_button(qtbot, translator, settings, db):
    runtime = RuntimeContext(
        db_path=db.path,
        pipeline_factory=lambda *a, **k: None,
    )
    w = MainWindow(translator, settings, runtime)
    w._show_error = lambda *a: None
    qtbot.addWidget(w)

    # 1. Trigger from menu action
    w.history_action.trigger()
    assert hasattr(w, "_history_dialog")
    assert w._history_dialog is not None
    assert w._history_dialog.isVisible()
    w._history_dialog.close()

    # 2. Trigger from header button
    w.history_button.click()
    assert w._history_dialog.isVisible()
    w._history_dialog.close()
