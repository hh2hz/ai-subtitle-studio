"""GUI job control with fake engines."""

import threading

import pytest

from app.core.modes import Mode
from app.core.pipeline import Pipeline
from app.ui.main_window import MainWindow, RuntimeContext
from tests.fakes import CPU_ASR, CPU_MT, Factory, FakeAsrEngine, FakeTranslator
from tests.media import make_tone_file


def _factory(jobs_dir, delay=0.0):
    def factory(config, cancel, progress, recorder, extras=None):
        return Pipeline(config, jobs_dir, [CPU_ASR], [CPU_MT],
                        Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine(delay=delay)}),
                        Factory({(CPU_MT.model, "cpu"): FakeTranslator()}),
                        progress=progress, cancel=cancel, recorder=recorder)
    return factory


@pytest.fixture
def window_factory(qtbot, translator, settings, db, tmp_path):
    def make(delay=0.0):
        runtime = RuntimeContext(db_path=db.path, pipeline_factory=_factory(tmp_path / "jobs", delay))
        w = MainWindow(translator, settings, runtime)
        w._show_error = lambda *a: None
        qtbot.addWidget(w)
        w.show()
        media = make_tone_file(tmp_path / "ep.m4a", seconds=10.0)
        w.input_edit.setText(str(media))
        w.source_combo.setCurrentIndex(w.source_combo.findData("tr"))
        w.output_edit.setText(str(tmp_path / "out"))
        return w
    return make


def test_job_completes_and_updates_ui(qtbot, window_factory, db):
    w = window_factory()
    assert w.start_job()
    runner = w._runner
    assert not w.input_edit.isEnabled()
    with qtbot.waitSignal(runner.finished, timeout=60000):
        pass
    qtbot.waitUntil(lambda: w._runner is None, timeout=5000)
    assert w.progress_bar.value() == 100
    assert w.open_output_button.isEnabled()
    assert w.input_edit.isEnabled()
    jobs = db.conn.execute("SELECT status FROM jobs").fetchall()
    assert [r["status"] for r in jobs] == ["completed"]
    assert w.review_button.isEnabled()
    w.open_review(w._last_output_dir)
    review = w._review_windows[-1]
    qtbot.addWidget(review)
    assert review.isVisible() and review.session.units


def test_cancel_job(qtbot, window_factory, db):
    w = window_factory(delay=0.5)
    assert w.start_job()
    runner = w._runner
    qtbot.waitUntil(lambda: w.progress_bar.value() > 0, timeout=30000)
    with qtbot.waitSignal(runner.cancelled, timeout=30000):
        w.start_button.click()          # acts as Cancel while running
    qtbot.waitUntil(lambda: w._runner is None, timeout=10000)
    assert w.start_button.text() == "Translate"
    assert [r["status"] for r in db.conn.execute("SELECT status FROM jobs")] == ["cancelled"]


def test_url_job_runs(qtbot, translator, settings, db, tmp_path):
    from tests.fakes import FakeDownloader

    media = make_tone_file(tmp_path / "src.m4a", seconds=6.0)

    def factory(config, cancel, progress, recorder, extras=None):
        assert extras["fetch_platform_subtitles"] is True and extras["llm_refine"] is False
        assert extras["translation_engine"] == "local" and extras["local_model"] == "translategemma-4b-q4km"
        assert extras["cookies_source"] == "" and extras["force_ipv4"] is False
        return Pipeline(config, tmp_path / "jobs", [CPU_ASR], [CPU_MT],
                        Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine()}),
                        Factory({(CPU_MT.model, "cpu"): FakeTranslator()}),
                        progress=progress, cancel=cancel, recorder=recorder, downloader=FakeDownloader(media))

    w = MainWindow(translator, settings, RuntimeContext(db_path=db.path, pipeline_factory=factory))
    w._show_error = lambda *a: None
    qtbot.addWidget(w)
    w.input_edit.setText("https://www.youtube.com/watch?v=x")
    w.source_combo.setCurrentIndex(w.source_combo.findData("tr"))
    w.output_edit.setText(str(tmp_path / "out"))
    w.season_spin.setValue(3)
    assert w.start_job()
    runner = w._runner
    with qtbot.waitSignal(runner.finished, timeout=60000):
        pass
    row = db.conn.execute("SELECT input_type, status, season, episode, series_id FROM jobs").fetchone()
    assert (row["input_type"], row["status"], row["season"], row["episode"]) == ("url", "completed", 3, 2)
    assert row["series_id"] is not None


def test_close_while_running_cancels_first(qtbot, window_factory):
    w = window_factory(delay=0.5)
    assert w.start_job()
    runner = w._runner
    qtbot.waitUntil(lambda: w.progress_bar.value() > 0, timeout=30000)
    if runner.isRunning():
        with qtbot.waitSignal(runner.finished, timeout=30000):
            w.close()
            assert w.isVisible() or not runner.isRunning()
    else:
        w.close()
    qtbot.waitUntil(lambda: not w.isVisible(), timeout=10000)
