"""Tests for Phase 6.6: progress with ETA (per-stage and total queue ETA, Qt offscreen)."""

import time
from unittest.mock import patch

from PySide6.QtCore import Qt

from app.core.pipeline import Pipeline
from app.database.jobs import JobsRepo
from app.services import eta
from app.ui.main_window import MainWindow, RuntimeContext
from tests.fakes import CPU_ASR, CPU_MT, Factory, FakeAsrEngine, FakeTranslator
from tests.media import make_tone_file


def _stats(media: float = 600.0, total: float = 300.0, stages: dict | None = None) -> dict:
    """What a finished 10 minute job leaves in the database on this machine."""
    if stages is None:
        stages = {
            "download": {"cached": False, "elapsed_s": 30.0, "real_time_factor": 0.05},
            "audio": {"cached": False, "elapsed_s": 6.0, "real_time_factor": 0.01},
            "transcribe": {"cached": False, "elapsed_s": 60.0, "real_time_factor": 0.10},
            "subtitles": {"cached": False, "elapsed_s": 3.0},
            "translate": {"cached": False, "elapsed_s": 42.0, "real_time_factor": 0.07},
            "refine": {"cached": False, "elapsed_s": 120.0, "real_time_factor": 0.20},
            "export": {"cached": False, "elapsed_s": 5.0},
        }
    return {"media_duration_s": media, "total_elapsed_s": total, "stages": stages}


def _finished_job(repo: JobsRepo, stats: dict) -> int:
    job_id = repo.create(input_type="file", input_value="ep.mp4", source_language="tr",
                         target_language="ar", mode="fast", output_dir=None)
    repo.update(job_id, status="completed", stats=stats)
    return job_id


class _FakeRunner:
    """Stands in for a running JobRunner so the window treats a job as in progress."""

    def __init__(self, job_id: int):
        self.job_id = job_id

    def isRunning(self) -> bool:
        return True

    def request_cancel(self) -> None:
        pass


# -- estimates (no Qt) ------------------------------------------------------------------------

def test_timings_are_medians_of_finished_jobs():
    first = _stats(stages={"transcribe": {"cached": False, "elapsed_s": 10.0, "real_time_factor": 0.1},
                           "export": {"cached": False, "elapsed_s": 4.0}})
    second = _stats(stages={"transcribe": {"cached": False, "elapsed_s": 30.0, "real_time_factor": 0.3},
                            "export": {"cached": False, "elapsed_s": 8.0},
                            "refine": {"cached": True, "elapsed_s": 999.0},
                            "subtitles": {"skipped": "no providers"}})
    timings = eta.timings_from_stats([first, second])
    assert timings.samples == 2
    assert timings.stage_rtf["transcribe"] == 0.2
    assert timings.stage_seconds["export"] == 6.0
    assert "refine" not in timings.stage_seconds          # cached work is not the normal case
    assert "subtitles" not in timings.stage_seconds       # skipped stages never take time
    assert timings.job_ratio == 0.5
    assert timings.job_seconds == 300.0
    assert eta.timings_from_stats([]) == eta.EMPTY_TIMINGS


def test_stage_and_job_remaining_use_real_time_factors():
    timings = eta.timings_from_stats([_stats()])
    # half of the 600 s episode left in a stage with a real-time factor of 0.10 -> 30 s
    assert eta.stage_remaining(timings, "transcribe", 600.0, 0.5) == 30.0
    assert eta.stage_remaining(timings, "transcribe", None, 0.5) is None     # media length unknown
    assert eta.stage_remaining(eta.EMPTY_TIMINGS, "transcribe", 600.0, 0.5) is None
    # this stage (30 s) + subtitles (3 s) + translate (0.07 x 600) + refine (0.20 x 600) + export (5 s)
    assert eta.job_remaining(timings, "transcribe", 600.0, 0.5) == 200.0
    assert eta.job_remaining(timings, "transcribe", None, 0.5) == 200.0     # fixed stages only
    assert eta.job_remaining(eta.EMPTY_TIMINGS, "transcribe", 600.0, 0.5) is None
    assert eta.job_remaining(timings, "not a stage", 600.0, 0.5) is None


def test_pending_job_seconds_and_format_eta():
    timings = eta.timings_from_stats([_stats()])
    assert eta.pending_job_seconds(timings, 1200.0) == 600.0   # length known -> media length x ratio
    assert eta.pending_job_seconds(timings, None) == 300.0     # length unknown -> median job length
    assert eta.pending_job_seconds(eta.EMPTY_TIMINGS, None) is None
    assert eta.format_eta(9) == "00:09"
    assert eta.format_eta(95) == "01:35"
    assert eta.format_eta(3725) == "1:02:05"


def test_recent_stats_reads_only_finished_jobs(db):
    repo = JobsRepo(db)
    done = _finished_job(repo, _stats())
    pending = repo.enqueue(input_type="file", input_value="b.mp4", source_language="tr",
                           target_language="ar", mode="fast")
    rows = repo.recent_stats()
    assert len(rows) == 1 and rows[0]["media_duration_s"] == 600.0
    assert repo.recent_stats(limit=0) == []
    assert done != pending


# -- window (Qt, offscreen) -------------------------------------------------------------------

def test_media_duration_message_only_sets_the_duration(qtbot, translator, settings, db):
    w = MainWindow(translator, settings, RuntimeContext(db_path=db.path, pipeline_factory=None))
    qtbot.addWidget(w)
    w.stage_label.setText("Transcribing 10%")
    w.progress_bar.setValue(10)
    w._on_progress("", 0.0, 0.0, "media_duration:600")
    assert w._job_media_duration == 600.0
    assert w.stage_label.text() == "Transcribing 10%"      # a side channel carries no stage progress
    assert w.progress_bar.value() == 10
    w._on_progress("", 0.0, 0.0, "media_duration:not-a-number")
    assert w._job_media_duration == 600.0                  # a broken message keeps the last good value


def test_stage_and_job_eta_from_stored_real_time_factors(qtbot, translator, settings, db):
    repo = JobsRepo(db)
    _finished_job(repo, _stats())
    job_id = repo.enqueue(input_type="file", input_value="ep.mp4", source_language="tr",
                          target_language="ar", mode="fast", title="Episode 1")

    w = MainWindow(translator, settings, RuntimeContext(db_path=db.path, pipeline_factory=None))
    qtbot.addWidget(w)
    assert w._timings.samples == 1                         # read from the finished jobs on this machine

    w._runner = _FakeRunner(job_id)
    w._job_started = time.monotonic()
    w._job_media_duration = 600.0
    w._on_progress("transcribe", 0.5, 0.4, "")

    # Per-stage ETA: 600 s of media, real-time factor 0.10, half of the stage left -> 30 s.
    assert w.stage_label.text() == translator.t(
        "status.stage_eta", stage=translator.t("stage.transcribe"), percent=50, eta="00:30")
    # Whole job left: 30 s here plus subtitles 3 s, translate 42 s, refine 120 s, export 5 s = 200 s.
    row = next(r for r in range(w.queue_table.rowCount())
               if w.queue_table.item(r, 1).data(Qt.ItemDataRole.UserRole) == job_id)
    assert w.queue_table.item(row, 2).text() == translator.t("queue.status_running") + " (40%)"
    assert w.queue_table.item(row, 3).text() == "~03:20"
    w._runner = None                                        # not a real thread: nothing to cancel


def test_queue_eta_sums_the_pending_jobs(qtbot, translator, settings, db, tmp_path):
    repo = JobsRepo(db)
    _finished_job(repo, _stats())                       # 600 s of media took 300 s -> half the length
    media = [make_tone_file(tmp_path / "ep01.mp4", seconds=2.0),
             make_tone_file(tmp_path / "ep02.mp4", seconds=2.0)]

    w = MainWindow(translator, settings, RuntimeContext(db_path=db.path, pipeline_factory=None))
    qtbot.addWidget(w)
    settings.set("parallel_jobs", 1)
    with patch("app.ui.main_window.eta.media_duration", return_value=100.0):
        w.add_files_to_queue(media)
        w._update_queue_eta()
    # One video after the other: two pending jobs of 100 s of media each, learned ratio 0.5 -> 100 s for the queue.
    assert w.queue_eta_label.text() == translator.t("queue.total_eta", eta="01:40")

    # In parallel (D-120) the two jobs overlap: no resource is busier than one whole job (GPU 2 x 17 s, AI 2 x 20 s
    # over two slots), so the queue takes as long as its longest job.
    settings.set("parallel_jobs", 4)
    w._update_queue_eta()
    assert w.queue_eta_label.text() == translator.t("queue.total_eta", eta="00:50")


def test_queue_eta_unknown_without_history_and_cleared_when_idle(qtbot, translator, settings, db, tmp_path):
    media = make_tone_file(tmp_path / "ep.mp4", seconds=2.0)
    w = MainWindow(translator, settings, RuntimeContext(db_path=db.path, pipeline_factory=None))
    qtbot.addWidget(w)
    w.add_files_to_queue([media])
    assert w.queue_eta_label.text() == translator.t("queue.total_eta", eta="--:--")

    w.cancel_queue()                                      # nothing has ever finished -> no estimate
    assert w.queue_table.item(0, 2).text() == translator.t("queue.status_cancelled")
    assert w.queue_eta_label.text() == ""                 # idle queue has no ETA at all


def test_real_job_feeds_the_next_eta(qtbot, translator, settings, db, tmp_path):
    def factory(config, cancel, progress, recorder, extras=None):
        return Pipeline(config, tmp_path / "jobs", [CPU_ASR], [CPU_MT],
                        Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine()}),
                        Factory({(CPU_MT.model, "cpu"): FakeTranslator()}),
                        progress=progress, cancel=cancel, recorder=recorder)

    _finished_job(JobsRepo(db), _stats())                # known history so the estimates have data
    first = make_tone_file(tmp_path / "ep01.mp4", seconds=2.0)
    second = make_tone_file(tmp_path / "ep02.mp4", seconds=4.0)

    w = MainWindow(translator, settings, RuntimeContext(db_path=db.path, pipeline_factory=factory))
    w._show_error = lambda *a: None
    qtbot.addWidget(w)
    w.output_edit.setText(str(tmp_path / "out"))
    w.add_files_to_queue([first, second])
    assert w.start_queue()

    qtbot.waitUntil(lambda: "left" in w.stage_label.text(), timeout=30000)   # per-stage ETA shown
    qtbot.waitUntil(lambda: not w.is_running() and w.queue_table.item(0, 2).text() == "Completed",
                    timeout=60000)
    assert w._job_media_duration and w._job_media_duration > 0   # the pipeline announced the length
    assert w._timings.samples >= 1                                # the finished job is now history

    third = make_tone_file(tmp_path / "ep03.mp4", seconds=6.0)
    w.add_files_to_queue([third])
    assert w.queue_eta_label.text() and "--:--" not in w.queue_eta_label.text()
