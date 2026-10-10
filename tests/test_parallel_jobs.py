"""Several jobs at once (D-120): resource gates, the pipeline's use of them, the speaker-detection child process,
the shared AI provider state and the window's job slots."""

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app.core import brief as br
from app.core import llm_providers as lp
from app.core.errors import JobCancelled
from app.core.llm_providers import HealthStore, ProviderCallError, ProviderPool, SharedRouteState
from app.core.modes import Mode
from app.core.pipeline import JobConfig, Pipeline
from app.database.jobs import JobsRepo
from app.services import diarize_worker, eta
from app.services.resources import GPU, ResourceGate, ResourceSet
from app.ui.main_window import MainWindow, RuntimeContext
from tests.fakes import CPU_ASR, CPU_MT, Factory, FakeAsrEngine, FakeDownloader, FakeTranslator
from tests.media import make_tone_file


# -- resource gates ----------------------------------------------------------------------------------------------

def _wait_for(predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


def test_gate_serves_waiters_by_priority_not_by_arrival():
    gate = ResourceGate("gpu", 1)
    gate.acquire(priority=0)
    order = []

    def worker(priority):
        gate.acquire(priority=priority)
        order.append(priority)
        gate.release()

    threads = []
    for priority in (3, 1, 2):                    # the last episode asks first
        thread = threading.Thread(target=worker, args=(priority,))
        thread.start()
        threads.append(thread)
        _wait_for(lambda n=len(threads): gate.waiting == n)
    gate.release()
    for thread in threads:
        thread.join(5)
    assert order == [1, 2, 3]
    assert gate.in_use == 0 and gate.waiting == 0


def test_gate_capacity_two_lets_two_in_and_reports_the_wait():
    gate = ResourceGate("ai", 2)
    gate.acquire(1)
    gate.acquire(2)
    waited = threading.Event()
    got = threading.Event()

    def third():
        gate.acquire(3, on_wait=waited.set)
        got.set()

    thread = threading.Thread(target=third)
    thread.start()
    assert waited.wait(5) and not got.is_set()
    gate.release()
    assert got.wait(5)
    gate.release()
    gate.release()
    thread.join(5)


def test_cancel_ends_the_wait_and_frees_the_place():
    gate = ResourceGate("cpu", 1)
    gate.acquire(0)
    cancel = threading.Event()
    errors = []

    def waiter():
        try:
            gate.acquire(1, cancel=cancel)
        except JobCancelled as exc:
            errors.append(exc)

    thread = threading.Thread(target=waiter)
    thread.start()
    _wait_for(lambda: gate.waiting == 1)
    cancel.set()
    thread.join(5)
    assert errors and gate.waiting == 0
    gate.release()
    with gate.hold(5):                            # nothing is stuck behind the cancelled ticket
        assert gate.in_use == 1


# -- pipeline ----------------------------------------------------------------------------------------------------

class _TimedAsr(FakeAsrEngine):
    """Records when each transcription call ran."""

    def __init__(self, log, label, **kw):
        super().__init__(**kw)
        self._log = log
        self._label = label

    def transcribe(self, audio, language, offset, initial_prompt):
        start = time.monotonic()
        yield from super().transcribe(audio, language, offset, initial_prompt)
        self._log.append(("asr", self._label, start, time.monotonic()))


class _TimedDiarizer:
    def __init__(self, log, label, seconds=0.6):
        self._log = log
        self._label = label
        self._seconds = seconds

    def __call__(self, audio, windows, partial, progress=None, cancel=None):
        start = time.monotonic()
        time.sleep(self._seconds)
        progress(1.0)
        self._log.append(("diarize", self._label, start, time.monotonic()))
        return [(0.0, 5.0, 0)]


def _pipeline(tmp_path, media, label, log, resources, priority):
    config = JobConfig(input_path=media, source_language="tr", target_language="ar", mode=Mode.BALANCED,
                       output_dir=tmp_path / "out")
    return Pipeline(config, tmp_path / "jobs", [CPU_ASR], [CPU_MT],
                    Factory({(CPU_ASR.model, "cpu"): _TimedAsr(log, label, delay=0.1)}),
                    Factory({(CPU_MT.model, "cpu"): FakeTranslator()}),
                    diarizer=_TimedDiarizer(log, label), resources=resources, priority=priority)


def _overlap(a, b):
    return a[2] < b[3] and b[2] < a[3]


def test_two_jobs_never_share_the_gpu_and_speakers_run_next_to_the_transcription(tmp_path):
    media = [make_tone_file(tmp_path / f"ep{i}.mp4", seconds=8.0 + i) for i in (1, 2)]
    resources = ResourceSet()
    log, errors = [], []

    def run(index):
        try:
            _pipeline(tmp_path, media[index], index, log, resources, priority=index).run()
        except Exception as exc:        # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors

    def spans(kind, label):
        return [e for e in log if e[0] == kind and e[1] == label]

    asr = [(min(e[2] for e in spans("asr", i)), max(e[3] for e in spans("asr", i))) for i in (0, 1)]
    assert not _overlap(("", "", *asr[0]), ("", "", *asr[1]))           # one job on the GPU at a time
    diar = [spans("diarize", i)[0] for i in (0, 1)]
    assert not _overlap(diar[0], diar[1])                               # one job on the CPU at a time
    # Speaker detection needs only the audio, so at least one job ran it while its own speech recognition ran.
    assert any(_overlap(diar[i], ("", "", *asr[i])) for i in (0, 1))


def test_the_same_video_twice_runs_one_after_the_other(tmp_path):
    media = make_tone_file(tmp_path / "ep.mp4", seconds=6.0)
    resources = ResourceSet()
    log, results, errors = [], [], []

    def run(index):
        try:
            results.append(_pipeline(tmp_path, media, index, log, resources, priority=index).run())
        except Exception as exc:        # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors and len(results) == 2
    assert all(Path(r.outputs["target_srt"]).is_file() for r in results)


def test_waiting_for_a_busy_resource_is_reported(tmp_path):
    media = make_tone_file(tmp_path / "ep.mp4", seconds=4.0)
    resources = ResourceSet()
    messages = []
    config = JobConfig(input_path=media, source_language="tr", target_language="ar", mode=Mode.BALANCED,
                       output_dir=tmp_path / "out")
    pipeline = Pipeline(config, tmp_path / "jobs", [CPU_ASR], [CPU_MT],
                        Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine()}),
                        Factory({(CPU_MT.model, "cpu"): FakeTranslator()}),
                        progress=lambda *a: messages.append(a[3]), resources=resources)
    resources.gate(GPU).acquire(-1)                 # another job holds the GPU
    thread = threading.Thread(target=pipeline.run)
    thread.start()
    _wait_for(lambda: "waiting:gpu:transcribe" in messages)
    resources.gate(GPU).release()
    thread.join(30)
    assert any(m == "waiting:gpu:transcribe" for m in messages)


# -- speaker detection child process ---------------------------------------------------------------------------

_CHILD_OK = """
import json, sys
spec = json.load(open(sys.argv[1], encoding="utf-8"))
import numpy as np
audio = np.load(spec["audio"])
open(spec["progress"], "w").write("0.5")
json.dump({"turns": [[0.0, float(len(audio)) / 16000, 0]]}, open(spec["result"], "w"))
"""

_CHILD_FAIL = """
import json, sys
spec = json.load(open(sys.argv[1], encoding="utf-8"))
json.dump({"error": "RuntimeError: boom"}, open(spec["result"], "w"))
sys.exit(1)
"""

_CHILD_SLOW = "import time; time.sleep(60)"


def _command(code):
    return lambda spec: [sys.executable, "-c", code, str(spec)]


def test_worker_returns_turns_and_cleans_up(tmp_path):
    partial = tmp_path / "diarize.k.partial.jsonl"
    seen = []
    turns = diarize_worker.run(np.zeros(32000, dtype=np.float32), tmp_path, [(0.0, 2.0)], partial,
                               progress=seen.append, cancel=threading.Event(), command=_command(_CHILD_OK))
    assert turns == [(0.0, 2.0, 0)]
    assert not list(tmp_path.glob("*.worker.*"))


def test_worker_failure_is_an_exception_with_the_reason(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        diarize_worker.run(np.zeros(16000, dtype=np.float32), tmp_path, [(0.0, 1.0)],
                           tmp_path / "d.partial.jsonl", cancel=threading.Event(), command=_command(_CHILD_FAIL))


def test_cancel_kills_the_worker_at_once(tmp_path):
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(JobCancelled):
        diarize_worker.run(np.zeros(16000, dtype=np.float32), tmp_path, [(0.0, 1.0)],
                           tmp_path / "d.partial.jsonl", cancel=cancel, command=_command(_CHILD_SLOW))
    assert time.monotonic() - started < 10


# -- queue ETA ---------------------------------------------------------------------------------------------------

def test_queue_eta_with_parallel_jobs():
    job = {"download": 10.0, "transcribe": 60.0, "diarize": 90.0, "refine": 200.0}
    assert eta.queue_remaining([job, job, job], 1) == 3 * 360.0
    # Three jobs in parallel: the AI (2 slots) carries 300 s, the CPU 270 s, one job 360 s -> 360 s.
    assert eta.queue_remaining([job, job, job], 4) == 360.0
    # Ten jobs: the AI is the bottleneck (2000 s over two slots), more than all work over four jobs (900 s).
    assert eta.queue_remaining([job] * 10, 4) == 1000.0
    assert eta.queue_remaining([], 4) == 0.0


# -- AI providers shared between running jobs -------------------------------------------------------------------

class _Client:
    def __init__(self, name, models):
        self.name = name
        self.models = list(models)
        self.spec = lp.PROVIDERS.get(name) or next(iter(lp.PROVIDERS.values()))
        self.timeout_s = 60.0
        self.asked = []

    def chat(self, model, messages, **kw):
        self.asked.append(model)
        raise ProviderCallError(f"{self.name} request failed: The read operation timed out")


def test_a_cool_down_in_one_job_applies_to_the_job_next_to_it(tmp_path):
    shared = SharedRouteState()
    first = ProviderPool([_Client("gemini", ["g1"])], shared=shared)
    second = ProviderPool([_Client("gemini", ["g1"])], shared=shared)
    first.cool_down("gemini:g1", 60.0)
    assert not second.available()
    alone = ProviderPool([_Client("gemini", ["g1"])])
    assert alone.available()                       # without the shared state nothing changes


def test_a_failure_remembered_by_one_job_is_seen_by_a_running_job(tmp_path):
    path = tmp_path / "h.json"
    assert HealthStore.shared(path) is HealthStore.shared(path)
    running = ProviderPool([_Client("nvidia", ["n1", "n2"])], health=HealthStore.shared(path))
    other = ProviderPool([_Client("nvidia", ["n1", "n2"])], health=HealthStore.shared(path))
    other.remember("nvidia:n1", 600, "read timeout")
    assert [r.key for r in running.available()] == ["nvidia:n2"]


def test_a_model_that_timed_out_on_the_brief_is_not_asked_again_for_the_speaker_map():
    client = _Client("nvidia", ["n1"])
    pool = ProviderPool([client])
    assert br._ask(pool, "s", {"x": 1}, lambda d: d, "Episode brief part 1", sleep=lambda s: None) is None
    assert br._ask(pool, "s", {"x": 1}, lambda d: d, "Speaker map", sleep=lambda s: None) is None
    assert client.asked == ["n1"]                  # asked once, then skipped
    assert pool.available()                        # still usable for the translation itself


# -- window: several jobs in progress -------------------------------------------------------------------------

def _window_factory(jobs_dir, delay=0.0):
    def factory(config, cancel, progress, recorder, extras=None):
        return Pipeline(config, jobs_dir, [CPU_ASR], [CPU_MT],
                        Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine(delay=delay)}),
                        Factory({(CPU_MT.model, "cpu"): FakeTranslator()}),
                        progress=progress, cancel=cancel, recorder=recorder,
                        resources=factory.resources, priority=(extras or {}).get("queue_priority", 0))
    factory.resources = ResourceSet()
    return factory


def test_window_keeps_the_configured_number_of_jobs_in_progress(qtbot, translator, settings, db, tmp_path):
    settings.set("parallel_jobs", 2)
    media = [make_tone_file(tmp_path / f"ep{i}.mp4", seconds=3.0 + i) for i in range(4)]
    w = MainWindow(translator, settings, RuntimeContext(db_path=db.path,
                                                         pipeline_factory=_window_factory(tmp_path / "jobs", 0.05)))
    w._show_error = lambda *a: None
    qtbot.addWidget(w)
    w.output_edit.setText(str(tmp_path / "out"))
    w.add_files_to_queue(media)
    most = []
    assert w.start_queue()
    qtbot.waitUntil(lambda: (most.append(len(w._runners)) or True) and not w.is_running()
                    and all(j["status"] == "completed" for j in JobsRepo(db).get_queue()), timeout=90000)
    assert max(most) == 2
    assert w._runner is None and w.progress_bar.value() == 100
    assert all(w.queue_table.item(r, 2).text() == "Completed" for r in range(4))


def test_parallel_cancel_stops_every_job(qtbot, translator, settings, db, tmp_path):
    settings.set("parallel_jobs", 3)
    media = [make_tone_file(tmp_path / f"long{i}.mp4", seconds=8.0 + i) for i in range(4)]
    w = MainWindow(translator, settings, RuntimeContext(db_path=db.path,
                                                         pipeline_factory=_window_factory(tmp_path / "jobs", 0.3)))
    w._show_error = lambda *a: None
    qtbot.addWidget(w)
    w.output_edit.setText(str(tmp_path / "out"))
    w.add_files_to_queue(media)
    assert w.start_queue()
    qtbot.waitUntil(lambda: len(w._runners) == 3, timeout=30000)
    w.cancel_queue()
    qtbot.waitUntil(lambda: not w.is_running() and not w._runners, timeout=60000)
    statuses = [j["status"] for j in JobsRepo(db).get_queue()]
    assert "completed" not in statuses and statuses.count("cancelled") == 4


def test_playlist_is_read_without_blocking_and_then_started(qtbot, translator, settings, db, tmp_path):
    entries = [{"url": f"https://www.youtube.com/watch?v={i}", "title": f"Episode {i}"} for i in range(3)]
    release = threading.Event()

    class SlowDownloader(FakeDownloader):
        def extract_playlist(self, url):
            release.wait(10)                       # yt-dlp reading a long playlist
            return super().extract_playlist(url)

    started = []
    w = MainWindow(translator, settings, RuntimeContext(
        db_path=db.path, pipeline_factory=None,
        downloader=SlowDownloader(tmp_path / "x.mp4", playlist_entries=entries)))
    qtbot.addWidget(w)
    w.start_queue = lambda: started.append(True) or True
    w.input_edit.setText("https://www.youtube.com/playlist?list=PL_TEST")
    w.source_combo.setCurrentIndex(w.source_combo.findData("tr"))
    w._start_after_playlist = True
    w._load_playlist_async("https://www.youtube.com/playlist?list=PL_TEST")
    assert w.queue_table.rowCount() == 0            # returned at once; the window keeps running
    release.set()
    qtbot.waitUntil(lambda: w.queue_table.rowCount() == 3, timeout=10000)
    assert started == [True]
    assert json.loads(JobsRepo(db).get_queue()[0]["queue_config"])["title"] == "Episode 0"


def test_translate_with_a_playlist_link_starts_once_the_playlist_is_read(qtbot, translator, settings, db, tmp_path):
    media = make_tone_file(tmp_path / "src.m4a", seconds=3.0)
    entries = [{"url": f"https://www.youtube.com/watch?v=v{i}", "title": f"Episode {i}"} for i in range(2)]
    downloader = FakeDownloader(media, playlist_entries=entries)

    def factory(config, cancel, progress, recorder, extras=None):
        return Pipeline(config, tmp_path / "jobs", [CPU_ASR], [CPU_MT],
                        Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine()}),
                        Factory({(CPU_MT.model, "cpu"): FakeTranslator()}),
                        progress=progress, cancel=cancel, recorder=recorder, downloader=downloader)

    w = MainWindow(translator, settings, RuntimeContext(db_path=db.path, pipeline_factory=factory,
                                                         downloader=downloader))
    w._show_error = lambda *a: None
    qtbot.addWidget(w)
    w.output_edit.setText(str(tmp_path / "out"))
    w.source_combo.setCurrentIndex(w.source_combo.findData("tr"))
    w.input_edit.setText("https://www.youtube.com/playlist?list=PL_TEST")
    assert w.start_queue()
    qtbot.waitUntil(lambda: not w.is_running() and len(JobsRepo(db).get_queue()) == 2
                    and all(j["status"] == "completed" for j in JobsRepo(db).get_queue()), timeout=60000)


def test_adding_a_playlist_from_the_queue_window_does_not_start_it(qtbot, translator, settings, db, tmp_path):
    entries = [{"url": "https://www.youtube.com/watch?v=a", "title": "A"}]
    w = MainWindow(translator, settings, RuntimeContext(
        db_path=db.path, pipeline_factory=None,
        downloader=FakeDownloader(tmp_path / "x.mp4", playlist_entries=entries)))
    qtbot.addWidget(w)
    w.source_combo.setCurrentIndex(w.source_combo.findData("tr"))
    w.input_edit.setText("https://www.youtube.com/playlist?list=PL_TEST")
    w._on_add_current_to_queue()
    qtbot.waitUntil(lambda: w.queue_table.rowCount() == 1, timeout=10000)
    assert not w.is_running() and JobsRepo(db).get_queue()[0]["status"] == "pending"
