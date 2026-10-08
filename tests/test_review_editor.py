"""Tests for Phase 6 Task 3: review editor features."""

from pathlib import Path
import pytest
from PySide6.QtCore import Qt

from app.core.audio_processor import SAMPLE_RATE, load_wav
from app.core.exporter import read_srt
from app.core.modes import Mode
from app.core.pipeline import JobConfig, Pipeline
from app.core.review import ReviewError, ReviewSession
from app.ui.review_window import ReviewWindow, TimelineStrip, ShiftDialog, FindReplaceDialog
from tests.fakes import CPU_ASR, CPU_MT, Factory, FakeAsrEngine, FakeTranslator
from tests.media import make_tone_file

ARABIC_LINE = "\u0647\u0644 \u0623\u0646\u062a \u0645\u062a\u0623\u0643\u062f\u061f"


class ArabicTranslator(FakeTranslator):
    def translate(self, texts, source_language, target_language, context=None):
        super().translate(texts, source_language, target_language, context)
        return [ARABIC_LINE if "3" not in t else "untranslated text" for t in texts]


@pytest.fixture(autouse=True)
def auto_discard_msgbox(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Discard)


@pytest.fixture
def episode(tmp_path):
    media = make_tone_file(tmp_path / "Ep.m4a", seconds=20.0)
    config = JobConfig(media, "tr", "ar", Mode.BALANCED, output_dir=tmp_path / "out")
    result = Pipeline(config, tmp_path / "jobs", [CPU_ASR], [CPU_MT],
                      Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine()}),
                      Factory({(CPU_MT.model, "cpu"): ArabicTranslator()})).run()
    return result.output_dir


def test_split_and_merge_cues(episode):
    session = ReviewSession(episode)
    initial_count = len(session.units)
    unit0 = session.unit(0)
    orig_start, orig_end = unit0["start"], unit0["end"]

    # Split unit 0
    u1, u2 = session.split_unit(0)
    assert len(session.units) == initial_count + 1
    assert u1["start"] == orig_start
    assert u1["end"] == u2["start"]
    assert u2["end"] == orig_end

    # Merge unit 0 with unit 2 (the new unit)
    merged = session.merge_units(u1["id"], u2["id"])
    assert len(session.units) == initial_count
    assert merged["start"] == orig_start
    assert merged["end"] == orig_end


def test_shift_cues(episode):
    session = ReviewSession(episode)
    u0 = session.unit(0)
    u1 = session.unit(1)
    s0, e0 = u0["start"], u0["end"]
    s1, e1 = u1["start"], u1["end"]

    # Shift single cue
    session.shift_cues(1.5, [u0["id"]])
    assert round(session.unit(0)["start"], 3) == round(s0 + 1.5, 3)
    assert round(session.unit(0)["end"], 3) == round(e0 + 1.5, 3)
    assert round(session.unit(1)["start"], 3) == round(s1, 3)

    # Shift all cues
    session.shift_cues(-0.5)
    assert round(session.unit(0)["start"], 3) == round(s0 + 1.0, 3)
    assert round(session.unit(1)["start"], 3) == round(s1 - 0.5, 3)


def test_delete_cue_and_blank_text(episode):
    session = ReviewSession(episode)
    initial_count = len(session.units)
    u0_id = session.unit(0)["id"]

    # Blanking text
    session.set_text(u0_id, "")
    assert session.unit(u0_id)["reviewed_text"] == ""
    assert session.unit(u0_id)["final_text"] == ""

    # Delete cue
    session.delete_unit(u0_id)
    assert len(session.units) == initial_count - 1
    assert not any(u["id"] == u0_id for u in session.units)

    # Save and reload
    srt_path = session.save()
    reloaded = ReviewSession(episode)
    assert len(reloaded.units) == initial_count - 1
    assert not any(u["id"] == u0_id for u in reloaded.units)


def test_find_replace_episode_and_series(tmp_path, episode):
    session = ReviewSession(episode)
    # Put known text in line 0
    session.set_text(0, "Apple Banana Apple")

    # Replace in current episode
    replaced = session.find_replace_episode("Apple", "Orange")
    assert replaced == 2
    assert "Orange Banana Orange" in session.current_text(session.unit(0))

    # Test series find & replace
    series_res = session.find_replace_series("Orange", "Grape")
    assert sum(series_res.values()) >= 2
    assert "Grape Banana Grape" in session.current_text(session.unit(0))


def test_retranscribe_span_uses_real_engine(episode):
    session = ReviewSession(episode)
    unit = session.unit(1)
    unit["text"] = "stale source text"
    translation = unit["translation"]
    reviewed = unit.get("reviewed_text")
    engine = FakeAsrEngine()

    new_text = session.retranscribe_unit(1, engine)

    total = len(load_wav(session.work_dir / "audio.wav"))
    first = max(0, round((unit["start"] - 0.3) * SAMPLE_RATE))
    last = min(total, round((unit["end"] + 0.3) * SAMPLE_RATE))
    call = engine.calls[-1]
    assert call["samples"] == last - first          # the real slice, with the 0.3 s margin
    assert call["offset"] == pytest.approx(first / SAMPLE_RATE)
    assert call["prompt"] is None
    assert new_text and new_text != "stale source text"
    assert session.unit(1)["text"] == new_text      # the source line is replaced
    assert session.unit(1)["translation"] == translation       # the translation is not redone
    assert session.unit(1).get("reviewed_text") == reviewed
    assert session.dirty


def test_retranscribe_span_without_speech_keeps_old_text(episode):
    session = ReviewSession(episode)
    unit = session.unit(0)
    unit["text"] = "old source text"

    class SilentAsrEngine(FakeAsrEngine):
        def transcribe(self, audio, language, offset, initial_prompt):
            self.calls.append({"offset": offset, "prompt": initial_prompt, "samples": len(audio)})
            return iter(())

    engine = SilentAsrEngine()
    assert session.retranscribe_unit(0, engine) == ""
    assert unit["text"] == "old source text"
    assert engine.calls[-1]["samples"] > 0
    assert not session.dirty


def test_retranscribe_span_releases_the_engine(episode):
    session = ReviewSession(episode)
    engine = FakeAsrEngine()
    released = []

    class Factory:
        def __call__(self):
            return engine

        def release(self):
            released.append(True)

    session.retranscribe_unit(0, Factory())
    assert released == [True]


def test_retranscribe_span_without_engine_raises(episode):
    session = ReviewSession(episode)
    with pytest.raises(ReviewError):
        session.retranscribe_unit(0, None)


def test_window_reports_an_empty_retranscription(qtbot, translator, episode):
    window = ReviewWindow(episode, translator, enable_media=False)
    qtbot.addWidget(window)
    window._on_retranscribed(0, "", "")
    assert window.statusBar().currentMessage() == translator.t("review.retranscribe_empty")


def test_timeline_strip_widget(qtbot, episode):
    session = ReviewSession(episode)
    timeline = TimelineStrip(session)
    qtbot.addWidget(timeline)

    timeline.set_position(5.0)
    assert timeline._current_time == 5.0
    timeline.set_selected_id(0)
    assert timeline._selected_id == 0

    clicked_ids = []
    seek_times = []
    timeline.cue_clicked.connect(clicked_ids.append)
    timeline.time_seek.connect(seek_times.append)

    # Trigger click on timeline
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QPointF
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(10, 10), Qt.MouseButton.LeftButton,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    timeline.mousePressEvent(ev)
    assert len(seek_times) == 1


def test_window_cue_operations(qtbot, translator, episode):
    window = ReviewWindow(episode, translator, enable_media=False)
    qtbot.addWidget(window)
    window.filter_combo.setCurrentIndex(1)  # All lines
    initial_rows = window.table.rowCount()

    # Split line
    window.select_row(0)
    window.split_line()
    assert window.table.rowCount() == initial_rows + 1

    # Merge line
    window.select_row(0)
    window.merge_line()
    assert window.table.rowCount() == initial_rows

    # Blank text in editor commits empty text
    window.select_row(0)
    window.text_edit.setPlainText("")
    window._commit_edit()
    assert window.session.unit(window._rows[0])["reviewed_text"] == ""

    # Delete line
    window.select_row(0)
    window.delete_line()
    assert window.table.rowCount() == initial_rows - 1
    window.session.dirty = False


def test_batch_retranslate_selected_lines(qtbot, translator, episode):
    from PySide6.QtCore import QItemSelectionModel
    from tests.fake_llm_api import FakeLlmApi
    from tests.test_pipeline import _refiner_factory

    with FakeLlmApi() as server:
        refiner_factory = _refiner_factory(server, review=False)
        window = ReviewWindow(episode, translator, refiner_factory, enable_media=False)
        qtbot.addWidget(window)
        window.filter_combo.setCurrentIndex(1)  # All lines

        # Select two rows
        window.select_row(0)
        idx1 = window.table.model().index(1, 0)
        window.table.selectionModel().select(
            idx1, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        )

        window.retranslate_lines()
        qtbot.waitUntil(lambda: window._thread is None, timeout=15000)

        # Both lines should have received translation even if user navigated away
        assert window.session.unit(window._rows[0])["reviewed_text"].startswith("AR(")
        assert window.session.unit(window._rows[1])["reviewed_text"].startswith("AR(")
        window.session.dirty = False


def test_alignment_and_language_switch(qtbot, translator, episode):
    translator.set_language("en")
    window = ReviewWindow(episode, translator, enable_media=False)
    qtbot.addWidget(window)

    # English target or Arabic target check
    is_rtl = window.session.target_language.split("-")[0].lower() in ("ar", "fa", "he", "ur")
    expected_align = Qt.AlignmentFlag.AlignRight if is_rtl else Qt.AlignmentFlag.AlignLeft
    tgt_item = window.table.item(0, 4)
    assert (tgt_item.textAlignment() & expected_align) == expected_align

    # Test dynamic UI language switch
    translator.set_language("ar")
    assert window.save_button.text() == translator.t("review.save")
    assert window.delete_button.text() == translator.t("review.delete")
    assert window.split_button.text() == translator.t("review.split")


def test_split_after_deleting_last_cue_keeps_new_id_unique(episode):
    session = ReviewSession(episode)
    last_id = session.units[-1]["id"]
    session.delete_unit(last_id)
    _, new = session.split_unit(session.units[0]["id"])
    assert new["id"] > last_id
    session.save()
    reloaded = ReviewSession(episode)
    assert new["id"] in {u["id"] for u in reloaded.units}


def test_unit_lookup_after_delete_does_not_raise(episode):
    session = ReviewSession(episode)
    last_id = session.units[-1]["id"]
    session.delete_unit(session.units[0]["id"])
    assert session.unit(last_id)["id"] == last_id


def _main_window(qtbot, translator, settings, db, pipeline_factory=None):
    from app.ui.main_window import MainWindow, RuntimeContext

    runtime = RuntimeContext(db_path=db.path, pipeline_factory=pipeline_factory or (lambda *a, **k: None))
    window = MainWindow(translator, settings, runtime)
    window._show_error = lambda *a: None
    qtbot.addWidget(window)
    return window


def _install_fake_whisper(models_dir, name="tiny"):
    from app.models.model_manager import COMPLETE_MARKER, ModelManager

    model_dir = ModelManager(models_dir).path("whisper", name)
    model_dir.mkdir(parents=True)
    (model_dir / COMPLETE_MARKER).write_text("test", encoding="utf-8")


def test_main_window_passes_asr_factory_when_a_model_is_installed(qtbot, translator, settings, db, tmp_path, episode):
    models_dir = tmp_path / "models"
    _install_fake_whisper(models_dir)
    settings.set("model_dir", str(models_dir))

    window = _main_window(qtbot, translator, settings, db)
    window.open_review(episode)

    review = window._review_windows[-1]
    qtbot.addWidget(review)
    assert review._asr_factory is not None


def test_main_window_passes_no_asr_factory_without_a_model(qtbot, translator, settings, db, tmp_path, episode):
    settings.set("model_dir", str(tmp_path / "empty-models"))

    window = _main_window(qtbot, translator, settings, db)
    window.open_review(episode)

    review = window._review_windows[-1]
    qtbot.addWidget(review)
    assert review._asr_factory is None


def test_asr_engine_factory_loads_once_and_releases(monkeypatch, tmp_path):
    from app.models import engines
    from app.models.model_manager import ModelManager
    from app.services import hardware_detection
    from app.services.hardware_detection import HardwareInfo
    from app.services.job_manager import AsrEngineFactory

    models_dir = tmp_path / "models"
    _install_fake_whisper(models_dir)
    models = ModelManager(models_dir)
    loaded = []

    def fake_load(plan, manager, options):
        loaded.append(plan.model)
        return FakeAsrEngine()

    monkeypatch.setattr(engines, "load_asr_engine", fake_load)
    monkeypatch.setattr(hardware_detection, "detect", lambda data_dir: HardwareInfo(
        os="test", cpu_name="test", logical_cores=4, ram_mb=8000, disk_free_mb=None))

    factory = AsrEngineFactory(models, ["tiny"])
    engine = factory()
    assert engine is not None
    assert factory() is engine                 # the engine is cached, not loaded twice
    assert loaded == ["tiny"]
    factory.release()
    assert factory() is not None
    assert loaded == ["tiny", "tiny"]          # release() drops it, so the next span loads again
    # No installed model -> nothing to load.
    assert AsrEngineFactory(ModelManager(tmp_path / "none"), [])() is None
