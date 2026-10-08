"""Series picker: TVMaze suggestions, cached posters and kept series (D-076).

No network: HTTP is injected into the lookup, and the payloads are the shapes TVMaze really returns.
"""

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QLabel, QLineEdit

from app.database.database import Database
from app.database.series import SeriesRepo
from app.services import series_lookup
from app.ui.series_picker import SeriesSuggestions, load_pixmap

# A real (tiny) PNG, produced by Qt itself: enough to prove the poster path without shipping an image.
_PNG: bytes | None = None


def _png() -> bytes:
    global _PNG
    if _PNG is None:
        from PySide6.QtCore import QBuffer
        from PySide6.QtGui import QColor, QPixmap

        pixmap = QPixmap(4, 6)
        pixmap.fill(QColor("black"))
        buffer = QBuffer()
        buffer.open(QBuffer.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        _PNG = bytes(buffer.data())
        buffer.close()
    return _PNG

SEARCH_PAYLOAD = json.dumps([
    {"score": 0.67, "show": {"id": 7749, "name": "Kurtlar Vadisi Pusu", "premiered": "2007-04-19",
                             "image": {"medium": "https://img/pusu.jpg"}}},
    {"score": 0.54, "show": {"id": 18332, "name": "Kurtlar Vadisi", "premiered": "2003-01-15",
                             "image": {"medium": "https://img/vadisi.jpg"}}},
]).encode("utf-8")


def _fetch(url: str) -> bytes:
    return _png() if url.startswith("https://img/") else SEARCH_PAYLOAD


# -- the lookup itself -------------------------------------------------------------------------

def test_parse_search_reads_the_tvmaze_shape():
    found = series_lookup.parse_search(SEARCH_PAYLOAD)

    assert [c.name for c in found] == ["Kurtlar Vadisi Pusu", "Kurtlar Vadisi"]
    assert found[0].source_id == "7749" and found[0].year == 2007
    assert found[0].image_url == "https://img/pusu.jpg" and found[0].label == "Kurtlar Vadisi Pusu (2007)"
    assert found[1].label == "Kurtlar Vadisi (2003)"


def test_parse_search_survives_a_bad_payload():
    assert series_lookup.parse_search(b"not json") == []
    assert series_lookup.parse_search(b'{"shows": []}') == []
    assert series_lookup.parse_search(json.dumps([{"show": {"id": 1}}]).encode()) == []


def test_search_skips_short_queries_and_failures():
    assert series_lookup.search("ku", fetch=_fetch) == []       # no request is made at all

    def offline(url):
        raise OSError("no network")

    assert series_lookup.search("kurtlar", fetch=offline) == []


def test_cache_poster_downloads_once_and_refuses_other_schemes(qtbot, tmp_path):
    calls: list[str] = []

    def fetch(url):
        calls.append(url)
        return _png()

    first = series_lookup.cache_poster("https://img/pusu.jpg", tmp_path, fetch=fetch)
    second = series_lookup.cache_poster("https://img/pusu.jpg", tmp_path, fetch=fetch)

    assert first == second and first.is_file() and len(calls) == 1
    assert load_pixmap(first) is not None
    assert series_lookup.cache_poster("file:///c:/windows/win.ini", tmp_path, fetch=fetch) is None


# -- the picker --------------------------------------------------------------------------------

def _picker(tmp_path, repo=None, edit=None, preview=None):
    edit = edit or QLineEdit()
    preview = preview or QLabel()
    return SeriesSuggestions(edit, preview, repo, tmp_path, fetch=_fetch), edit, preview


def test_picker_offers_kept_series_with_their_poster_offline(qtbot, db, tmp_path):
    repo = SeriesRepo(db)
    poster = series_lookup.cache_poster("https://img/pusu.jpg", tmp_path, fetch=_fetch)
    repo.save("Kurtlar Vadisi Pusu", "7749", "https://img/pusu.jpg", str(poster))

    picker, edit, preview = _picker(tmp_path, repo)
    model = edit.completer().model()

    assert model.rowCount() == 1 and model.item(0).text() == "Kurtlar Vadisi Pusu"
    assert not model.item(0).icon().isNull()          # the cached poster became the icon
    edit.setText("Kurtlar Vadisi Pusu")
    picker._show_preview_for("Kurtlar Vadisi Pusu")
    assert not preview.pixmap().isNull()


def test_picker_looks_up_typed_text_in_the_background(qtbot, tmp_path):
    picker, edit, _preview = _picker(tmp_path)
    edit.setText("kurtlar")
    edit.textEdited.emit("kurtlar")
    picker._start_lookup()
    picker.wait_for_lookup()

    qtbot.waitUntil(lambda: edit.completer().model().rowCount() == 2, timeout=10000)
    model = edit.completer().model()
    assert [model.item(row).text() for row in range(2)] == ["Kurtlar Vadisi Pusu (2007)", "Kurtlar Vadisi (2003)"]
    assert not model.item(0).icon().isNull()          # the poster was downloaded and cached
    assert (tmp_path / series_lookup.poster_name("https://img/pusu.jpg")).is_file()


def test_closing_the_window_stops_a_running_lookup(qtbot, translator, settings, db, tmp_path):
    """A window closed (and later freed) during a lookup must stop the thread, not destroy it (D-106).

    The lookup thread used to be parented to its owner with no shutdown path, so closing or dropping a window
    mid-lookup destroyed a running QThread - Qt aborts the whole process then, which pytest reported as
    "Fatal Python error: Aborted" around this module.
    """
    import threading

    from app.ui.main_window import MainWindow, RuntimeContext
    from app.ui.series_picker import _ACTIVE

    started, release = threading.Event(), threading.Event()

    def slow_fetch(url):
        started.set()
        release.wait(5)
        return b"[]"

    runtime = RuntimeContext(db_path=db.path, pipeline_factory=lambda *args, **kwargs: None,
                             cache_dir=tmp_path / "cache", series_fetch=slow_fetch)
    window = MainWindow(translator, settings, runtime)
    qtbot.addWidget(window)

    window.series_edit.setText("kurtlar vadisi")
    window.series_picker._start_lookup()
    assert started.wait(5), "the lookup thread never started"

    threading.Timer(0.5, release.set).start()          # let the worker return during the wait
    window.close()                                     # closeEvent stops and waits for the thread

    assert window.series_picker._thread is None and window.series_picker._worker is None
    assert not any(thread.isRunning() for thread, _ in _ACTIVE)


def test_picker_activation_fills_the_field_and_preview(qtbot, tmp_path):
    picker, edit, preview = _picker(tmp_path)
    edit.setText("kurtlar")
    edit.textEdited.emit("kurtlar")
    picker._start_lookup()
    picker.wait_for_lookup()
    qtbot.waitUntil(lambda: edit.completer().model().rowCount() == 2, timeout=10000)

    picker._on_activated("Kurtlar Vadisi (2003)")

    assert edit.text() == "Kurtlar Vadisi"
    assert picker.current().source_id == "18332"
    assert not preview.pixmap().isNull()


def test_picker_without_a_repository_offers_nothing_but_still_works(qtbot, tmp_path):
    picker, edit, _preview = _picker(tmp_path, repo=None)
    edit.setText("Typed Name")
    assert edit.completer().model().rowCount() == 0
    assert picker.current().name == "Typed Name"


# -- keeping and forgetting --------------------------------------------------------------------

def test_main_window_keeps_and_forgets_a_series(qtbot, translator, settings, db, tmp_path):
    from app.ui.main_window import MainWindow, RuntimeContext

    runtime = RuntimeContext(db_path=db.path, pipeline_factory=lambda *args, **kwargs: None,
                             cache_dir=tmp_path / "cache")
    window = MainWindow(translator, settings, runtime)
    qtbot.addWidget(window)
    repo = SeriesRepo(Database(db.path))

    window.series_edit.setText("Kurtlar Vadisi Pusu")
    window.series_keep_check.setChecked(True)          # checking keeps the series
    assert [row["name"] for row in repo.saved()] == ["Kurtlar Vadisi Pusu"]

    window.series_keep_check.setChecked(False)         # unchecking drops it again
    assert repo.saved() == []
    window.close()


def test_settings_dialog_removes_a_kept_series_on_save(qtbot, translator, settings, db, tmp_path):
    from app.ui.settings_window import ProviderSettingsDialog

    repo = SeriesRepo(Database(db.path))
    repo.save("Kurtlar Vadisi")
    repo.save("Kurtlar Vadisi Pusu")
    dialog = ProviderSettingsDialog(translator, settings, series_repo=repo, poster_cache_dir=tmp_path)
    qtbot.addWidget(dialog)

    assert dialog.saved_series_list.count() == 2
    for row in range(dialog.saved_series_list.count()):
        item = dialog.saved_series_list.item(row)
        item.setSelected(item.text() == "Kurtlar Vadisi")
    dialog._remove_selected_series()
    assert dialog.saved_series_list.count() == 1
    assert len(repo.saved()) == 2                      # nothing is deleted until Save

    dialog.save()
    assert [row["name"] for row in repo.saved()] == ["Kurtlar Vadisi Pusu"]


def test_running_a_job_disables_the_series_keep_box_and_the_quality_combo(qtbot, translator, settings, db, tmp_path):
    from app.ui.main_window import MainWindow, RuntimeContext

    runtime = RuntimeContext(db_path=db.path, pipeline_factory=lambda *args, **kwargs: None,
                             cache_dir=tmp_path / "cache")
    window = MainWindow(translator, settings, runtime)
    qtbot.addWidget(window)
    window.input_edit.setText("https://example.invalid/watch?v=1")
    window._update_quality_state()
    assert window.series_keep_check.isEnabled() and window.quality_combo.isEnabled()

    window._set_running_ui(True)
    assert not window.series_keep_check.isEnabled() and not window.quality_combo.isEnabled()
    assert window.start_button.isEnabled()                      # the cancel button stays usable

    window._set_running_ui(False)
    assert window.series_keep_check.isEnabled() and window.quality_combo.isEnabled()
    window.input_edit.setText("C:/video.mp4")                   # a local file has no quality choice
    window._update_quality_state()
    window._set_running_ui(True)
    window._set_running_ui(False)
    assert not window.quality_combo.isEnabled()
