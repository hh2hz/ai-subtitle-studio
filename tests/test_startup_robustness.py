"""Startup robustness: an unwritable log file must not stop the app, and the window must be on screen (D-109)."""

import logging
from logging.handlers import RotatingFileHandler

from PySide6.QtGui import QGuiApplication

from app.utils.logging_setup import setup_logging


def _root_file_handlers():
    return [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]


def test_setup_logging_writes_the_file(tmp_path):
    log_file = setup_logging(tmp_path, "INFO")

    logging.getLogger("test").info("hello")
    assert log_file == tmp_path / "app.log"
    assert "hello" in log_file.read_text(encoding="utf-8")
    assert _root_file_handlers()


def test_setup_logging_survives_an_unwritable_file(tmp_path, monkeypatch):
    """The reported crash: PermissionError on app.log while starting the app."""
    real = RotatingFileHandler
    calls = []

    def failing(filename, *args, **kwargs):
        calls.append(str(filename))
        if str(filename).endswith("app.log"):
            raise PermissionError(13, "Permission denied", str(filename))
        return real(filename, *args, **kwargs)

    monkeypatch.setattr("app.utils.logging_setup.RotatingFileHandler", failing)
    log_file = setup_logging(tmp_path, "INFO")

    assert calls[0].endswith("app.log")
    assert log_file.name.startswith("app-") and log_file.suffix == ".log"   # the timestamped fallback
    logging.getLogger("test").warning("fallback works")
    assert "fallback works" in log_file.read_text(encoding="utf-8")


def test_setup_logging_survives_a_completely_read_only_folder(tmp_path, monkeypatch):
    """Sandbox or read-only data folder: no file at all, but logging still works and nothing raises."""

    def always_failing(filename, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(filename))

    monkeypatch.setattr("app.utils.logging_setup.RotatingFileHandler", always_failing)
    log_file = setup_logging(tmp_path, "INFO")

    assert log_file == tmp_path / "app.log"
    assert _root_file_handlers() == []
    assert logging.getLogger().handlers            # stderr handler still installed
    logging.getLogger("test").error("still alive")  # must not raise


def _window(qtbot, translator, settings, db, tmp_path):
    from app.ui.main_window import MainWindow, RuntimeContext

    runtime = RuntimeContext(db_path=db.path, pipeline_factory=lambda *args, **kwargs: None,
                             cache_dir=tmp_path / "cache")
    window = MainWindow(translator, settings, runtime)
    qtbot.addWidget(window)
    return window


def test_window_is_moved_inside_the_screen(qtbot, translator, settings, db, tmp_path):
    """A window placed below the usable area (the reported "opens shifted down") is moved up."""
    window = _window(qtbot, translator, settings, db, tmp_path)
    available = QGuiApplication.primaryScreen().availableGeometry()

    window.setGeometry(available.left(), available.bottom() - 40, 860, 580)   # mostly off-screen
    window._fit_to_screen()

    frame = window.frameGeometry()
    assert available.contains(frame), (frame, available)


def test_window_is_shrunk_when_it_is_taller_than_the_screen(qtbot, translator, settings, db, tmp_path):
    window = _window(qtbot, translator, settings, db, tmp_path)
    available = QGuiApplication.primaryScreen().availableGeometry()

    window.setGeometry(available.left(), available.top(), 860, available.height() + 400)
    window._fit_to_screen()

    assert window.frameGeometry().height() <= available.height()
    assert window.frameGeometry().top() >= available.top()


def test_geometry_is_saved_as_hex_and_reads_back(qtbot, translator, settings, db, tmp_path):
    from PySide6.QtCore import QByteArray

    from app.ui.main_window import GEOMETRY_VERSION

    window = _window(qtbot, translator, settings, db, tmp_path)
    window._save_geometry()

    saved = settings.get("window_geometry")
    assert saved and saved == saved.lower() and len(saved) % 2 == 0
    assert settings.get("window_geometry_version") == GEOMETRY_VERSION
    assert bytes(window.saveGeometry()).hex() == saved                    # the hex form is the geometry itself
    assert window.restoreGeometry(QByteArray.fromHex(saved.encode("ascii"))) is True


def test_a_geometry_from_an_older_layout_is_ignored(qtbot, translator, settings, db, tmp_path):
    """Restoring the old (too large) window would bring the reported problem straight back (D-111)."""
    window = _window(qtbot, translator, settings, db, tmp_path)
    window._save_geometry()
    settings.set("window_geometry_version", 0)          # as if it had been saved by the old layout

    second = _window(qtbot, translator, settings, db, tmp_path)

    from app.ui.main_window import DEFAULT_WINDOW_SIZE

    assert second.width() <= max(DEFAULT_WINDOW_SIZE[0], second.minimumSizeHint().width())


def test_a_reopened_window_is_always_inside_the_screen(qtbot, translator, settings, db, tmp_path):
    """Even a geometry saved on a bigger monitor must not place the window off-screen, title bar included."""
    window = _window(qtbot, translator, settings, db, tmp_path)
    available = QGuiApplication.primaryScreen().availableGeometry()
    window.setGeometry(available.left(), available.bottom() - 30, 700, 500)   # mostly below the screen
    window._fit_to_screen()
    window._save_geometry()

    second = _window(qtbot, translator, settings, db, tmp_path)

    assert available.contains(second.frameGeometry()), second.frameGeometry()


def test_the_frame_not_only_the_client_area_fits(qtbot, translator, settings, db, tmp_path):
    """setGeometry ignores the title bar, which is how a "fitted" window still stuck out (D-111)."""
    window = _window(qtbot, translator, settings, db, tmp_path)
    available = QGuiApplication.primaryScreen().availableGeometry()

    window.setGeometry(available.left(), available.top(), available.width(), available.height() + 200)
    window._fit_to_screen()

    frame = window.frameGeometry()
    assert frame.height() <= available.height(), (frame, available)
    assert frame.top() >= available.top(), (frame, available)
    assert frame.left() >= available.left() and frame.right() <= available.right(), (frame, available)


def test_a_corrupt_saved_geometry_falls_back_to_the_default(qtbot, translator, settings, db, tmp_path):
    settings.set("window_geometry", "")             # the validator rejects anything that is not hex
    window = _window(qtbot, translator, settings, db, tmp_path)

    assert window.width() > 0 and window.height() > 0


def test_the_window_can_shrink_into_a_laptop_work_area(qtbot, translator, settings, db, tmp_path):
    """The reported problem: the window was 1908 px wide on a 1536 px work area, so it stuck out (D-110).

    Measured on the user's machine: 1920x1080 at 125 % scaling gives 1536x816 usable pixels. The minimum depends on
    the language (Arabic labels are longer), so the assertion is against that work area, not a fixed number.
    """
    work_area = (1536, 816)
    window = _window(qtbot, translator, settings, db, tmp_path)
    minimum = window.minimumSizeHint()

    assert minimum.width() <= work_area[0], minimum
    assert minimum.height() <= work_area[1], minimum


def test_the_default_size_fits_the_work_area(qtbot, translator, settings, db, tmp_path):
    from app.ui.main_window import DEFAULT_WINDOW_SIZE

    window = _window(qtbot, translator, settings, db, tmp_path)

    assert DEFAULT_WINDOW_SIZE[0] <= 1536 and DEFAULT_WINDOW_SIZE[1] <= 816
    assert window.width() <= 1536 and window.height() <= 816
