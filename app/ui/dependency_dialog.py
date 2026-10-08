"""Start-up window that checks, repairs and updates the libraries by itself (D-114, supersedes D-113).

It shows "Checking libraries for updates..." while a worker thread verifies the libraries, repairs what is broken
and updates yt-dlp. It closes by itself when everything is fine. Only when something is still wrong afterwards does
the user see an error message, with the choice to quit or to continue anyway. The user never types a command.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QLabel, QMessageBox, QPlainTextEdit, QProgressBar, QVBoxLayout

from app.services import dependency_check as deps

log = logging.getLogger(__name__)

OUTCOME_OK = "ok"
OUTCOME_CONTINUE = "continue"
OUTCOME_QUIT = "quit"
OUTCOME_RESTART = "restart"

# Running workers are referenced here as well, so a dialog that is closed while a check runs can never destroy a
# running QThread (the same lifetime rule as the series picker, D-106).
_ACTIVE: "set[tuple[QThread, QObject]]" = set()


def _retire(pair: tuple["QThread", "QObject"]) -> None:
    _ACTIVE.discard(pair)
    pair[0].deleteLater()


class _Worker(QObject):
    progress = Signal(str)
    finished = Signal(object)         # deps.Result

    def __init__(self, perform: Callable[[Callable[[str], None]], "deps.Result"]):
        super().__init__()
        self._perform = perform

    @Slot()
    def run(self) -> None:
        try:
            result = self._perform(self.progress.emit)
        except Exception as exc:       # noqa: BLE001 - the check itself must never crash the start-up
            log.exception("Library check failed")
            result = deps.Result(errors=[f"The library check itself failed: {exc}"])
        self.finished.emit(result)


class DependencyDialog(QDialog):
    """Small notification window; `outcome` tells main() what to do afterwards."""

    def __init__(self, translator, settings=None, parent=None, perform=None):
        super().__init__(parent)
        self._tr = translator
        self._settings = settings
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self.report: deps.Result | None = None
        self.outcome = OUTCOME_OK
        self.setModal(True)
        self.setMinimumWidth(480)
        self.setWindowTitle(self._tr.t("deps.title"))

        layout = QVBoxLayout(self)
        self.status_label = QLabel(self._tr.t("deps.checking"))
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)                       # busy until the check finishes
        layout.addWidget(self.progress)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setFixedHeight(90)
        layout.addWidget(self.details)

        # Read on the UI thread: the settings database connection must not be used from the worker thread.
        self._last_lookup = float(settings.get("dependency_check_last")) if settings is not None else 0.0
        self._perform = perform or self._default_perform
        self.start_check()

    # -- the job ------------------------------------------------------------------------

    def _default_perform(self, progress: Callable[[str], None]) -> "deps.Result":
        from app.main import SELF_TEST_MODULES

        last = self._last_lookup
        modules = tuple(m for m in SELF_TEST_MODULES if not m.startswith("app."))
        return deps.perform_check(modules=modules, last_lookup=last, progress=progress)

    def start_check(self) -> None:
        if self._thread is not None:
            return
        thread = QThread()                                  # no parent: never destroyed while running (D-106)
        worker = _Worker(self._perform)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_result)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)
        _ACTIVE.add((thread, worker))
        thread.finished.connect(lambda pair=(thread, worker): _retire(pair))
        self._thread, self._worker = thread, worker
        thread.start()

    @Slot(str)
    def _on_progress(self, line: str) -> None:
        self.details.appendPlainText(line)

    @Slot(object)
    def _on_result(self, result) -> None:
        self.report = result
        if result.looked_up and self._settings is not None:
            try:
                self._settings.set("dependency_check_last", float(time.time()))
            except (TypeError, ValueError) as exc:
                log.warning("Cannot store the check time: %s", exc)
        for line in result.updated:
            log.info("Library updated: %s", line)
        for line in result.notes:
            log.info("Library check: %s", line)
        if result.errors:
            self.progress.setRange(0, 1)
            self.status_label.setText(self._tr.t("deps.failed"))
            self.outcome = OUTCOME_CONTINUE if self._ask_continue(result.errors) else OUTCOME_QUIT
        elif result.restart:
            self.status_label.setText(self._tr.t("deps.restarting"))
            self.outcome = OUTCOME_RESTART
        else:
            self.outcome = OUTCOME_OK
        self.accept()

    def _ask_continue(self, errors: list[str]) -> bool:
        """The error message. True = start the app anyway, False = quit. (Replaced in tests.)"""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(self._tr.t("deps.error_title"))
        box.setText(self._tr.t("deps.error_body"))
        box.setInformativeText("\n".join(f"- {error}" for error in errors))
        go_on = box.addButton(self._tr.t("deps.continue_anyway"), QMessageBox.ButtonRole.AcceptRole)
        quit_button = box.addButton(self._tr.t("deps.quit"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(quit_button)
        box.exec()
        return box.clickedButton() is go_on

    @Slot()
    def _on_thread_finished(self) -> None:
        thread, self._thread, self._worker = self._thread, None, None
        if thread is not None:
            thread.deleteLater()

    # -- lifetime -----------------------------------------------------------------------

    def stop(self) -> None:
        """Wait for the running worker (called when the dialog closes)."""
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(5000):
                thread.terminate()
                thread.wait(2000)
        self._on_thread_finished()

    def closeEvent(self, event) -> None:      # noqa: N802 - Qt naming
        self.stop()
        super().closeEvent(event)

    def center_on_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        size = self.sizeHint()
        self.resize(min(max(size.width(), 480), available.width() - 40), size.height())
        self.move(available.center() - self.rect().center())


def run_check_dialog(translator, settings, parent=None, perform=None) -> str:
    """Show the window modally (before the main window exists) and return its outcome."""
    dialog = DependencyDialog(translator, settings, parent, perform=perform)
    dialog.center_on_screen()
    if dialog.report is None:
        dialog.exec()
    dialog.stop()
    return dialog.outcome
