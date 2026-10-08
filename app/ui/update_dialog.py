"""Update check and update window (D-118): background threads plus one dialog; the logic lives in services.updater."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from app.core.errors import JobCancelled
from app.services import updater
from app.services.updater import UpdateError, UpdateInfo
from app.utils.i18n import Translator

log = logging.getLogger(__name__)


class UpdateCheckRunner(QThread):
    """Asks GitHub for the newest release; never raises (a failure is reported through `failed`)."""

    found = Signal(object)           # UpdateInfo of the latest release
    failed = Signal(str)

    def __init__(self, fetch: Callable[[str], bytes] | None = None, parent=None):
        super().__init__(parent)
        self._fetch = fetch or updater.http_get

    def run(self) -> None:
        try:
            self.found.emit(updater.check_latest(self._fetch))
        except Exception as exc:          # noqa: BLE001 - reported, never fatal
            log.info("Update check failed: %s", exc)
            self.failed.emit(str(exc))


class UpdateDownloadRunner(QThread):
    progress = Signal(float)
    succeeded = Signal(str)          # path of the verified installer
    failed = Signal(str)

    def __init__(self, info: UpdateInfo, parent=None):
        super().__init__(parent)
        self._info = info
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            path = updater.download_installer(self._info, self.progress.emit, self.cancel_event)
            self.succeeded.emit(str(path))
        except JobCancelled:
            self.failed.emit("")
        except UpdateError as exc:
            log.warning("Update download failed: %s", exc)
            self.failed.emit(str(exc))
        except Exception as exc:          # noqa: BLE001
            log.exception("Update download failed")
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.cancel_event.set()


class UpdateDialog(QDialog):
    """Offers the update. `launched` is True after the verified installer was started (the caller then closes)."""

    def __init__(self, translator: Translator, info: UpdateInfo, current: str, *, allow_skip: bool,
                 can_update: bool | None = None, busy: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        t = translator.t
        self._tr, self._info = translator, info
        self._can_update = updater.can_self_update() if can_update is None else can_update
        self._runner: UpdateDownloadRunner | None = None
        self.launched = False
        self.skipped = False
        self.setWindowTitle(t("update.title"))
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        text = t("update.text", latest=info.version, current=current)
        if info.size:
            text += "\n" + t("update.size", size=f"{info.size / 1_048_576:.0f}")
        text += "\n\n" + t("update.keep" if self._can_update else "update.manual")
        if busy and self._can_update:
            text += "\n\n" + t("update.busy")
        self._base_text = text
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        row = QHBoxLayout()
        self.update_button = QPushButton(t("update.button_now") if self._can_update else t("update.button_page"))
        self.update_button.setDefault(True)
        self.update_button.setEnabled(not (busy and self._can_update))
        self.update_button.clicked.connect(self._start)
        self.skip_button = QPushButton(t("update.button_skip"))
        self.skip_button.setVisible(allow_skip)
        self.skip_button.clicked.connect(self._skip)
        self.later_button = QPushButton(t("update.button_later"))
        self.later_button.clicked.connect(self._later)
        for button in (self.update_button, self.skip_button, self.later_button):
            row.addWidget(button)
        layout.addLayout(row)

    def _start(self) -> None:
        if not self._can_update:
            QDesktopServices.openUrl(QUrl(self._info.page_url))
            self.reject()
            return
        t = self._tr.t
        self.update_button.setEnabled(False)
        self.skip_button.setEnabled(False)
        self.later_button.setText(t("update.button_cancel"))
        self.label.setText(t("update.downloading"))
        self.progress_bar.setVisible(True)
        self._runner = UpdateDownloadRunner(self._info, self)
        self._runner.progress.connect(lambda fraction: self.progress_bar.setValue(int(fraction * 100)))
        self._runner.succeeded.connect(self._downloaded)
        self._runner.failed.connect(self._download_failed)
        self._runner.start()

    def _downloaded(self, path: str) -> None:
        from pathlib import Path

        try:
            updater.launch_installer(Path(path))
        except UpdateError as exc:
            self._download_failed(str(exc))
            return
        self.launched = True
        self.accept()

    def _download_failed(self, message: str) -> None:
        t = self._tr.t
        self.progress_bar.setVisible(False)
        self.later_button.setText(t("update.button_later"))
        self.skip_button.setEnabled(True)
        self.update_button.setEnabled(True)
        self.label.setText(t("update.download_failed", error=message) if message else self._base_text)

    def _stop_runner(self) -> None:
        if self._runner is not None and self._runner.isRunning():
            self._runner.cancel()
            self._runner.wait(10000)

    def _later(self) -> None:
        if self._runner is not None and self._runner.isRunning():
            self._stop_runner()                          # the button said "Cancel download"
            self._download_failed("")
            return
        self.reject()

    def _skip(self) -> None:
        self.skipped = True
        self.reject()

    def reject(self) -> None:
        self._stop_runner()
        super().reject()
