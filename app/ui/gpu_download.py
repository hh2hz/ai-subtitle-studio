"""Background download of the NVIDIA GPU libraries (app.services.gpu_runtime)."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QThread, Signal

from app.core.errors import JobCancelled
from app.services import gpu_runtime

log = logging.getLogger(__name__)


class GpuDownloadRunner(QThread):
    progress = Signal(float, str)
    succeeded = Signal()
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            gpu_runtime.download_all(self.progress.emit, self.cancel_event)
            self.succeeded.emit()
        except JobCancelled:
            self.failed.emit("cancelled")
        except Exception as exc:          # noqa: BLE001 - reported to the user
            log.exception("GPU library download failed")
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.cancel_event.set()
