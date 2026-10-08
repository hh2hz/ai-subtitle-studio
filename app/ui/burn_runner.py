"""Runs app.core.burn in a worker thread so the window stays responsive."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.core import burn
from app.core.errors import JobCancelled, PipelineError

log = logging.getLogger(__name__)


class BurnRunner(QThread):
    progress = Signal(float)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, video: Path | None, target_srt: Path | None, episode_dir: Path | None = None, parent=None):
        super().__init__(parent)
        self.video, self.target_srt, self.episode_dir = video, target_srt, episode_dir
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            video, srt = self.video, self.target_srt
            if video is None or srt is None:
                video, srt = burn.episode_files(self.episode_dir)
            out = burn.burn(video, srt, progress=self.progress.emit, cancel=self.cancel_event)
            self.succeeded.emit(str(out))
        except JobCancelled:
            self.failed.emit("cancelled")
        except PipelineError as exc:
            log.error("Burning subtitles failed: %s", exc)
            self.failed.emit(exc.ui_key)          # translated by the window (details are in the log)
        except Exception as exc:          # noqa: BLE001 - reported to the user
            log.exception("Burning subtitles failed")
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.cancel_event.set()
