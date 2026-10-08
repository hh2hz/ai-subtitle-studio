"""Detect series, season and episode for the current input (D-077).

A file name is parsed immediately; a URL needs one yt-dlp metadata call, so it runs on a worker thread and only
after the user stops typing. The window fills empty fields only, so nothing the user typed is overwritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from app.core.downloader import YtDlpAdapter, available_qualities
from app.core.metadata import SeriesInfo, detect
from app.utils.validation import is_url

DEBOUNCE_MS = 800

# Running probes are referenced here as well, so an owner destroyed without closeEvent() can never destroy a
# running QThread (see series_picker for the abort this prevents, D-106).
_ACTIVE: "set[tuple[QThread, QObject]]" = set()


def _retire(pair: tuple["QThread", "QObject"]) -> None:
    _ACTIVE.discard(pair)
    pair[0].deleteLater()


class _ProbeWorker(QObject):
    """One metadata lookup; a failure means "nothing detected", never an error dialog."""

    found = Signal(object)                 # SeriesInfo
    qualities = Signal(list)               # heights offered for this link, highest first (D-080)

    def __init__(self, url: str, adapter_factory: Callable[[], YtDlpAdapter]):
        super().__init__()
        self._url = url
        self._adapter_factory = adapter_factory

    @Slot()
    def run(self) -> None:
        try:
            info = self._adapter_factory().probe(self._url)
            result = detect(title=info.get("title"), filename=None, ytdlp_info=info)
            heights = available_qualities(info)
        except Exception:      # noqa: BLE001 - a failed probe must not disturb the window
            result, heights = SeriesInfo(), []
        self.qualities.emit(heights)
        self.found.emit(result)


class LinkProbe(QObject):
    """Watches the input field and reports what the link (or file name) says about the series."""

    resolved = Signal(object, str)         # SeriesInfo, origin: "url" or "filename"
    qualities = Signal(list)               # heights offered for the probed link, highest first

    def __init__(self, edit, adapter_factory: Callable[[], YtDlpAdapter] | None = None,
                 delay_ms: int = DEBOUNCE_MS, parent: QObject | None = None):
        super().__init__(parent)
        self._edit = edit
        # None = no URL probing at all (file names are still parsed). The app injects its downloader, so a test
        # window without one can never reach the network.
        self._adapter_factory = adapter_factory
        self._pending = ""
        self._started_for = ""
        self._thread: QThread | None = None
        self._worker: _ProbeWorker | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(self._start)
        edit.textChanged.connect(self._on_text_changed)

    @Slot(str)
    def _on_text_changed(self, text: str) -> None:
        text = text.strip()
        if not text or text == self._started_for:
            return
        self._pending = text
        self._timer.start()

    @Slot()
    def _start(self) -> None:
        text = self._pending
        if not text or text == self._started_for:
            return
        self._started_for = text
        if not is_url(text):
            path = Path(text)
            if path.is_file():
                self.resolved.emit(detect(filename=path.name), "filename")
            return
        if self._adapter_factory is None or self._thread is not None:
            return
        # No parent: a running QThread must never be destroyed by its owner (that aborts the process, D-106).
        thread = QThread()
        worker = _ProbeWorker(text, self._adapter_factory)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.found.connect(self._on_found)
        worker.found.connect(thread.quit)
        worker.qualities.connect(self.qualities)
        thread.finished.connect(self._on_finished)
        _ACTIVE.add((thread, worker))
        thread.finished.connect(lambda pair=(thread, worker): _retire(pair))
        # Keep the worker alive while the thread runs: without a reference Qt may free it mid-flight.
        self._thread, self._worker = thread, worker
        thread.start()

    @Slot(object)
    def _on_found(self, info) -> None:
        self.resolved.emit(info, "url")

    @Slot()
    def _on_finished(self) -> None:
        thread, self._thread, self._worker = self._thread, None, None
        if thread is not None:
            thread.deleteLater()

    def start_now(self) -> None:
        """Probe the current text immediately (Enter, or a test)."""
        self._timer.stop()
        self._pending = self._edit.text().strip()
        self._start()

    def wait_for_lookup(self) -> None:
        if self._thread is not None:
            self._thread.wait(15000)

    def stop(self) -> None:
        """Stop the probe thread and wait for it; safe to call when no probe runs (D-106)."""
        self._timer.stop()
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(5000):
                thread.terminate()
                thread.wait(2000)
        self._on_finished()
