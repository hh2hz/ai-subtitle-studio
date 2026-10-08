"""Suggestions and poster preview for the series name field (D-076).

Debounced: the field is looked up only after the user stops typing, on a worker thread, so the window never waits
on the network. Kept series come from the database and are offered immediately, with their cached poster, even
without a connection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QCompleter, QLabel, QLineEdit

from app.database.series import SeriesRepo
from app.services import series_lookup
from app.services.series_lookup import SeriesCandidate

MIN_QUERY_CHARS = 3
DEBOUNCE_MS = 500
MAX_SUGGESTIONS = 8
PREVIEW_SIZE = (44, 62)

# Running lookups are referenced here as well, so a window that is destroyed without closeEvent() can never
# destroy a running QThread (Qt aborts the process then: "QThread: Destroyed while thread is still running",
# which pytest reported as "Fatal Python error: Aborted"). The pair is released when the thread really finished,
# so nothing outlives the lookup (D-106).
_ACTIVE: "set[tuple[QThread, QObject]]" = set()


def _retire(pair: tuple["QThread", "QObject"]) -> None:
    """A lookup thread finished: release its objects and let Qt free them in the main thread."""
    _ACTIVE.discard(pair)
    pair[0].deleteLater()


def load_pixmap(path: Path | None) -> QPixmap | None:
    """Read a cached poster file; a broken or missing file is simply ignored."""
    if not path:
        return None
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    pixmap = QPixmap()
    return pixmap if data and pixmap.loadFromData(data) else None


class _LookupWorker(QObject):
    """Search plus poster download, always off the UI thread."""

    finished = Signal(list)      # list[SeriesCandidate]

    def __init__(self, query: str, cache_dir: Path, fetch: Callable[[str], bytes]):
        super().__init__()
        self._query = query
        self._cache_dir = Path(cache_dir)
        self._fetch = fetch

    @Slot()
    def run(self) -> None:
        candidates = series_lookup.search(self._query, MAX_SUGGESTIONS, fetch=self._fetch)
        with_posters = []
        for candidate in candidates:
            image_path = (series_lookup.cache_poster(candidate.image_url, self._cache_dir, fetch=self._fetch)
                          if candidate.image_url else None)
            with_posters.append(SeriesCandidate(candidate.source_id, candidate.name, candidate.year,
                                                candidate.image_url, image_path))
        self.finished.emit(with_posters)


class SeriesSuggestions(QObject):
    """Owns the completer, the debounce timer and the poster preview of one series field."""

    searched = Signal(str, int)        # query, number of suggestions (for the status line)

    def __init__(self, edit: QLineEdit, preview: QLabel, repo: SeriesRepo | None, cache_dir: Path,
                 fetch: Callable[[str], bytes] | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._edit = edit
        self._preview = preview
        self._repo = repo
        self._cache_dir = Path(cache_dir)
        self._fetch = fetch
        self._current: SeriesCandidate | None = None
        self._thread: QThread | None = None
        self._worker: _LookupWorker | None = None
        self._stopped = False      # set by stop(): a late result must not touch a closed database (D-106)

        self._model = QStandardItemModel(0, 1, self)
        completer = QCompleter(self._model, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.activated.connect(self._on_activated)
        edit.setCompleter(completer)
        self._completer = completer

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._start_lookup)
        edit.textEdited.connect(self._on_text_edited)

        self._preview.setFixedSize(*PREVIEW_SIZE)
        self._preview.setScaledContents(True)
        self.show_saved_suggestions()

    # -- kept series ---------------------------------------------------------------------------

    @Slot()
    def show_saved_suggestions(self) -> None:
        """Offer the kept series first, from the database, with their cached posters (no network)."""
        rows = self._repo.saved() if self._repo is not None else []
        candidates = [SeriesCandidate(str(row["source_id"] or ""), row["name"], None, row["image_url"],
                                      Path(row["image_path"]) if row["image_path"] else None)
                      for row in rows]
        self._fill(candidates)
        self._show_preview_for(self._edit.text().strip())

    def refresh_after_external_change(self) -> None:
        """Called when something else filled the series field (a detected name, a queue entry)."""
        self._current = None
        self._on_text_edited(self._edit.text())

    def current(self) -> SeriesCandidate | None:
        """The chosen suggestion, or a bare candidate carrying whatever the user typed."""
        typed = self._edit.text().strip()
        if self._current is not None and self._current.name.lower() == typed.lower():
            return self._current
        found = self._repo.find(typed) if self._repo is not None and typed else None
        if found:
            return SeriesCandidate(str(found["source_id"] or ""), found["name"], None, found["image_url"],
                                   Path(found["image_path"]) if found["image_path"] else None)
        return SeriesCandidate("", typed) if typed else None

    # -- lookup --------------------------------------------------------------------------------

    @Slot(str)
    def _on_text_edited(self, text: str) -> None:
        self._show_preview_for(text.strip())
        if len(text.strip()) >= MIN_QUERY_CHARS:
            self._timer.start()
        else:
            self._timer.stop()

    @Slot()
    def _start_lookup(self) -> None:
        query = self._edit.text().strip()
        # No fetcher = no lookup: the app injects series_lookup.http_get, a test window injects nothing and can
        # therefore never reach the network.
        if len(query) < MIN_QUERY_CHARS or self._thread is not None or self._fetch is None:
            return
        # The thread has NO parent on purpose: destroying a running QThread aborts the process ("QThread:
        # Destroyed while thread is still running", seen as "Fatal Python error: Aborted" when a window was closed
        # during a lookup). We own it, keep it referenced and stop it in stop() (D-106).
        thread = QThread()
        worker = _LookupWorker(query, self._cache_dir, self._fetch)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_results)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)
        _ACTIVE.add((thread, worker))
        thread.finished.connect(lambda pair=(thread, worker): _retire(pair))
        self._thread, self._worker = thread, worker
        thread.start()

    @Slot(list)
    def _on_results(self, candidates: list) -> None:
        if self._stopped:
            return                                     # the window is closing; its database is gone
        self._fill(candidates)
        self._show_preview_for(self._edit.text().strip())
        self.searched.emit(self._edit.text().strip(), len(candidates))

    @Slot()
    def _on_thread_finished(self) -> None:
        # Only now is the thread really stopped, so the worker (and the thread) may be dropped and freed safely.
        thread, self._thread, self._worker = self._thread, None, None
        if thread is not None:
            thread.deleteLater()

    def wait_for_lookup(self) -> None:
        """Test helper: the lookup thread, when one is running."""
        if self._thread is not None:
            self._thread.wait(15000)

    def stop(self) -> None:
        """Stop the lookup thread and wait for it; safe to call when no lookup runs (D-106)."""
        self._stopped = True
        self._timer.stop()
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(5000):
                thread.terminate()
                thread.wait(2000)
        self._on_thread_finished()

    # -- internals -----------------------------------------------------------------------------

    def _fill(self, candidates: list[SeriesCandidate]) -> None:
        self._model.clear()
        for candidate in candidates:
            item = QStandardItem(candidate.label)
            pixmap = load_pixmap(candidate.image_path)
            if pixmap is not None:
                item.setIcon(QIcon(pixmap.scaled(*PREVIEW_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)))
            item.setData(candidate, Qt.ItemDataRole.UserRole)
            self._model.appendRow(item)

    @Slot(str)
    def _on_activated(self, text: str) -> None:
        for row in range(self._model.rowCount()):
            item = self._model.item(row)
            if item is not None and item.text() == text:
                candidate = item.data(Qt.ItemDataRole.UserRole)
                if candidate is not None:
                    self._current = candidate
                    self._edit.setText(candidate.name)
                    self._paint(candidate.image_path)
                return

    def _show_preview_for(self, name: str) -> None:
        if not name or self._repo is None:
            self._paint(None)
            return
        found = self._repo.find(name)
        self._paint(Path(found["image_path"]) if found and found["image_path"] else None)

    def _paint(self, image_path: Path | None) -> None:
        pixmap = load_pixmap(image_path)
        self._preview.setPixmap(pixmap.scaled(*PREVIEW_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation) if pixmap else QPixmap())
