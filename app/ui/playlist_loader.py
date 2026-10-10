"""Reads a playlist on a worker thread (D-120).

yt-dlp needs one or more network requests per playlist page (and reads the browser's cookies when they are enabled);
on the GUI thread this froze the window for seconds to minutes on long playlists.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QThread, Signal

# Running loaders are referenced here as well, so a window closed while one runs can never destroy a running QThread
# (the same rule as app.ui.link_probe, D-106).
_ACTIVE: "set[PlaylistLoader]" = set()


class PlaylistLoader(QThread):
    loaded = Signal(str, list)          # url, entries [{"url", "title"}]
    failed = Signal(str, str)           # url, error text

    def __init__(self, url: str, extract: Callable[[str], list], parent=None):
        super().__init__(parent)
        self.url = url
        self._extract = extract
        _ACTIVE.add(self)
        self.finished.connect(lambda: _ACTIVE.discard(self))

    def run(self) -> None:
        try:
            entries = self._extract(self.url) or []
        except Exception as exc:      # noqa: BLE001 - reported to the window, never raised in the thread
            self.failed.emit(self.url, str(exc))
            return
        self.loaded.emit(self.url, list(entries))
