"""Software video surface for the review window (D-120).

QVideoWidget draws through the GPU, and on some Windows machines (hybrid graphics, old drivers) it stays black while
the audio plays. This widget receives the decoded frames through a QVideoSink and paints them as an ordinary
QPixmap, which does not depend on the GPU video path. `frames` counts the pictures received, so the window can tell
"playing but nothing to show" from "playing".
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QVideoFrame, QVideoSink
from PySide6.QtWidgets import QLabel, QSizePolicy


class VideoSurface(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)   # a big frame must not resize it
        self.frames = 0
        self._image: QImage | None = None
        self.sink = QVideoSink(self)
        self.sink.videoFrameChanged.connect(self._on_frame)

    def _on_frame(self, frame: QVideoFrame) -> None:
        if not frame.isValid():
            return
        image = frame.toImage()
        if image.isNull():
            return
        self.frames += 1
        self._image = image
        self._paint()

    def _paint(self) -> None:
        if self._image is None or self.width() < 2 or self.height() < 2:
            return
        self.setPixmap(QPixmap.fromImage(self._image).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._paint()

    def clear_picture(self) -> None:
        self._image = None
        self.clear()
