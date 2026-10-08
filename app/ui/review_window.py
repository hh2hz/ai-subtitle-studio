"""Review window: video preview, line table with confidence and reasons, editing and approval."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QBrush, QCloseEvent, QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from app.core import confidence as conf
from app.core import finalize, preview_proxy
from app.core.exporter import format_timestamp
from app.core.review import ReviewSession
from app.ui.burn_runner import BurnRunner
from app.utils.i18n import Translator
from app.utils.languages import RTL_LANGUAGES

log = logging.getLogger(__name__)

CONTEXT_S = 5.0
_COLORS = {conf.LOW: QColor(220, 60, 60, 70), conf.MEDIUM: QColor(230, 170, 30, 60), conf.HIGH: QColor(0, 0, 0, 0)}
COL_ID, COL_TIME, COL_CONF, COL_SRC, COL_TGT, COL_ISSUES = range(6)


class _RetranslateWorker(QObject):
    finished = Signal(int, str, str)        # unit id, text ("" on failure), message (backward compat)
    line_finished = Signal(int, str, str)   # unit id, text, message
    all_finished = Signal(int)              # total processed count

    def __init__(self, session: ReviewSession, unit_ids: int | list[int], refiner_factory):
        super().__init__()
        self._session = session
        self._unit_ids = [unit_ids] if isinstance(unit_ids, int) else list(unit_ids)
        self._factory = refiner_factory
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            media = {k: v for k, v in (self._session.doc.get("series") or {}).items() if v and k != "source"}
            media["title"] = (self._session.doc.get("media_info") or {}).get("title")
            refiner = self._factory(self._session.source_language, self._session.target_language, media)
            if refiner is None:
                for uid in self._unit_ids:
                    self.finished.emit(uid, "", "no_providers")
                    self.line_finished.emit(uid, "", "no_providers")
                self.all_finished.emit(0)
                return
            refiner.review = False
            glossary = (self._session.doc.get("stats", {}).get("refine", {}).get("glossary", {}))
            success_count = 0
            for uid in self._unit_ids:
                if self._cancelled:
                    break
                try:
                    units = self._session.units
                    idx = next(i for i, u in enumerate(units) if u["id"] == uid)
                    unit = units[idx]
                    previous = [{"source": u["text"], "translation": self._session.current_text(u)}
                                for u in units[max(0, idx - 6):idx]]
                    following = [{"source": u["text"]} for u in units[idx + 1:idx + 4]]
                    result = refiner.translate_block([{"id": unit["id"], "source": unit["text"], "draft": ""}],
                                                     previous, following, glossary)
                    text = result.translations.get(unit["id"], "")
                    msg = result.translator or "; ".join(result.notes)
                    self.finished.emit(uid, text, msg)
                    self.line_finished.emit(uid, text, msg)
                    if text:
                        success_count += 1
                except Exception as exc:
                    log.warning("Retranslating unit %d failed: %s", uid, exc)
                    self.finished.emit(uid, "", str(exc))
                    self.line_finished.emit(uid, "", str(exc))
            self.all_finished.emit(success_count)
        except Exception as exc:
            log.exception("Retranslation worker failed: %s", exc)
            self.all_finished.emit(0)


class _RetranscribeWorker(QObject):
    finished = Signal(int, str, str)   # unit_id, new_text, error

    def __init__(self, session: ReviewSession, unit_id: int, asr_factory):
        super().__init__()
        self._session, self._unit_id, self._asr_factory = session, unit_id, asr_factory

    @Slot()
    def run(self) -> None:
        try:
            text = self._session.retranscribe_unit(self._unit_id, self._asr_factory)
            self.finished.emit(self._unit_id, text, "")
        except Exception as exc:
            log.exception("Retranscribe span failed: %s", exc)
            self.finished.emit(self._unit_id, "", str(exc))


class TimelineStrip(QWidget):
    cue_clicked = Signal(int)     # unit_id
    time_seek = Signal(float)     # seconds

    def __init__(self, session: ReviewSession, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_time: float = 0.0
        self._selected_id: int | None = None
        self.setMinimumHeight(38)
        self.setMaximumHeight(52)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_position(self, seconds: float) -> None:
        self._current_time = max(0.0, seconds)
        self.update()

    def set_selected_id(self, unit_id: int | None) -> None:
        self._selected_id = unit_id
        self.update()

    def _total_duration(self) -> float:
        if self.session.units:
            return max(1.0, max(u["end"] for u in self.session.units))
        return 10.0

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        total = self._total_duration()

        painter.fillRect(0, 0, w, h, QColor(24, 26, 32))

        # Ruler guideline
        painter.setPen(QPen(QColor(55, 60, 75), 1))
        painter.drawLine(0, h // 2, w, h // 2)

        box_y = 4
        box_h = max(10, h - 18)
        for u in self.session.units:
            x1 = int((u["start"] / total) * w)
            x2 = int((u["end"] / total) * w)
            box_w = max(3, x2 - x1)
            is_selected = (u["id"] == self._selected_id)

            if u.get("approved"):
                bg = QColor(46, 160, 75, 200)
            elif u["confidence"] == conf.LOW:
                bg = QColor(215, 60, 60, 200)
            elif u["confidence"] == conf.MEDIUM:
                bg = QColor(225, 155, 30, 200)
            else:
                bg = QColor(70, 130, 215, 180)

            painter.fillRect(x1, box_y, box_w, box_h, bg)
            if is_selected:
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            else:
                painter.setPen(QPen(QColor(255, 255, 255, 50), 1))
            painter.drawRect(x1, box_y, box_w, box_h)

        # Playhead
        px = int((self._current_time / total) * w)
        painter.setPen(QPen(QColor(245, 65, 65), 2))
        painter.drawLine(px, 0, px, h)

        # Time labels
        painter.setPen(QPen(QColor(150, 155, 170), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(4, h - 3, "00:00")
        dur_str = f"{int(total // 60):02d}:{int(total % 60):02d}"
        painter.drawText(max(0, w - 38), h - 3, dur_str)

    def mousePressEvent(self, event) -> None:
        total = self._total_duration()
        w = max(1, self.width())
        t = (event.position().x() / w) * total
        clicked = self.session.unit_at(t)
        if clicked:
            self.cue_clicked.emit(clicked["id"])
        self.time_seek.emit(t)


class ShiftDialog(QDialog):
    def __init__(self, translator: Translator, parent=None):
        super().__init__(parent)
        self._tr = translator
        self.setWindowTitle(translator.t("review.shift_title"))
        self.resize(320, 160)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(translator.t("review.shift_seconds")))
        self.delta_spin = QDoubleSpinBox()
        self.delta_spin.setRange(-3600.0, 3600.0)
        self.delta_spin.setDecimals(3)
        self.delta_spin.setSingleStep(0.1)
        self.delta_spin.setValue(0.5)
        layout.addWidget(self.delta_spin)

        layout.addWidget(QLabel(translator.t("review.shift_scope")))
        self.scope_combo = QComboBox()
        self.scope_combo.addItems([
            translator.t("review.shift_current"),
            translator.t("review.shift_forward"),
            translator.t("review.shift_all"),
        ])
        layout.addWidget(self.scope_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[float, int]:
        return self.delta_spin.value(), self.scope_combo.currentIndex()


class FindReplaceDialog(QDialog):
    find_next_requested = Signal(str, bool)
    replace_requested = Signal(str, str, bool)
    replace_all_requested = Signal(str, str, bool, str)

    def __init__(self, translator: Translator, parent=None):
        super().__init__(parent)
        self._tr = translator
        self.setWindowTitle(translator.t("review.find_title"))
        self.resize(360, 220)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(translator.t("review.find_label")))
        self.find_edit = QLineEdit()
        layout.addWidget(self.find_edit)

        layout.addWidget(QLabel(translator.t("review.replace_label")))
        self.replace_edit = QLineEdit()
        layout.addWidget(self.replace_edit)

        self.match_case = QCheckBox(translator.t("review.match_case"))
        layout.addWidget(self.match_case)

        self.scope_combo = QComboBox()
        self.scope_combo.addItems([
            translator.t("review.find_scope_episode"),
            translator.t("review.find_scope_series"),
        ])
        layout.addWidget(self.scope_combo)

        btn_row = QHBoxLayout()
        self.find_btn = QPushButton(translator.t("review.find_next"))
        self.replace_btn = QPushButton(translator.t("review.replace_btn"))
        self.replace_all_btn = QPushButton(translator.t("review.replace_all"))

        self.find_btn.clicked.connect(lambda: self.find_next_requested.emit(
            self.find_edit.text(), self.match_case.isChecked()))
        self.replace_btn.clicked.connect(lambda: self.replace_requested.emit(
            self.find_edit.text(), self.replace_edit.text(), self.match_case.isChecked()))
        self.replace_all_btn.clicked.connect(lambda: self.replace_all_requested.emit(
            self.find_edit.text(), self.replace_edit.text(), self.match_case.isChecked(),
            "episode" if self.scope_combo.currentIndex() == 0 else "series"))

        btn_row.addWidget(self.find_btn)
        btn_row.addWidget(self.replace_btn)
        btn_row.addWidget(self.replace_all_btn)
        layout.addLayout(btn_row)


class PreviewCopyRunner(QThread):
    """Builds the small H.264 preview copy (app.core.preview_proxy) without blocking the window."""

    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, video: Path, parent=None):
        super().__init__(parent)
        self._video = video
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            self.succeeded.emit(str(preview_proxy.make_preview(self._video, cancel=self.cancel_event)))
        except Exception as exc:          # noqa: BLE001 - the window shows a short message, details are logged
            log.warning("Preview copy failed: %s", exc)
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.cancel_event.set()


class ReviewWindow(QMainWindow):
    def __init__(self, episode_dir: Path, translator: Translator, refiner_factory=None, parent=None,
                 enable_media: bool = True, asr_factory=None):
        super().__init__(parent)
        self._tr = translator
        self.session = ReviewSession(episode_dir)
        self._refiner_factory = refiner_factory
        self._asr_factory = asr_factory
        self._stop_at: float | None = None
        self._thread: QThread | None = None
        self._current_id: int | None = None
        self._rows: list[int] = []
        self.player = None
        self._proxy_tried = False
        self._proxy_runner = None
        self._first_play_checked = False
        self._first_play_frames = 0
        self.setObjectName("ReviewWindow")
        self.resize(1300, 850)
        self._build(enable_media)
        self._populate()
        self.retranslate_ui()
        self._tr.language_changed.connect(lambda _: self.retranslate_ui())
        if self._rows:
            self.select_row(0)

    # -- construction --------------------------------------------------------------------------

    def _build(self, enable_media: bool) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.currentIndexChanged.connect(self._populate)
        self.counts_label = QLabel()
        self.save_button = QPushButton()
        self.save_button.clicked.connect(self.save)
        self.burn_button = QPushButton()
        self.burn_button.clicked.connect(self.burn_video)
        self.burn_button.setEnabled(self.session.video_path is not None)
        self._burn_runner = None
        top.addWidget(self.filter_combo)
        top.addWidget(self.counts_label, 1)
        top.addWidget(self.save_button)
        top.addWidget(self.burn_button)
        root.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        self.video_widget = None
        if enable_media and self.session.video_path:
            try:
                from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

                from app.ui.video_surface import VideoSurface

                self.video_widget = VideoSurface()           # frames painted in software, not by the GPU (D-120)
                self.player = QMediaPlayer(self)
                self._audio = QAudioOutput(self)
                self.player.setAudioOutput(self._audio)
                self.player.setVideoSink(self.video_widget.sink)
                self.player.setSource(QUrl.fromLocalFile(str(self.session.video_path)))
                self.player.positionChanged.connect(self._on_position)
                self.player.errorOccurred.connect(self._on_player_error)
                self.player.mediaStatusChanged.connect(self._on_media_status)
            except Exception as exc:
                log.warning("Video preview unavailable: %s", exc)
                self.player = None
        if self.video_widget is not None:
            self.video_widget.setMinimumSize(480, 260)
            video_layout.addWidget(self.video_widget, 1)
        else:
            self.no_video_label = QLabel()
            self.no_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            video_layout.addWidget(self.no_video_label, 1)

        self.overlay = QLabel()
        self.overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay.setWordWrap(True)
        self.overlay.setMinimumHeight(45)
        self.overlay.setStyleSheet("font-size: 18px; padding: 4px;")
        video_layout.addWidget(self.overlay)

        # Timeline strip
        self.timeline = TimelineStrip(self.session, self)
        self.timeline.cue_clicked.connect(self._select_unit_id)
        self.timeline.time_seek.connect(self._seek_to_time)
        video_layout.addWidget(self.timeline)

        controls = QHBoxLayout()
        self.prev_issue_button, self.prev_button = QPushButton(), QPushButton()
        self.play_line_button, self.pause_button = QPushButton(), QPushButton()
        self.next_button, self.next_issue_button = QPushButton(), QPushButton()
        for button, slot in ((self.prev_issue_button, lambda: self._jump(-1, issues=True)),
                             (self.prev_button, lambda: self._jump(-1)),
                             (self.play_line_button, self.play_line),
                             (self.pause_button, self._toggle_pause),
                             (self.next_button, lambda: self._jump(1)),
                             (self.next_issue_button, lambda: self._jump(1, issues=True))):
            button.clicked.connect(slot)
            controls.addWidget(button)
        for button in (self.play_line_button, self.pause_button):
            button.setEnabled(self.player is not None)
        video_layout.addLayout(controls)
        splitter.addWidget(video_panel)

        self.table = QTableWidget(0, 6)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for col in (COL_ID, COL_TIME, COL_CONF):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        for col in (COL_SRC, COL_TGT, COL_ISSUES):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_select)
        splitter.addWidget(self.table)
        splitter.setSizes([520, 780])
        root.addWidget(splitter, 1)

        # Cue operations toolbar
        cue_bar = QHBoxLayout()
        self.split_button = QPushButton()
        self.split_button.clicked.connect(self.split_line)
        self.merge_button = QPushButton()
        self.merge_button.clicked.connect(self.merge_line)
        self.shift_button = QPushButton()
        self.shift_button.clicked.connect(self.shift_timing)
        self.delete_button = QPushButton()
        self.delete_button.clicked.connect(self.delete_line)
        self.find_replace_button = QPushButton()
        self.find_replace_button.clicked.connect(self.open_find_replace)
        self.retranscribe_button = QPushButton()
        self.retranscribe_button.clicked.connect(self.retranscribe_span)

        for b in (self.split_button, self.merge_button, self.shift_button, self.delete_button,
                  self.find_replace_button, self.retranscribe_button):
            cue_bar.addWidget(b)
        cue_bar.addStretch(1)
        root.addLayout(cue_bar)

        editor = QVBoxLayout()
        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setMaximumHeight(90)
        self.text_edit.setStyleSheet("font-size: 16px;")
        is_rtl = self.session.target_language.split("-")[0].lower() in RTL_LANGUAGES
        self.text_edit.setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl else Qt.LayoutDirection.LeftToRight)

        self.reasons_label = QLabel()
        self.reasons_label.setWordWrap(True)
        self.suggestion_label = QLabel()
        self.suggestion_label.setWordWrap(True)
        buttons = QHBoxLayout()
        self.use_suggestion_button = QPushButton()
        self.use_suggestion_button.clicked.connect(self._use_suggestion)
        self.retranslate_button = QPushButton()
        self.retranslate_button.clicked.connect(self.retranslate_lines)
        self.retranslate_button.setEnabled(self._refiner_factory is not None)
        self.revert_button = QPushButton()
        self.revert_button.clicked.connect(self._revert)
        self.approve_button = QPushButton()
        self.approve_button.clicked.connect(self.approve_and_next)
        for b in (self.use_suggestion_button, self.retranslate_button, self.revert_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.approve_button)
        for w in (self.source_label, self.text_edit, self.reasons_label, self.suggestion_label):
            editor.addWidget(w)
        editor.addLayout(buttons)
        root.addLayout(editor)
        self.setCentralWidget(central)

        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.approve_and_next)
        QShortcut(QKeySequence("Ctrl+Space"), self, activated=self.play_line)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save)
        QShortcut(QKeySequence("Ctrl+Down"), self, activated=lambda: self._jump(1, issues=True))
        QShortcut(QKeySequence("Ctrl+Up"), self, activated=lambda: self._jump(-1, issues=True))
        QShortcut(QKeySequence("F2"), self, activated=lambda: self._jump(1, issues=True))
        QShortcut(QKeySequence("Shift+F2"), self, activated=lambda: self._jump(-1, issues=True))
        QShortcut(QKeySequence("Ctrl+D"), self, activated=self.delete_line)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.open_find_replace)
        QShortcut(QKeySequence("Ctrl+H"), self, activated=self.open_find_replace)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, activated=self.split_line)
        QShortcut(QKeySequence("Ctrl+Shift+M"), self, activated=self.merge_line)

    def retranslate_ui(self) -> None:
        t = self._tr.t
        self.setWindowTitle(t("review.title", name=self.session.episode_dir.name))
        index = max(self.filter_combo.currentIndex(), 0)
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItems([t("review.filter_review"), t("review.filter_all")])
        self.filter_combo.setCurrentIndex(index)
        self.filter_combo.blockSignals(False)
        self.save_button.setText(t("review.save"))
        self.burn_button.setText(t("review.burn"))
        self.prev_issue_button.setText(t("review.prev_issue"))
        self.prev_button.setText(t("review.prev"))
        self.play_line_button.setText(t("review.play_line"))
        self.pause_button.setText(t("review.pause"))
        self.next_button.setText(t("review.next"))
        self.next_issue_button.setText(t("review.next_issue"))
        self.split_button.setText(t("review.split"))
        self.merge_button.setText(t("review.merge"))
        self.shift_button.setText(t("review.shift"))
        self.delete_button.setText(t("review.delete"))
        self.find_replace_button.setText(t("review.find_replace"))
        self.retranscribe_button.setText(t("review.retranscribe"))
        self.use_suggestion_button.setText(t("review.use_suggestion"))
        self.retranslate_button.setText(t("review.retranslate"))
        self.revert_button.setText(t("review.revert"))
        self.approve_button.setText(t("review.approve"))
        self.table.setHorizontalHeaderLabels([t("review.col_id"), t("review.col_time"), t("review.col_conf"),
                                              t("review.col_source"), t("review.col_translation"),
                                              t("review.col_issues")])
        if hasattr(self, "no_video_label"):
            self.no_video_label.setText(t("review.no_video"))
        self._update_counts()

    # -- table ---------------------------------------------------------------------------------

    def select_row(self, row: int) -> None:
        if 0 <= row < self.table.rowCount():
            self.table.setCurrentCell(row, 0)

    def _select_unit_id(self, unit_id: int) -> None:
        if unit_id in self._rows:
            self.select_row(self._rows.index(unit_id))

    def _reason_text(self, unit: dict) -> str:
        return "; ".join(self._tr.t(f"flag.{code}") for code in finalize.reasons(unit))

    def _populate(self) -> None:
        only_review = self.filter_combo.currentIndex() <= 0
        selected = self._current_id
        units = [u for u in self.session.units if not only_review or self.session.needs_review(u)]
        if only_review and not units:
            units = list(self.session.units)
        self._rows = [u["id"] for u in units]
        self.table.blockSignals(True)
        self.table.setRowCount(len(units))
        for row, unit in enumerate(units):
            self._fill_row(row, unit)
        self.table.blockSignals(False)
        if selected in self._rows:
            self.select_row(self._rows.index(selected))
        self._update_counts()
        if hasattr(self, "timeline"):
            self.timeline.update()

    def _fill_row(self, row: int, unit: dict) -> None:
        state = self._tr.t("review.approved") if unit.get("approved") else self._tr.t(f"confidence.{unit['confidence']}")
        values = [str(unit["id"] + 1), format_timestamp(unit["start"])[:8], state, unit["text"],
                  unit["final_text"].replace("\n", " "), self._reason_text(unit)]
        color = QColor(0, 0, 0, 0) if unit.get("approved") else _COLORS[unit["confidence"]]
        is_rtl = self.session.target_language.split("-")[0].lower() in RTL_LANGUAGES
        tgt_align = Qt.AlignmentFlag.AlignRight if is_rtl else Qt.AlignmentFlag.AlignLeft
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setBackground(QBrush(color))
            if col == COL_TGT:
                item.setTextAlignment(tgt_align | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, col, item)

    def _refresh_row(self, unit_id: int) -> None:
        if unit_id in self._rows:
            self._fill_row(self._rows.index(unit_id), self.session.unit(unit_id))
        self._update_counts()
        if hasattr(self, "timeline"):
            self.timeline.update()

    def _update_counts(self) -> None:
        c = self.session.counts()
        self.counts_label.setText(self._tr.t("review.counts", **c))
        self.setWindowModified(self.session.dirty)

    def _selected_unit_ids(self) -> list[int]:
        rows = sorted(set(idx.row() for idx in self.table.selectedIndexes()))
        if rows:
            return [self._rows[r] for r in rows if r < len(self._rows)]
        if self._current_id is not None:
            return [self._current_id]
        return []

    # -- selection / editing -------------------------------------------------------------------

    @Slot()
    def _on_select(self) -> None:
        self._commit_edit()
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        self._current_id = self._rows[rows[0].row()]
        unit = self.session.unit(self._current_id)
        self.source_label.setText(f"{format_timestamp(unit['start'])} --> {format_timestamp(unit['end'])}\n{unit['text']}")
        self.text_edit.setPlainText(self.session.current_text(unit))
        reasons = self._reason_text(unit)
        self.reasons_label.setText(self._tr.t("review.reasons", reasons=reasons) if reasons else "")
        suggestion = unit.get("review")
        self.suggestion_label.setText(self._tr.t("review.suggestion", text=suggestion["text"],
                                                 reason=suggestion.get("reason", "")) if suggestion else "")
        self.use_suggestion_button.setEnabled(bool(suggestion))
        self.revert_button.setEnabled(unit.get("reviewed_text") is not None)
        self.overlay.setText(self.session.current_text(unit))
        if hasattr(self, "timeline"):
            self.timeline.set_selected_id(self._current_id)
        if self.player is not None:
            self.player.setPosition(int(max(0.0, unit["start"] - CONTEXT_S) * 1000))

    def _commit_edit(self) -> None:
        if self._current_id is None:
            return
        try:
            unit = self.session.unit(self._current_id)
        except StopIteration:
            return
        text = self.text_edit.toPlainText().strip()
        if text != self.session.current_text(unit).strip():
            self.session.set_text(self._current_id, text)
            self._refresh_row(self._current_id)

    @Slot()
    def _use_suggestion(self) -> None:
        unit = self.session.unit(self._current_id)
        if unit.get("review"):
            self.text_edit.setPlainText(unit["review"]["text"])

    @Slot()
    def _revert(self) -> None:
        unit = self.session.unit(self._current_id)
        self.session.set_text(self._current_id, unit["translation"])
        self.text_edit.setPlainText(self.session.current_text(unit))
        self._refresh_row(self._current_id)
        self.revert_button.setEnabled(False)

    @Slot()
    def approve_and_next(self) -> None:
        if self._current_id is None:
            return
        self._commit_edit()
        self.session.approve(self._current_id)
        self._refresh_row(self._current_id)
        self._jump(1)

    # -- cue editing (split, merge, shift, delete) ---------------------------------------------

    @Slot()
    def split_line(self) -> None:
        if self._current_id is None:
            return
        self._commit_edit()
        split_time = (self.player.position() / 1000.0) if self.player is not None else None
        u1, u2 = self.session.split_unit(self._current_id, split_time)
        self._populate()
        self._select_unit_id(u1["id"])
        self.statusBar().showMessage(self._tr.t("review.split_done"), 4000)

    @Slot()
    def merge_line(self) -> None:
        if self._current_id is None:
            return
        self._commit_edit()
        try:
            idx = next(i for i, u in enumerate(self.session.units) if u["id"] == self._current_id)
        except StopIteration:
            return
        if idx + 1 >= len(self.session.units):
            return
        next_id = self.session.units[idx + 1]["id"]
        u = self.session.merge_units(self._current_id, next_id)
        self._populate()
        self._select_unit_id(u["id"])
        self.statusBar().showMessage(self._tr.t("review.merge_done"), 4000)

    @Slot()
    def delete_line(self) -> None:
        if self._current_id is None:
            return
        target_id = self._current_id
        self.session.delete_unit(target_id)
        self._current_id = None
        self._populate()
        self.statusBar().showMessage(self._tr.t("review.deleted_done"), 4000)

    @Slot()
    def shift_timing(self) -> None:
        dialog = ShiftDialog(self._tr, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            delta, scope = dialog.values()
            if scope == 0:
                target_ids = [self._current_id] if self._current_id is not None else []
            elif scope == 1:
                try:
                    idx = next(i for i, u in enumerate(self.session.units) if u["id"] == self._current_id)
                    target_ids = [u["id"] for u in self.session.units[idx:]]
                except StopIteration:
                    target_ids = []
            else:
                target_ids = None
            self.session.shift_cues(delta, target_ids)
            self._populate()
            if self._current_id is not None:
                self._select_unit_id(self._current_id)
            self.statusBar().showMessage(self._tr.t("review.shifted_done", delta=delta), 4000)

    # -- find and replace ----------------------------------------------------------------------

    @Slot()
    def open_find_replace(self) -> None:
        dialog = FindReplaceDialog(self._tr, self)
        dialog.find_next_requested.connect(self._find_next)
        dialog.replace_requested.connect(self._replace_current)
        dialog.replace_all_requested.connect(self._replace_all)
        dialog.exec()

    def _find_next(self, find_text: str, match_case: bool) -> None:
        if not find_text or not self._rows:
            return
        current_row = 0
        rows = self.table.selectionModel().selectedRows()
        if rows:
            current_row = rows[0].row() + 1
        n = len(self._rows)
        for offset in range(n):
            row = (current_row + offset) % n
            u = self.session.unit(self._rows[row])
            cur = self.session.current_text(u)
            match = (find_text in cur) if match_case else (find_text.lower() in cur.lower())
            if match:
                self.select_row(row)
                return

    def _replace_current(self, find_text: str, replace_text: str, match_case: bool) -> None:
        if self._current_id is None or not find_text:
            return
        u = self.session.unit(self._current_id)
        cur = self.session.current_text(u)
        import re
        flags = 0 if match_case else re.IGNORECASE
        pattern = re.compile(re.escape(find_text), flags)
        if pattern.search(cur):
            new_text = pattern.sub(replace_text, cur, count=1)
            self.session.set_text(self._current_id, new_text)
            self.text_edit.setPlainText(new_text)
            self._refresh_row(self._current_id)
        self._find_next(find_text, match_case)

    def _replace_all(self, find_text: str, replace_text: str, match_case: bool, scope: str) -> None:
        if not find_text:
            return
        self._commit_edit()
        if scope == "series":
            results = self.session.find_replace_series(find_text, replace_text, match_case)
            total = sum(results.values())
            self._populate()
            self.statusBar().showMessage(
                self._tr.t("review.replaced_series_count", count=total, episodes=len(results)), 6000)
        else:
            total = self.session.find_replace_episode(find_text, replace_text, match_case)
            self._populate()
            self.statusBar().showMessage(self._tr.t("review.replaced_count", count=total), 4000)

    # -- re-transcribe span --------------------------------------------------------------------

    @Slot()
    def retranscribe_span(self) -> None:
        if self._current_id is None:
            return
        if self._asr_factory is None:
            self.statusBar().showMessage(self._tr.t("review.retranscribe_no_engine"), 4000)
            return
        self.statusBar().showMessage(self._tr.t("review.retranscribing"), 3000)
        thread = QThread(self)
        worker = _RetranscribeWorker(self.session, self._current_id, self._asr_factory)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_retranscribed)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    @Slot(int, str, str)
    def _on_retranscribed(self, unit_id: int, new_text: str, error: str) -> None:
        if error:
            self.statusBar().showMessage(self._tr.t("review.retranscribe_failed", error=error), 5000)
        elif not new_text:
            # The session kept the old source text; only the source is ever replaced, never the translation.
            self.statusBar().showMessage(self._tr.t("review.retranscribe_empty"), 5000)
        else:
            self._refresh_row(unit_id)
            if unit_id == self._current_id:
                u = self.session.unit(unit_id)
                self.source_label.setText(
                    f"{format_timestamp(u['start'])} --> {format_timestamp(u['end'])}\n{new_text}")
            self.statusBar().showMessage(self._tr.t("review.retranscribed", text=new_text[:30]), 4000)

    # -- navigation / playback -----------------------------------------------------------------

    def _jump(self, step: int, issues: bool = False) -> None:
        if not self._rows:
            return
        rows = self.table.selectionModel().selectedRows()
        current = rows[0].row() if rows else -1
        row = current + step
        while 0 <= row < len(self._rows):
            if not issues or self.session.needs_review(self.session.unit(self._rows[row])):
                self.select_row(row)
                return
            row += step

    def _seek_to_time(self, seconds: float) -> None:
        if self.player is not None:
            self.player.setPosition(int(seconds * 1000))

    # -- preview that cannot be played (D-119) ---------------------------------------------------

    def _on_player_error(self, error, text: str = "") -> None:
        log.warning("Video preview error: %s %s (%s)", error, text, getattr(self.session, "video_path", None))
        self._use_preview_copy()

    def _on_media_status(self, status) -> None:
        from PySide6.QtMultimedia import QMediaPlayer

        statuses = QMediaPlayer.MediaStatus
        if status == statuses.InvalidMedia:
            log.warning("Video preview: the media backend rejected %s", self.session.video_path)
            self._use_preview_copy()
        elif status in (statuses.LoadedMedia, statuses.BufferedMedia) and self.player is not None:
            if not self.player.hasVideo():            # the sound is decodable but the picture is not
                log.warning("Video preview: no usable video track in %s", self.session.video_path)
                self._use_preview_copy()

    def _use_preview_copy(self) -> None:
        """The original does not play here: build a small H.264 copy in the background and play that (once)."""
        if self.player is None or self._proxy_tried or self.session.video_path is None:
            return
        self._proxy_tried = True
        self.overlay.setText(self._tr.t("review.preview_preparing"))
        runner = PreviewCopyRunner(self.session.video_path, self)
        runner.succeeded.connect(self._preview_copy_ready)
        runner.failed.connect(lambda message: self.overlay.setText(self._tr.t("review.preview_failed")))
        self._proxy_runner = runner
        runner.start()

    def _preview_copy_ready(self, path: str) -> None:
        if self.player is None:
            return
        position = self.player.position()
        self._first_play_checked = False              # watch the copy as well
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.setPosition(position)
        self.overlay.setText("")

    def _check_first_play(self) -> None:
        """Playing but no picture arrived: the backend accepted the file yet cannot show it."""
        from PySide6.QtMultimedia import QMediaPlayer

        if (self.player is None or self.video_widget is None
                or self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState
                or self.video_widget.frames != self._first_play_frames):
            return
        log.warning("Video preview: no picture after starting playback of %s (copy tried: %s)",
                    self.session.video_path, self._proxy_tried)
        if self._proxy_tried:
            self.overlay.setText(self._tr.t("review.preview_failed"))
        else:
            self._use_preview_copy()

    def _watch_first_play(self) -> None:
        if self._first_play_checked or self.player is None or self.video_widget is None:
            return
        self._first_play_checked = True
        self._first_play_frames = self.video_widget.frames
        QTimer.singleShot(5000, self._check_first_play)

    @Slot()
    def play_line(self) -> None:
        if self.player is None or self._current_id is None:
            return
        unit = self.session.unit(self._current_id)
        self._stop_at = unit["end"] + CONTEXT_S
        self.player.setPosition(int(max(0.0, unit["start"] - CONTEXT_S) * 1000))
        self.player.play()
        self._watch_first_play()

    @Slot()
    def _toggle_pause(self) -> None:
        if self.player is None:
            return
        from PySide6.QtMultimedia import QMediaPlayer

        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self._stop_at = None
            self.player.play()
            self._watch_first_play()

    @Slot(int)
    def _on_position(self, ms: int) -> None:
        seconds = ms / 1000
        unit = self.session.unit_at(seconds)
        self.overlay.setText(self.session.current_text(unit) if unit else "")
        if hasattr(self, "timeline"):
            self.timeline.set_position(seconds)
        if self._stop_at is not None and seconds >= self._stop_at:
            self._stop_at = None
            self.player.pause()

    # -- AI retranslation ----------------------------------------------------------------------

    @Slot()
    def retranslate_lines(self) -> None:
        unit_ids = self._selected_unit_ids()
        if not unit_ids or self._refiner_factory is None or self._thread is not None:
            return
        self.retranslate_button.setEnabled(False)
        self.retranslate_button.setText(self._tr.t("review.retranslating"))
        self._thread = QThread(self)
        self._worker = _RetranslateWorker(self.session, unit_ids, self._refiner_factory)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.line_finished.connect(self._on_line_retranslated)
        self._worker.all_finished.connect(self._on_retranslate_all_finished)
        self._worker.all_finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread_done)
        self._thread.start()

    # Alias for single-line backward compatibility:
    retranslate_line = retranslate_lines

    @Slot(int, str, str)
    def _on_line_retranslated(self, unit_id: int, text: str, message: str) -> None:
        if text:
            self.session.set_text(unit_id, text)
            self._refresh_row(unit_id)
            if unit_id == self._current_id:
                self.text_edit.setPlainText(text)
                self.reasons_label.setText(self._tr.t("review.retranslated", engine=message))
        elif not text and unit_id == self._current_id:
            self.reasons_label.setText(self._tr.t("review.retranslate_failed", error=message))

    @Slot(int)
    def _on_retranslate_all_finished(self, count: int) -> None:
        self.statusBar().showMessage(self._tr.t("review.retranslate_done", count=count), 5000)

    @Slot()
    def _thread_done(self) -> None:
        self._thread = None
        self.retranslate_button.setEnabled(True)
        self.retranslate_button.setText(self._tr.t("review.retranslate"))

    # -- saving ----------------------------------------------------------------------------------

    @Slot()
    def save(self) -> Path:
        self._commit_edit()
        path = self.session.save()
        self._populate()
        self.statusBar().showMessage(self._tr.t("review.saved", path=str(path)), 8000)
        return path

    @Slot()
    def burn_video(self) -> None:
        if self.session.video_path is None or (self._burn_runner is not None and self._burn_runner.isRunning()):
            return
        srt = self.save()
        self.burn_button.setEnabled(False)
        runner = BurnRunner(self.session.video_path, srt, parent=self)
        runner.progress.connect(lambda f: self.statusBar().showMessage(
            self._tr.t("review.burn_progress", percent=int(f * 100))))
        runner.succeeded.connect(lambda path: self.statusBar().showMessage(
            self._tr.t("status.burn_done", path=path), 15000))
        runner.failed.connect(lambda error: self.statusBar().showMessage(
            self._tr.t("status.burn_failed", error=self._tr.t(error)), 15000))
        runner.finished.connect(lambda: self.burn_button.setEnabled(True))
        self._burn_runner = runner
        runner.start()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._burn_runner is not None and self._burn_runner.isRunning():
            self._burn_runner.cancel()
            self._burn_runner.wait(10000)
        self._commit_edit()
        if self.session.dirty:
            answer = QMessageBox.question(
                self, self._tr.t("review.title", name=self.session.episode_dir.name), self._tr.t("review.unsaved"),
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Save:
                self.save()
        if self._proxy_runner is not None and self._proxy_runner.isRunning():
            self._proxy_runner.cancel()
            self._proxy_runner.wait(10000)
        if self.player is not None:
            self.player.stop()
        super().closeEvent(event)
