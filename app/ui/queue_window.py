"""Batch queue dialog for multi-item media processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from app.utils.i18n import Translator

if TYPE_CHECKING:
    from app.ui.main_window import MainWindow


class BatchQueueDialog(QDialog):
    def __init__(self, main_window: MainWindow, translator: Translator, parent: QWidget | None = None):
        super().__init__(parent or main_window)
        self._main_window = main_window
        self._tr = translator
        self.setObjectName("BatchQueueDialog")
        self.resize(880, 520)
        self._build_ui()
        self.retranslate_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header row: title, queue ETA, and execution controls
        header_row = QHBoxLayout()
        self.title_label = QLabel()
        self.title_label.setObjectName("headerTitle")
        self.eta_label = QLabel()
        self.eta_label.setObjectName("mutedLabel")

        header_row.addWidget(self.title_label)
        header_row.addSpacing(12)
        header_row.addWidget(self.eta_label)
        header_row.addStretch(1)

        self.start_button = QPushButton()
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._on_start_clicked)

        self.cancel_button = QPushButton()
        self.cancel_button.clicked.connect(self._on_cancel_clicked)

        header_row.addWidget(self.start_button)
        header_row.addWidget(self.cancel_button)
        layout.addLayout(header_row)

        # Action bar: Add files, folder, playlist, and clear finished
        action_bar = QHBoxLayout()
        self.add_files_button = QPushButton()
        self.add_files_button.clicked.connect(self._main_window._browse_queue_files)

        self.add_folder_button = QPushButton()
        self.add_folder_button.clicked.connect(self._main_window._browse_queue_folder)

        self.add_playlist_button = QPushButton()
        self.add_playlist_button.clicked.connect(self._main_window._browse_queue_playlist)

        self.clear_button = QPushButton()
        self.clear_button.clicked.connect(self._main_window.clear_completed_queue)

        action_bar.addWidget(self.add_files_button)
        action_bar.addWidget(self.add_folder_button)
        action_bar.addWidget(self.add_playlist_button)
        action_bar.addStretch(1)
        action_bar.addWidget(self.clear_button)
        layout.addLayout(action_bar)

        # Queue table
        self.queue_table = QTableWidget()
        self.queue_table.setColumnCount(5)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.queue_table, 1)

        # Bottom row
        bottom_row = QHBoxLayout()
        self.count_label = QLabel()
        self.count_label.setObjectName("mutedLabel")
        bottom_row.addWidget(self.count_label)
        bottom_row.addStretch(1)

        self.close_button = QPushButton()
        self.close_button.clicked.connect(self.hide)
        bottom_row.addWidget(self.close_button)
        layout.addLayout(bottom_row)

    def retranslate_ui(self) -> None:
        t = self._tr.t
        self.setWindowTitle(t("queue.title"))
        self.title_label.setText(t("queue.title"))
        self.add_files_button.setText(t("queue.add_files"))
        self.add_folder_button.setText(t("queue.add_folder"))
        self.add_playlist_button.setText(t("queue.add_playlist"))
        self.clear_button.setText(t("queue.clear_completed"))
        self.start_button.setText(t("queue.start"))
        self.cancel_button.setText(t("queue.cancel"))
        self.close_button.setText(t("queue.close"))
        self.queue_table.setHorizontalHeaderLabels([
            t("queue.column_order"),
            t("queue.column_title"),
            t("queue.column_status"),
            t("queue.column_eta"),
            t("queue.column_actions"),
        ])
        self._update_count_label()

    def _update_count_label(self) -> None:
        count = self.queue_table.rowCount()
        self.count_label.setText(self._tr.t("queue.count", count=count))

    @Slot()
    def _on_start_clicked(self) -> None:
        self._main_window.start_queue()

    @Slot()
    def _on_cancel_clicked(self) -> None:
        self._main_window.cancel_queue()
