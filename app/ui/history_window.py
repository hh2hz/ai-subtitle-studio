"""Job history dialog listing recent jobs with resume and open folder actions."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.database.database import Database
from app.database.jobs import JobsRepo
from app.utils.i18n import Translator


class JobHistoryDialog(QDialog):
    def __init__(self, main_window, translator: Translator, db_path: Path | str | None, parent: QWidget | None = None):
        super().__init__(parent or main_window)
        self._main_window = main_window
        self._tr = translator
        self._db_path = db_path
        self.setObjectName("JobHistoryDialog")
        self.resize(880, 520)
        self._build_ui()
        self.retranslate_ui()
        self.reload()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header_row = QHBoxLayout()
        self.title_label = QLabel()
        self.title_label.setObjectName("headerTitle")
        self.count_label = QLabel()
        self.count_label.setObjectName("mutedLabel")
        header_row.addWidget(self.title_label)
        header_row.addWidget(self.count_label)
        header_row.addStretch(1)

        self.refresh_button = QPushButton()
        self.refresh_button.clicked.connect(self.reload)
        header_row.addWidget(self.refresh_button)
        layout.addLayout(header_row)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch(1)
        self.close_button = QPushButton()
        self.close_button.clicked.connect(self.accept)
        bottom_row.addWidget(self.close_button)
        layout.addLayout(bottom_row)

    def retranslate_ui(self) -> None:
        t = self._tr.t
        self.setWindowTitle(t("history.title"))
        self.title_label.setText(t("history.title"))
        self.refresh_button.setText(t("history.refresh"))
        self.close_button.setText(t("history.close"))
        self.table.setHorizontalHeaderLabels([
            t("history.column_id"),
            t("history.column_date"),
            t("history.column_input"),
            t("history.column_languages"),
            t("history.column_mode"),
            t("history.column_status"),
            t("history.column_actions"),
        ])
        self._update_count_label()

    def _update_count_label(self) -> None:
        count = self.table.rowCount()
        self.count_label.setText(self._tr.t("history.count", count=count))

    @Slot()
    def reload(self) -> None:
        if not self._db_path:
            self.table.setRowCount(0)
            self._update_count_label()
            return
        db = Database(self._db_path)
        try:
            repo = JobsRepo(db)
            jobs = repo.recent(limit=50)
        finally:
            db.close()

        self.table.setRowCount(len(jobs))
        t = self._tr.t
        for row_idx, job in enumerate(jobs):
            job_id = job["id"]

            # 0: ID
            id_item = QTableWidgetItem(str(job_id))
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 0, id_item)

            # 1: Date
            raw_date = job.get("created_at") or ""
            date_str = raw_date[:16].replace("T", " ") if "T" in raw_date else raw_date
            date_item = QTableWidgetItem(date_str)
            date_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 1, date_item)

            # 2: Input / Title
            display_name = ""
            if job.get("queue_config"):
                try:
                    cfg = json.loads(job["queue_config"])
                    display_name = cfg.get("title", "")
                except Exception:
                    pass
            if not display_name:
                display_name = job.get("episode_title") or ""
            if not display_name and job.get("series_name"):
                s_info = job["series_name"]
                if job.get("season") and job.get("episode"):
                    s_info += f" S{job['season']:02d}E{job['episode']:02d}"
                display_name = s_info
            if not display_name:
                val = job.get("input_value", "")
                display_name = Path(val).name if job.get("input_type") == "file" else val
            title_item = QTableWidgetItem(display_name)
            title_item.setData(Qt.ItemDataRole.UserRole, job_id)
            title_item.setToolTip(job.get("input_value", ""))
            self.table.setItem(row_idx, 2, title_item)

            # 3: Languages
            src = job.get("source_language", "")
            tgt = job.get("target_language", "")
            lang_item = QTableWidgetItem(f"{src} \u2192 {tgt}")
            lang_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 3, lang_item)

            # 4: Mode
            raw_mode = job.get("mode", "")
            mode_item = QTableWidgetItem(t(f"mode.{raw_mode}", default=raw_mode.capitalize()))
            mode_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 4, mode_item)

            # 5: Status
            st = job.get("status", "pending")
            status_item = QTableWidgetItem(t(f"queue.status_{st}", default=st.capitalize()))
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 5, status_item)

            # 6: Actions (Resume, Open Folder)
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(4, 2, 4, 2)
            actions_layout.setSpacing(6)

            resume_btn = QPushButton(t("history.resume"))
            resume_btn.setEnabled(st != "running")
            resume_btn.clicked.connect(lambda _, jid=job_id: self._on_resume(jid))
            actions_layout.addWidget(resume_btn)

            open_btn = QPushButton(t("history.open_folder"))
            out_dir = job.get("output_dir")
            open_btn.setEnabled(bool(out_dir and Path(out_dir).exists()))
            if out_dir:
                open_btn.clicked.connect(lambda _, d=out_dir: self._main_window._open_dir(d))
            actions_layout.addWidget(open_btn)

            self.table.setCellWidget(row_idx, 6, actions_widget)

        self._update_count_label()

    def _on_resume(self, job_id: int) -> None:
        if self._main_window:
            self._main_window.resume_job(job_id)
            self.reload()
