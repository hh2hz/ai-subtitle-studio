"""Settings dialogs (M2: subtitle sources)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from app.core.downloader import COOKIE_BROWSERS
from app.database.series import SeriesRepo
from app.database.settings import Settings
from app.services.api_keys import PROJECT_FILE, load_keys
from app.ui.api_keys_dialog import ApiKeysDialog
from app.ui.series_picker import load_pixmap
from app.utils.i18n import Translator


class ProviderSettingsDialog(QDialog):
    def __init__(self, translator: Translator, settings: Settings, parent: QWidget | None = None,
                 series_repo: "SeriesRepo | None" = None, poster_cache_dir: Path | None = None):
        super().__init__(parent)
        t = translator.t
        self._tr = translator
        self._settings = settings
        self.setWindowTitle(t("dialog.settings_title"))
        # The dialog holds four groups and can be taller than the screen, which pushed the Save/Cancel buttons
        # out of reach. The groups scroll; the buttons stay pinned below the scroll area.
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        outer.addWidget(scroll, 1)

        speech = QGroupBox(t("group.speech"))
        speech_form = QFormLayout(speech)
        self.enhance_combo = QComboBox()
        for value in ("auto", "on", "off"):
            self.enhance_combo.addItem(t(f"enhance.{value}"), value)
        self.enhance_combo.setCurrentIndex(max(self.enhance_combo.findData(settings.get("audio_enhance")), 0))
        speech_form.addRow(t("label.audio_enhance"), self.enhance_combo)
        self.diarization_check = QCheckBox(t("label.diarization"))
        self.diarization_check.setChecked(settings.get("diarization"))
        self.diarization_check.setToolTip(t("tip.diarization"))
        speech_form.addRow(self.diarization_check)
        self.shots_check = QCheckBox(t("label.snap_to_shots"))
        self.shots_check.setChecked(settings.get("snap_to_shots"))
        self.shots_check.setToolTip(t("tip.snap_to_shots"))
        speech_form.addRow(self.shots_check)
        layout.addWidget(speech)

        translation = QGroupBox(t("group.translation"))
        tlayout = QVBoxLayout(translation)
        engine_form = QFormLayout()
        self.engine_combo = QComboBox()
        self.engine_combo.addItem(t("engine.local"), "local")
        self.engine_combo.addItem(t("engine.madlad"), "madlad")
        self.engine_combo.setCurrentIndex(max(self.engine_combo.findData(settings.get("translation_engine")), 0))
        self.local_model_edit = self._ltr_edit(settings.get("local_model"))
        engine_form.addRow(t("label.translation_engine"), self.engine_combo)
        engine_form.addRow(t("label.local_model"), self.local_model_edit)
        tlayout.addLayout(engine_form)
        self.refine_check = QCheckBox(t("label.llm_refine"))
        self.refine_check.setChecked(settings.get("llm_refine"))
        self.correct_check = QCheckBox(t("label.llm_correct_only"))
        self.correct_check.setChecked(settings.get("llm_correct_only"))
        self.review_check = QCheckBox(t("label.llm_review"))
        self.review_check.setChecked(settings.get("llm_review"))
        for box in (self.refine_check, self.correct_check, self.review_check):
            tlayout.addWidget(box)
        providers = ", ".join(sorted(load_keys())) or "-"
        keys_label = QLabel(t("label.api_providers", providers=providers) + "\n"
                            + t("label.api_keys", path=str(PROJECT_FILE)))
        keys_label.setWordWrap(True)
        keys_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        tlayout.addWidget(keys_label)
        self._keys_label = keys_label
        self.api_keys_button = QPushButton(t("keys.open"))
        self.api_keys_button.setToolTip(t("keys.open_tip"))
        self.api_keys_button.clicked.connect(self._edit_api_keys)
        tlayout.addWidget(self.api_keys_button)
        layout.addWidget(translation)

        sources = QGroupBox(t("dialog.providers_title"))
        layout_sources = QVBoxLayout(sources)
        self.platform_check = QCheckBox(t("label.fetch_platform_subs"))
        self.platform_check.setChecked(settings.get("fetch_platform_subtitles"))
        layout_sources.addWidget(self.platform_check)
        self.burn_check = QCheckBox(t("label.burn_video"))
        self.burn_check.setChecked(settings.get("burn_video"))
        layout_sources.addWidget(self.burn_check)
        self.keep_cache_check = QCheckBox(t("label.keep_job_cache"))
        self.keep_cache_check.setChecked(settings.get("keep_job_cache"))
        self.keep_cache_check.setToolTip(t("tip.keep_job_cache"))
        layout_sources.addWidget(self.keep_cache_check)
        self.update_check = QCheckBox(t("label.update_check"))
        self.update_check.setChecked(settings.get("update_check"))
        self.update_check.setToolTip(t("tip.update_check"))
        layout_sources.addWidget(self.update_check)
        form = QFormLayout()
        self.opensubtitles_edit = self._key_edit(settings.get("opensubtitles_api_key"))
        self.subdl_edit = self._key_edit(settings.get("subdl_api_key"))
        form.addRow(t("label.opensubtitles_key"), self.opensubtitles_edit)
        form.addRow(t("label.subdl_key"), self.subdl_edit)
        layout_sources.addLayout(form)
        note = QLabel(t("dialog.providers_note"))
        note.setWordWrap(True)
        layout_sources.addWidget(note)
        layout.addWidget(sources)

        access = QGroupBox(t("group.access"))
        access_form = QFormLayout(access)
        self.cookies_combo = QComboBox()
        for value in ("", "browser", "file"):
            self.cookies_combo.addItem(t(f"cookies.{value or 'none'}"), value)
        self.cookies_combo.setCurrentIndex(max(self.cookies_combo.findData(settings.get("cookies_source")), 0))
        self.cookies_browser_combo = QComboBox()
        for name in COOKIE_BROWSERS:
            self.cookies_browser_combo.addItem(name.capitalize(), name)
        self.cookies_browser_combo.setCurrentIndex(
            max(self.cookies_browser_combo.findData(settings.get("cookies_browser")), 0))
        access_form.addRow(t("label.cookies_source"), self.cookies_combo)
        access_form.addRow(t("label.cookies_browser"), self.cookies_browser_combo)
        self.cookies_file_edit = self._ltr_edit(settings.get("cookies_file"))
        browse_button = QPushButton(t("button.browse"))
        browse_button.clicked.connect(self._choose_cookies_file)
        file_row = QHBoxLayout()
        file_row.addWidget(self.cookies_file_edit)
        file_row.addWidget(browse_button)
        access_form.addRow(t("label.cookies_file"), file_row)
        self.ipv4_check = QCheckBox(t("label.force_ipv4"))
        self.ipv4_check.setChecked(settings.get("force_ipv4"))
        access_form.addRow("", self.ipv4_check)
        access_note = QLabel(t("access.note"))
        access_note.setWordWrap(True)
        access_form.addRow(access_note)
        # Only the chosen source is editable, so the dialog never suggests a setting that is not in use.
        self.cookies_combo.currentIndexChanged.connect(self._update_access_fields)
        self._update_access_fields()
        layout.addWidget(access)

        if series_repo is not None:
            kept = QGroupBox(t("group.saved_series"))
            kept_layout = QVBoxLayout(kept)
            self.saved_series_list = QListWidget()
            self.saved_series_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
            self._saved_series = {row["id"]: row for row in series_repo.saved()}
            for series_id, row in self._saved_series.items():
                item = QListWidgetItem(row["name"])
                item.setData(Qt.ItemDataRole.UserRole, series_id)
                pixmap = load_pixmap(Path(row["image_path"]) if row["image_path"] else None)
                if pixmap is not None:
                    item.setIcon(QIcon(pixmap.scaled(24, 34, Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation)))
                self.saved_series_list.addItem(item)
            self._series_to_forget: set[int] = set()
            self._series_repo = series_repo
            self._poster_cache_dir = poster_cache_dir
            kept_layout.addWidget(self.saved_series_list)
            remove_button = QPushButton(t("series.remove_selected"))
            remove_button.clicked.connect(self._remove_selected_series)
            kept_layout.addWidget(remove_button)
            note_series = QLabel(t("label.saved_series_note"))
            note_series.setWordWrap(True)
            kept_layout.addWidget(note_series)
            layout.addWidget(kept)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        save_button.setText(t("button.save"))
        save_button.setObjectName("primaryButton")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(t("button.cancel"))
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        # Start at a size that fits the screen, and let the user resize freely (the content scrolls).
        screen = self.screen() or QApplication.primaryScreen()
        available_height = screen.availableGeometry().height() if screen else 720
        self.setMinimumSize(520, 360)
        self.resize(560, max(360, min(content.sizeHint().height() + 90, available_height - 120)))

    @staticmethod
    def _ltr_edit(value: str) -> QLineEdit:
        edit = QLineEdit(value)
        edit.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        return edit

    @Slot()
    def _choose_cookies_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, self._tr.t("dialog.cookies_file"), "", self._tr.t("dialog.cookies_filter"))
        if path:
            self.cookies_file_edit.setText(path)

    @Slot()
    def _remove_selected_series(self) -> None:
        """Mark the selected kept series for removal; Save applies it, Cancel keeps them (D-076)."""
        for item in self.saved_series_list.selectedItems():
            series_id = item.data(Qt.ItemDataRole.UserRole)
            if series_id is not None:
                self._series_to_forget.add(int(series_id))
            self.saved_series_list.takeItem(self.saved_series_list.row(item))

    @Slot()
    def _update_access_fields(self) -> None:
        source = self.cookies_combo.currentData()
        self.cookies_browser_combo.setEnabled(source == "browser")
        self.cookies_file_edit.setEnabled(source == "file")

    @staticmethod
    def _key_edit(value: str) -> QLineEdit:
        edit = QLineEdit(value)
        edit.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        edit.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        return edit

    @Slot()
    def _edit_api_keys(self) -> None:
        """Open the API keys window (the same one as on the first start); show the providers that now have keys."""
        dialog = ApiKeysDialog(self._tr, self)
        dialog.exec()
        if dialog.saved:
            t = self._tr.t
            providers = ", ".join(sorted(load_keys())) or "-"
            self._keys_label.setText(t("label.api_providers", providers=providers) + "\n"
                                     + t("label.api_keys", path=str(PROJECT_FILE)))

    def save(self) -> None:
        self._settings.set("audio_enhance", self.enhance_combo.currentData())
        self._settings.set("diarization", self.diarization_check.isChecked())
        self._settings.set("snap_to_shots", self.shots_check.isChecked())
        self._settings.set("fetch_platform_subtitles", self.platform_check.isChecked())
        self._settings.set("burn_video", self.burn_check.isChecked())
        self._settings.set("keep_job_cache", self.keep_cache_check.isChecked())
        self._settings.set("update_check", self.update_check.isChecked())
        self._settings.set("opensubtitles_api_key", self.opensubtitles_edit.text().strip())
        self._settings.set("subdl_api_key", self.subdl_edit.text().strip())
        self._settings.set("translation_engine", self.engine_combo.currentData())
        if self.local_model_edit.text().strip():
            self._settings.set("local_model", self.local_model_edit.text().strip())
        self._settings.set("llm_refine", self.refine_check.isChecked())
        self._settings.set("llm_correct_only", self.correct_check.isChecked())
        self._settings.set("llm_review", self.review_check.isChecked())
        self._settings.set("cookies_source", self.cookies_combo.currentData())
        self._settings.set("cookies_browser", self.cookies_browser_combo.currentData())
        self._settings.set("cookies_file", self.cookies_file_edit.text().strip())
        self._settings.set("force_ipv4", self.ipv4_check.isChecked())
        if getattr(self, "_series_repo", None) is not None:
            for series_id in sorted(getattr(self, "_series_to_forget", set())):
                self._series_repo.forget(series_id, getattr(self, "_poster_cache_dir", None))
        self.accept()
