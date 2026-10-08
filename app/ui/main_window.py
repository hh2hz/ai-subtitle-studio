"""Main application window (M0 shell: inputs, options, output, activity log)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QByteArray, Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QApplication,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app import __version__
from app.core.downloader import YtDlpAdapter, js_runtime_available
from app.core.metadata import SeriesInfo
from app.core.modes import Mode
from app.core.pipeline import JobConfig, JobResult
from app.database.database import Database
from app.database.jobs import JobsRepo
from app.database.series import SeriesRepo
from app.database.settings import Settings
from app.services import eta
from app.services.job_manager import JobRunner, PipelineFactory
from app.ui import theme
from app.ui.burn_runner import BurnRunner
from app.ui.icons import app_icon
from app.utils.open_folder import open_folder
from app.ui.link_probe import LinkProbe
from app.ui.queue_window import BatchQueueDialog
from app.ui.series_picker import SeriesSuggestions
from app.ui.settings_window import ProviderSettingsDialog
from app.utils import languages
from app.utils.i18n import Translator
from app.utils.paths import default_data_root, default_output_dir
from app.utils.validation import is_url

log = logging.getLogger(__name__)

# Default window size: measured on the user's 1920x1080 screen at 125 % scaling the usable area is 1536x816, so the
# window opens with a margin instead of sticking out (D-110).
DEFAULT_WINDOW_SIZE = (1320, 680)
#: Bumped whenever the default layout or size changes: a geometry saved by an older layout is ignored once, because
#: restoring it would bring the old (too large) window back (D-111).
GEOMETRY_VERSION = 2

VIDEO_PATTERNS = "*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.ts *.flv *.wmv *.mp3 *.wav *.m4a *.flac"


@dataclass(frozen=True)
class RuntimeContext:
    """What the window needs to run jobs; tests inject a fake pipeline factory."""

    db_path: Path
    pipeline_factory: PipelineFactory
    downloader: Any = None
    cache_dir: Path | None = None     # app cache folder (series posters); None = <data root>/cache
    series_fetch: Any = None          # HTTP getter for series suggestions; None = no lookup (tests stay offline)


class MainWindow(QMainWindow):
    def __init__(self, translator: Translator, settings: Settings,
                 runtime: RuntimeContext | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._tr = translator
        self._settings = settings
        self._runtime = runtime
        # Series picker storage: kept series and their cached posters (D-076). Closed in closeEvent.
        self._series_db = Database(runtime.db_path) if runtime is not None else None
        self._series_repo = SeriesRepo(self._series_db) if self._series_db is not None else None
        self._poster_cache_dir = (
            runtime.cache_dir if runtime is not None and runtime.cache_dir else default_data_root() / "cache"
        ) / "posters"
        self._runner: JobRunner | None = None
        self._queue_running = False
        self._queue_cancelled = False
        self._job_times: list[float] = []
        self._job_started = 0.0
        # ETA state: timings learned from this machine's finished jobs, and the running job's inputs.
        self._timings = eta.EMPTY_TIMINGS
        self._job_media_duration: float | None = None
        self._duration_cache: dict[str, float | None] = {}
        self._current_stage = ""
        self._stage_fraction = 0.0
        self._last_overall = 0.0
        self._last_output_dir: Path | None = None
        self._close_when_done = False
        self._language_actions: dict[str, QAction] = {}
        self._theme_mode = settings.get("theme")
        self._gpu_runner = None
        self._update_runner = None       # update check thread (D-118)
        self.setObjectName("MainWindow")
        if not self._restore_geometry():
            self.resize(*DEFAULT_WINDOW_SIZE)    # a size that fits a 1536x816 work area (D-110)
        self._build_ui()
        self._build_menus()
        self.output_edit.setText(settings.get("output_dir"))
        self._refresh_timings()
        self.retranslate_ui()
        self._load_queue_from_db()
        translator.language_changed.connect(self._on_language_changed)
        self.statusBar().showMessage(self._tr.t("status.ready"))
        # After the layout knows its real size, make sure the whole window is on the screen (D-109).
        QTimer.singleShot(0, self._fit_to_screen)

    # -- construction ----------------------------------------------------------------------

    def _restore_geometry(self) -> bool:
        """Restore the saved window geometry; False when there is nothing usable to restore (D-109, D-111)."""
        saved = self._settings.get("window_geometry")
        if not saved:
            return False
        if self._settings.get("window_geometry_version") != GEOMETRY_VERSION:
            log.info("Ignoring a window geometry saved by an older layout (version %s)",
                     self._settings.get("window_geometry_version"))
            return False
        try:
            restored = self.restoreGeometry(QByteArray.fromHex(saved.encode("ascii")))
        except (TypeError, ValueError, UnicodeEncodeError):
            log.info("Ignoring an unreadable saved window geometry")
            return False
        if not restored:
            return False
        self._fit_to_screen()          # a geometry saved on another monitor may be off-screen now
        return True

    def _save_geometry(self) -> None:
        """Remember size and position for the next start (hex, because settings hold JSON strings)."""
        self._settings.set("window_geometry", bytes(self.saveGeometry()).hex())
        self._settings.set("window_geometry_version", GEOMETRY_VERSION)

    def _fit_to_screen(self) -> None:
        """Move (and if needed shrink) the window so that its whole **frame** is inside the usable screen area.

        Two things made this fail before (D-111): the frame's title bar and borders are not part of `setGeometry`,
        so a window "fitted" to the screen was still 30 px too tall at 125 % scaling, and a stale saved geometry
        brought back a window larger than the screen. The margins are measured and subtracted here, and the position
        is applied with `move()` so the title bar stays visible.
        """
        screen = QGuiApplication.screenAt(self.frameGeometry().center()) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        margin_width = max(frame.width() - self.width(), 0)         # left + right border
        margin_height = max(frame.height() - self.height(), 0)      # title bar + bottom border
        width = max(min(self.width(), available.width() - margin_width), 1)
        height = max(min(self.height(), available.height() - margin_height), 1)
        x = min(max(frame.left(), available.left()), available.left() + available.width() - width - margin_width)
        y = min(max(frame.top(), available.top()), available.top() + available.height() - height - margin_height)
        if (x, y, width, height) != (frame.left(), frame.top(), self.width(), self.height()):
            log.info("Fitting the window into the screen: client %sx%s at %s,%s", width, height, x, y)
            self.resize(width, height)
            self.move(x + (frame.left() - self.x()), y + (frame.top() - self.y()))

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 12, 18, 10)
        root.setSpacing(12)

        # Header: app name, tagline, theme switch and settings.
        header = QFrame()
        header.setObjectName("header")
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(0, 0, 0, 0)
        self.logo_label = QLabel()
        icon = app_icon()
        if not icon.isNull():
            self.logo_label.setPixmap(icon.pixmap(40, 40))
        self.title_label = QLabel()
        self.title_label.setObjectName("headerTitle")
        self.subtitle_label = QLabel()
        self.subtitle_label.setObjectName("headerSubtitle")
        # A long tagline must not force the whole window wider than the screen (D-110): it wraps instead.
        self.subtitle_label.setWordWrap(True)
        self.title_label.setMinimumWidth(0)
        self.subtitle_label.setMinimumWidth(0)
        header.setMinimumWidth(0)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(self.title_label)
        titles.addWidget(self.subtitle_label)
        self.queue_button = QToolButton()
        self.queue_button.setObjectName("themeButton")
        self.queue_button.clicked.connect(self.open_queue)
        self.history_button = QToolButton()
        self.history_button.setObjectName("themeButton")
        self.history_button.clicked.connect(self.open_history)
        self.theme_button = QToolButton()
        self.theme_button.setObjectName("themeButton")
        self.theme_button.clicked.connect(lambda: self.set_theme(theme.next_mode(self._theme_mode)))
        self.settings_button = QToolButton()
        self.settings_button.setObjectName("themeButton")
        self.settings_button.clicked.connect(self._open_provider_settings)
        header_row.addWidget(self.logo_label)
        header_row.addSpacing(8)
        header_row.addLayout(titles)
        header_row.addStretch(1)
        header_row.addWidget(self.queue_button)
        header_row.addWidget(self.history_button)
        header_row.addWidget(self.theme_button)
        header_row.addWidget(self.settings_button)
        root.addWidget(header)

        # Input card: URL or file, browse, and the main action.
        # URLs and paths are always left-to-right, even in an RTL UI.
        self.input_edit = QLineEdit()
        self.input_edit.setObjectName("bigInput")
        self.input_edit.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.input_edit.setClearButtonEnabled(True)
        self.input_edit.textChanged.connect(self._update_start_label)
        self.input_browse_button = QPushButton()
        self.input_browse_button.clicked.connect(self._browse_input)
        self.start_button = QPushButton()
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._on_start_or_cancel)
        self.input_label = QLabel()
        self.input_label.setObjectName("mutedLabel")
        self.input_group = QGroupBox()
        input_layout = QVBoxLayout(self.input_group)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(self.input_browse_button)
        self.add_to_queue_button = QPushButton()
        self.add_to_queue_button.clicked.connect(self._on_add_current_to_queue)
        input_row.addWidget(self.add_to_queue_button)
        input_row.addWidget(self.start_button)
        input_layout.addWidget(self.input_label)
        input_layout.addLayout(input_row)
        root.addWidget(self.input_group)

        # Options and series side by side.
        self.source_combo = QComboBox()
        self.target_combo = QComboBox()
        self.mode_combo = QComboBox()
        # Video quality for link downloads (D-080): filled with the heights this link really offers after the
        # probe; local files need no choice, so the combo is disabled for them.
        self.quality_combo = QComboBox()
        self.quality_label = QLabel()
        self.source_label, self.target_label, self.mode_label = QLabel(), QLabel(), QLabel()
        self.options_group = QGroupBox()
        self.options_group.setMinimumWidth(0)
        options_grid = QGridLayout(self.options_group)
        options_grid.setHorizontalSpacing(12)
        for column, (label, widget) in enumerate(((self.source_label, self.source_combo),
                                                  (self.target_label, self.target_combo),
                                                  (self.mode_label, self.mode_combo),
                                                  (self.quality_label, self.quality_combo))):
            label.setObjectName("mutedLabel")
            # Let a combo shrink below its longest entry, otherwise four of them forced a 1188 px minimum (D-110).
            widget.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            widget.setMinimumContentsLength(8)
            widget.setMinimumWidth(90)
            options_grid.addWidget(label, 0, column)
            options_grid.addWidget(widget, 1, column)
            options_grid.setColumnStretch(column, 1)
        self._fill_quality_combo([])
        self.quality_combo.currentIndexChanged.connect(self._on_quality_changed)
        self.source_combo.currentIndexChanged.connect(
            lambda: self._save_combo("source_language", self.source_combo))
        self.target_combo.currentIndexChanged.connect(
            lambda: self._save_combo("target_language", self.target_combo))
        self.mode_combo.currentIndexChanged.connect(lambda: self._save_combo("mode", self.mode_combo))

        self.series_edit = QLineEdit()
        self.season_spin = QSpinBox()
        self.season_spin.setRange(0, 99)
        self.episode_spin = QSpinBox()
        self.episode_spin.setRange(0, 9999)
        self.series_name_label, self.season_label, self.episode_label = QLabel(), QLabel(), QLabel()
        # Poster of the chosen series; filled from the lookup cache, so it also shows offline (D-076).
        self.series_poster = QLabel()
        self.series_poster.setObjectName("seriesPoster")
        self.series_keep_check = QCheckBox()
        self.series_group = QGroupBox()
        self.series_group.setMinimumWidth(0)
        series_grid = QGridLayout(self.series_group)
        series_grid.setHorizontalSpacing(12)
        series_grid.addWidget(self.series_poster, 0, 0, 2, 1)
        for column, (label, widget) in enumerate(((self.series_name_label, self.series_edit),
                                                   (self.season_label, self.season_spin),
                                                   (self.episode_label, self.episode_spin)), start=1):
            label.setObjectName("mutedLabel")
            series_grid.addWidget(label, 0, column)
            series_grid.addWidget(widget, 1, column)
        series_grid.addWidget(self.series_keep_check, 2, 0, 1, 4)
        series_grid.setColumnStretch(1, 1)
        self.series_edit.setMinimumWidth(140)
        for spin in (self.season_spin, self.episode_spin):
            spin.setMinimumWidth(84)
            spin.setMaximumWidth(110)
        self.series_picker = SeriesSuggestions(
            self.series_edit, self.series_poster, self._series_repo, self._poster_cache_dir,
            fetch=(self._runtime.series_fetch if self._runtime is not None else None), parent=self)
        self.series_keep_check.setEnabled(self._series_repo is not None)
        self.series_keep_check.toggled.connect(self._on_series_keep_toggled)
        self.series_edit.textEdited.connect(lambda _: self._sync_series_keep())
        self.series_picker.searched.connect(self._on_series_searched)
        # Series/season/episode from the link or the file name, without overwriting what the user typed (D-077).
        # A link is only probed when the window was given a downloader (the app injects the real one), so a plain
        # test window can never reach the network.
        probe_factory = (lambda: self._runtime.downloader) if (self._runtime is not None
                                                              and self._runtime.downloader is not None) else None
        self.link_probe = LinkProbe(self.input_edit, adapter_factory=probe_factory, parent=self)
        self.link_probe.resolved.connect(self._apply_detected_series)
        self.link_probe.qualities.connect(self._on_qualities_probed)
        self.input_edit.textChanged.connect(lambda _: self._update_quality_state())
        middle = QHBoxLayout()
        middle.setSpacing(12)
        middle.addWidget(self.options_group, 3)
        middle.addWidget(self.series_group, 2)
        root.addLayout(middle)

        self.output_edit = QLineEdit()
        self.output_edit.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.output_edit.editingFinished.connect(self._save_output_dir)
        self.output_browse_button = QPushButton()
        self.output_browse_button.clicked.connect(self._browse_output)
        self.output_label = QLabel()
        self.output_label.setObjectName("mutedLabel")
        self.output_group = QGroupBox()
        output_row = QHBoxLayout(self.output_group)
        output_row.addWidget(self.output_label)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.output_browse_button)
        root.addWidget(self.output_group)

        # Progress and result actions.
        self.open_output_button = QPushButton()
        self.open_output_button.clicked.connect(self._open_output)
        self.review_button = QPushButton()
        self.review_button.setEnabled(False)
        self.review_button.clicked.connect(lambda: self.open_review(self._last_output_dir))
        self.burn_button = QPushButton()
        self.burn_button.setEnabled(False)
        self.burn_button.clicked.connect(lambda: self.burn_video(self._last_output_dir))
        self._burn_runner = None
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.stage_label = QLabel()
        self.stage_label.setObjectName("mutedLabel")
        progress_card = QFrame()
        progress_card.setObjectName("card")
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(14, 12, 14, 12)
        progress_top = QHBoxLayout()
        progress_top.addWidget(self.stage_label, 1)
        progress_top.addWidget(self.open_output_button)
        progress_top.addWidget(self.review_button)
        progress_top.addWidget(self.burn_button)
        progress_layout.addLayout(progress_top)
        progress_layout.addWidget(self.progress_bar)
        root.addWidget(progress_card)

        # Batch queue dialog and components (opened via header button or menu)
        self.queue_dialog = BatchQueueDialog(self, translator=self._tr)
        self.queue_table = self.queue_dialog.queue_table
        self.queue_add_files_button = self.queue_dialog.add_files_button
        self.queue_add_folder_button = self.queue_dialog.add_folder_button
        self.queue_add_playlist_button = self.queue_dialog.add_playlist_button
        self.queue_clear_button = self.queue_dialog.clear_button
        self.queue_eta_label = self.queue_dialog.eta_label
        self.queue_group = self.queue_dialog

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_group = QGroupBox()
        log_layout = QVBoxLayout(self.log_group)
        log_layout.addWidget(self.log_view)
        root.addWidget(self.log_group, 1)

        self.setCentralWidget(central)
        self.elapsed_label = QLabel()
        self.statusBar().addPermanentWidget(self.elapsed_label)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

    def _build_menus(self) -> None:
        bar = self.menuBar()
        self.file_menu = bar.addMenu("")
        self.queue_action = QAction(self)
        self.queue_action.triggered.connect(self.open_queue)
        self.file_menu.addAction(self.queue_action)
        self.history_action = QAction(self)
        self.history_action.triggered.connect(self.open_history)
        self.file_menu.addAction(self.history_action)
        self.review_action = QAction(self)
        self.review_action.triggered.connect(self._choose_review_folder)
        self.file_menu.addAction(self.review_action)
        self.exit_action = QAction(self)
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        self.settings_menu = bar.addMenu("")
        self.provider_action = QAction(self)
        self.provider_action.triggered.connect(self._open_provider_settings)
        self.settings_menu.addAction(self.provider_action)
        self.theme_menu = self.settings_menu.addMenu("")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        self._theme_actions: dict[str, QAction] = {}
        for mode in theme.THEMES:
            action = QAction(self, checkable=True)
            action.setData(mode)
            theme_group.addAction(action)
            self.theme_menu.addAction(action)
            self._theme_actions[mode] = action
        theme_group.triggered.connect(lambda action: self.set_theme(action.data()))
        self.gpu_action = QAction(self)
        self.gpu_action.triggered.connect(lambda: self.download_gpu_libraries(ask=False))
        self.settings_menu.addAction(self.gpu_action)
        self.language_menu = self.settings_menu.addMenu("")
        group = QActionGroup(self)
        group.setExclusive(True)
        for code in self._tr.available_languages():
            # Each language is listed by its own native name so it can be found from any UI language.
            action = QAction(self._tr.native_name(code), self, checkable=True)
            action.setData(code)
            group.addAction(action)
            self.language_menu.addAction(action)
            self._language_actions[code] = action
        group.triggered.connect(lambda action: self.set_ui_language(action.data()))

        self.help_menu = bar.addMenu("")
        self.about_action = QAction(self)
        self.about_action.triggered.connect(self._show_about)
        self.help_menu.addAction(self.about_action)
        self.update_action = QAction(self)
        self.update_action.triggered.connect(lambda: self.check_for_updates(manual=True))
        self.help_menu.addAction(self.update_action)

    # -- localization ----------------------------------------------------------------------

    def retranslate_ui(self) -> None:
        t = self._tr.t
        self.setWindowTitle(t("app.title"))
        self.file_menu.setTitle(t("menu.file"))
        self.queue_action.setText(t("action.queue"))
        self.history_action.setText(t("action.history"))
        self.exit_action.setText(t("action.exit"))
        self.settings_menu.setTitle(t("menu.settings"))
        self.language_menu.setTitle(t("menu.language"))
        self.theme_menu.setTitle(t("menu.theme"))
        self.gpu_action.setText(t("action.gpu_download"))
        for mode, action in self._theme_actions.items():
            action.setText(t(f"theme.{mode}"))
            action.setChecked(mode == self._theme_mode)
        self.title_label.setText(t("app.title"))
        self.subtitle_label.setText(t("app.tagline"))
        self._update_theme_button()
        self.queue_button.setText("\u2261  " + t("button.queue"))
        self.history_button.setText("\u23f1  " + t("button.history"))
        self.settings_button.setText("\u2699  " + t("button.settings"))
        self.provider_action.setText(t("action.provider_settings"))
        self.series_group.setTitle(t("group.series"))
        self.series_name_label.setText(t("label.series_name"))
        self.season_label.setText(t("label.season"))
        self.episode_label.setText(t("label.episode"))
        self.season_spin.setSpecialValueText(t("series.auto"))
        self.episode_spin.setSpecialValueText(t("series.auto"))
        self.series_keep_check.setText(t("label.series_keep"))
        self.series_keep_check.setToolTip(t("series.keep_tip"))
        self.series_poster.setToolTip(t("series.poster_tip"))
        self.help_menu.setTitle(t("menu.help"))
        self.about_action.setText(t("action.about"))
        self.update_action.setText(t("action.check_updates"))
        self.input_group.setTitle(t("group.input"))
        self.input_label.setText(t("label.input"))
        self.input_edit.setPlaceholderText(t("input.placeholder"))
        self.input_browse_button.setText(t("button.browse"))
        self.options_group.setTitle(t("group.options"))
        self.source_label.setText(t("label.source_language"))
        self.target_label.setText(t("label.target_language"))
        self.mode_label.setText(t("label.mode"))
        self.quality_label.setText(t("label.video_quality"))
        self.quality_combo.setItemText(0, t("quality.best"))
        self._update_quality_state()
        self.output_group.setTitle(t("group.output"))
        self.output_label.setText(t("label.output_dir"))
        self.output_edit.setPlaceholderText(t("output.placeholder", path=str(default_output_dir())))
        self.output_browse_button.setText(t("button.browse"))
        self.start_button.setText(t("button.cancel") if self.is_running() else self._start_label())
        self.open_output_button.setText(t("button.open_output"))
        self.review_button.setText(t("button.review"))
        self.burn_button.setText(t("button.burn"))
        self.review_action.setText(t("action.open_review"))
        self.log_group.setTitle(t("group.log"))
        self.add_to_queue_button.setText(t("queue.add_to_queue"))
        self.queue_dialog.retranslate_ui()
        self._update_queue_eta()

        self._populate_language_combo(self.source_combo, "source_language", include_auto=True)
        self._populate_language_combo(self.target_combo, "target_language", include_auto=False)
        self._populate_mode_combo()
        action = self._language_actions.get(self._tr.language)
        if action is not None:
            action.setChecked(True)

    def _start_label(self) -> str:
        return self._tr.t("button.start_url" if is_url(self.input_edit.text().strip()) else "button.start")

    @Slot()
    def _update_start_label(self) -> None:
        if not self.is_running():
            self.start_button.setText(self._start_label())

    def _language_label(self, code: str) -> str:
        name = self._tr.t(f"lang.{code}")
        native = languages.native_name(code)
        return name if native == name else f"{name} ({native})"

    def _populate_language_combo(self, combo: QComboBox, key: str, include_auto: bool) -> None:
        selected = combo.currentData() or self._settings.get(key)
        combo.blockSignals(True)
        combo.clear()
        if include_auto:
            combo.addItem(self._tr.t("language.auto_detect"), languages.AUTO_DETECT)
        for code in languages.SUBTITLE_LANGUAGES:
            combo.addItem(self._language_label(code), code)
        index = combo.findData(selected)
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

    def _populate_mode_combo(self) -> None:
        selected = self.mode_combo.currentData() or self._settings.get("mode")
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        for mode in Mode:
            self.mode_combo.addItem(self._tr.t(f"mode.{mode.value}"), mode.value)
            self.mode_combo.setItemData(
                self.mode_combo.count() - 1, self._tr.t(f"mode.{mode.value}.tooltip"),
                Qt.ItemDataRole.ToolTipRole)
        self.mode_combo.setCurrentIndex(max(self.mode_combo.findData(selected), 0))
        self.mode_combo.blockSignals(False)

    def set_ui_language(self, code: str) -> None:
        self._tr.set_language(code)
        self._settings.set("ui_language", code)

    @Slot(str)
    def _on_language_changed(self, code: str) -> None:
        self.retranslate_ui()
        self._report(self._tr.t("status.language_changed", language=self._tr.native_name(code)))

    # -- actions ---------------------------------------------------------------------------

    def _save_combo(self, key: str, combo: QComboBox) -> None:
        value = combo.currentData()
        if value is not None:
            self._settings.set(key, value)

    # -- video quality (D-080) -----------------------------------------------------------------

    def _fill_quality_combo(self, heights: list) -> None:
        """Show "Best available" plus the heights this link really offers, keeping the user's choice if it is
        still offered. Nothing is hidden: a link that only offers low qualities says so."""
        self._offered_heights = [int(h) for h in heights]
        wanted = self.quality_combo.currentData() or self._settings.get("video_quality") or "best"
        options = ["best"] + [str(h) for h in self._offered_heights]
        if not self._offered_heights and wanted != "best":
            options.append(str(wanted))          # before the first probe, show the saved choice
        if self._offered_heights and wanted != "best" and wanted not in options:
            # The link does not offer the saved height any more: say so instead of downloading silently.
            self._report(self._tr.t("quality.not_offered", height=wanted, best=self._offered_heights[0]),
                         level=logging.WARNING)
            wanted = "best"
        self.quality_combo.blockSignals(True)
        self.quality_combo.clear()
        for value in options:
            text = self._tr.t("quality.best") if value == "best" else f"{value}p"
            self.quality_combo.addItem(text, value)
        self.quality_combo.setCurrentIndex(max(self.quality_combo.findData(wanted), 0))
        self.quality_combo.blockSignals(False)
        self._update_quality_state()
        best = self._offered_heights[0] if self._offered_heights else None
        if best is not None and best < 720:
            self._report(self._tr.t("quality.low_offered", height=best), level=logging.WARNING)

    def _update_quality_state(self) -> None:
        """The choice only exists for link downloads; a local file is used as it is."""
        link = is_url(self.input_edit.text().strip())
        self.quality_combo.setEnabled(link)
        self.quality_combo.setToolTip(self._tr.t("quality.tip") if link else self._tr.t("quality.local_tip"))

    @Slot()
    def _on_quality_changed(self) -> None:
        value = self.quality_combo.currentData()
        if value is None:
            return
        self._settings.set("video_quality", value)
        if value != "best" and self._offered_heights and int(value) > self._offered_heights[0]:
            self._report(self._tr.t("quality.not_offered", height=value,
                                    best=self._offered_heights[0]), level=logging.WARNING)

    @Slot(list)
    def _on_qualities_probed(self, heights: list) -> None:
        self._fill_quality_combo(heights)

    @Slot()
    def _save_output_dir(self) -> None:
        self._settings.set("output_dir", self.output_edit.text().strip())

    @Slot()
    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, self._tr.t("dialog.select_video"), "",
            self._tr.t("dialog.video_filter", patterns=VIDEO_PATTERNS))
        if path:
            self.input_edit.setText(path)

    @Slot()
    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, self._tr.t("dialog.select_output"), self.output_edit.text())
        if path:
            self.output_edit.setText(path)
            self._save_output_dir()

    def validate_inputs(self) -> str | None:
        """Return an error message, or None when the inputs are acceptable."""
        text = self.input_edit.text().strip()
        if not text:
            return self._tr.t("error.input_required")
        if not is_url(text) and not Path(text).is_file():
            return self._tr.t("error.input_not_found", path=text)
        if self.source_combo.currentData() == self.target_combo.currentData():
            return self._tr.t("error.same_language")
        return None

    # -- job control -----------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._runner is not None and self._runner.isRunning()

    @Slot()
    def _on_start_or_cancel(self) -> None:
        if self.is_running():
            self.cancel_queue()
        else:
            self.start_queue()

    def _current_extras(self) -> dict:
        return {
            key: self._settings.get(key)
            for key in ("fetch_platform_subtitles", "opensubtitles_api_key", "subdl_api_key",
                        "llm_refine", "llm_correct_only", "llm_review", "translation_engine",
                        "local_model", "audio_enhance", "diarization", "snap_to_shots", "name_normalization",
                        "cookies_source", "cookies_browser", "cookies_file", "force_ipv4", "video_quality",
                        "keep_job_cache")
        }

    @Slot()
    def _on_series_keep_toggled(self, keep: bool) -> None:
        """Keep (or stop keeping) the series named in the field, together with its poster (D-076)."""
        if self._series_repo is None:
            return
        name = self.series_edit.text().strip()
        if not name:
            return
        if keep:
            candidate = self.series_picker.current()
            self._series_repo.save(
                name,
                source_id=(candidate.source_id if candidate else "") or None,
                image_url=candidate.image_url if candidate else None,
                image_path=str(candidate.image_path) if candidate and candidate.image_path else None,
            )
            self._report(self._tr.t("series.kept", name=name))
        else:
            found = self._series_repo.find(name)
            if found:
                self._series_repo.forget(found["id"], self._poster_cache_dir)
        self.series_picker.show_saved_suggestions()

    @Slot()
    def _sync_series_keep(self) -> None:
        """The checkbox mirrors whether the typed name is already kept (without saving anything)."""
        if self._series_repo is None:
            return
        found = self._series_repo.find(self.series_edit.text().strip())
        self.series_keep_check.blockSignals(True)
        self.series_keep_check.setChecked(bool(found))
        self.series_keep_check.blockSignals(False)

    @Slot(str, int)
    def _on_series_searched(self, query: str, found: int) -> None:
        if found == 0:
            self._report(self._tr.t("series.not_found", query=query), level=logging.INFO)

    @Slot(object, str)
    def _apply_detected_series(self, info, origin: str) -> None:
        """Fill the series fields the user left empty; a typed value always wins (D-077)."""
        if info is None or (not info.series_name and info.season is None and info.episode is None):
            return
        detected = []
        if info.series_name and not self.series_edit.text().strip():
            self.series_edit.setText(info.series_name)
            detected.append(info.series_name)
        if info.season is not None and self.season_spin.value() == 0:
            self.season_spin.setValue(info.season)
            detected.append(f"S{info.season}")
        if info.episode is not None and self.episode_spin.value() == 0:
            self.episode_spin.setValue(info.episode)
            detected.append(f"E{info.episode}")
        if not detected:
            return
        self._sync_series_keep()
        self.series_picker.refresh_after_external_change()
        self._report(self._tr.t("series.detected", values=" ".join(detected),
                                origin=self._tr.t(f"series.detected_from_{origin}")))

    def _current_series_override(self, title: str | None = None) -> dict:
        return {
            "series_name": self.series_edit.text().strip() or None,
            "season": self.season_spin.value() or None,
            "episode": self.episode_spin.value() or None,
            "episode_title": title,
            "year": None,
            "source": "user",
        }

    def _open_dir(self, directory: str | Path | None) -> bool:
        """Open a folder in the file manager. Returns False (and tells the user) when it could not be opened."""
        if not directory:
            return False
        path = Path(directory)
        if not path.is_dir():
            self._report(self._tr.t("error.output_missing", path=str(path)), level=logging.WARNING)
            return False
        if open_folder(path, lambda p: QDesktopServices.openUrl(QUrl.fromLocalFile(p))):
            return True
        QGuiApplication.clipboard().setText(str(path))     # last resort: the path can be pasted into Explorer
        self._report(self._tr.t("error.open_folder_failed", path=str(path)), level=logging.WARNING)
        return False

    @Slot()
    def _browse_queue_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, self._tr.t("queue.dialog_select_files"), "",
            self._tr.t("dialog.video_filter", patterns=VIDEO_PATTERNS))
        if paths:
            self.add_files_to_queue(paths)

    @Slot()
    def _browse_queue_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self._tr.t("queue.dialog_select_folder"))
        if folder:
            self.add_folder_to_queue(folder)

    @Slot()
    def _browse_queue_playlist(self) -> None:
        url, ok = QInputDialog.getText(
            self, self._tr.t("queue.dialog_playlist_title"),
            self._tr.t("queue.dialog_playlist_prompt"))
        if ok and url.strip():
            self.add_playlist_to_queue(url.strip())

    @Slot()
    def _on_add_current_to_queue(self) -> None:
        text = self.input_edit.text().strip()
        if not text:
            self._report(self._tr.t("error.input_required"), level=logging.WARNING)
            return
        if not is_url(text) and not Path(text).is_file():
            self._report(self._tr.t("error.input_not_found", path=text), level=logging.WARNING)
            return
        if self.source_combo.currentData() == self.target_combo.currentData():
            self._report(self._tr.t("error.same_language"), level=logging.WARNING)
            return

        if is_url(text):
            if "list=" in text and "watch?v=" not in text:
                self.add_playlist_to_queue(text)
            else:
                if self._runtime:
                    db = Database(self._runtime.db_path)
                    try:
                        repo = JobsRepo(db)
                        jid = repo.enqueue(
                            input_type="url",
                            input_value=text,
                            source_language=self.source_combo.currentData(),
                            target_language=self.target_combo.currentData(),
                            mode=self.mode_combo.currentData(),
                            output_dir=self.output_edit.text().strip() or None,
                            series=self._current_series_override(),
                            extras=self._current_extras(),
                            title=text,
                        )
                        self._add_or_update_table_row(repo.get(jid))
                        self._report(self._tr.t("queue.items_added", count=1))
                    finally:
                        db.close()
        else:
            self.add_files_to_queue([Path(text)])
        self.input_edit.clear()

    def add_files_to_queue(self, paths: list[Path | str]) -> list[int]:
        if not paths or self._runtime is None:
            return []
        db = Database(self._runtime.db_path)
        job_ids = []
        try:
            repo = JobsRepo(db)
            for p in paths:
                path = Path(p)
                if not path.is_file():
                    continue
                title = path.stem
                series = self._current_series_override(title)
                jid = repo.enqueue(
                    input_type="file",
                    input_value=str(path.resolve()),
                    source_language=self.source_combo.currentData(),
                    target_language=self.target_combo.currentData(),
                    mode=self.mode_combo.currentData(),
                    output_dir=self.output_edit.text().strip() or None,
                    series=series,
                    extras=self._current_extras(),
                    title=title,
                )
                job_ids.append(jid)
                self._add_or_update_table_row(repo.get(jid))
        finally:
            db.close()
        if job_ids:
            self._report(self._tr.t("queue.items_added", count=len(job_ids)))
            self._update_queue_eta()
        return job_ids

    def add_folder_to_queue(self, folder: Path | str) -> list[int]:
        fpath = Path(folder)
        if not fpath.is_dir():
            return []
        extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".flv", ".wmv", ".mp3", ".wav", ".m4a", ".flac"}
        files = [p for p in fpath.iterdir() if p.is_file() and p.suffix.lower() in extensions]
        files.sort(key=lambda p: p.name.lower())
        return self.add_files_to_queue(files)

    def add_playlist_to_queue(self, url: str) -> list[int]:
        if not url or self._runtime is None:
            return []
        downloader = getattr(self._runtime, "downloader", None) or YtDlpAdapter()
        try:
            entries = downloader.extract_playlist(url)
        except Exception as exc:
            log.exception("Playlist extraction failed")
            self._report(self._tr.t("queue.playlist_error", error=str(exc)), level=logging.WARNING)
            return []
        if not entries:
            return []
        db = Database(self._runtime.db_path)
        job_ids = []
        try:
            repo = JobsRepo(db)
            for entry in entries:
                e_url = entry.get("url")
                if not e_url:
                    continue
                title = entry.get("title") or e_url
                series = self._current_series_override(title)
                jid = repo.enqueue(
                    input_type="url",
                    input_value=e_url,
                    source_language=self.source_combo.currentData(),
                    target_language=self.target_combo.currentData(),
                    mode=self.mode_combo.currentData(),
                    output_dir=self.output_edit.text().strip() or None,
                    series=series,
                    extras=self._current_extras(),
                    title=title,
                )
                job_ids.append(jid)
                self._add_or_update_table_row(repo.get(jid))
        finally:
            db.close()
        if job_ids:
            self._report(self._tr.t("queue.items_added", count=len(job_ids)))
            self._update_queue_eta()
        return job_ids

    def clear_completed_queue(self) -> None:
        if self._runtime is None:
            return
        db = Database(self._runtime.db_path)
        try:
            repo = JobsRepo(db)
            repo.clear_completed_queue()
        finally:
            db.close()
        self._load_queue_from_db()

    def _load_queue_from_db(self) -> None:
        if self._runtime is None:
            return
        db = Database(self._runtime.db_path)
        try:
            repo = JobsRepo(db)
            repo.mark_interrupted()
            queue = repo.get_queue()
            self.queue_table.setRowCount(0)
            for job in queue:
                self._add_or_update_table_row(job)
        finally:
            db.close()
        if hasattr(self, "queue_dialog"):
            self.queue_dialog._update_count_label()
        self._update_queue_eta()

    # -- ETA (Phase 6.6) --------------------------------------------------------------------

    def _refresh_timings(self) -> None:
        """Reload this machine's stage timings from the finished jobs in the database."""
        if self._runtime is None:
            return
        db = Database(self._runtime.db_path)
        try:
            self._timings = eta.timings_from_stats(JobsRepo(db).recent_stats(eta.MAX_HISTORY_JOBS))
        except Exception:                       # an estimate is never worth breaking the UI for
            log.exception("Could not read the stage timings")
            self._timings = eta.EMPTY_TIMINGS
        finally:
            db.close()

    def _pending_queue_jobs(self) -> list[dict]:
        if self._runtime is None:
            return []
        db = Database(self._runtime.db_path)
        try:
            queue = JobsRepo(db).get_queue()
        finally:
            db.close()
        return [job for job in queue if job.get("status") in ("pending", "paused")]

    def _probe_duration(self, path: str | None) -> float | None:
        """Media length of a queued local file, read once and remembered for this window."""
        if not path:
            return None
        if path not in self._duration_cache:
            target = Path(path)
            self._duration_cache[path] = eta.media_duration(target) if target.is_file() else None
        return self._duration_cache[path]

    def _job_remaining_seconds(self) -> float | None:
        """Seconds until the running job is done: the stage model, or the elapsed share as a fallback."""
        if not self.is_running():
            return None
        remaining = eta.job_remaining(self._timings, self._current_stage,
                                      self._job_media_duration, self._stage_fraction)
        if remaining is not None:
            return remaining
        elapsed = time.monotonic() - self._job_started
        if self._last_overall > 0.02:
            return max(0.0, elapsed / self._last_overall - elapsed)
        return None

    def _queue_eta_seconds(self, pending: list[dict]) -> float | None:
        """Seconds until the whole queue is done: the running job plus every pending job."""
        total = 0.0
        if self.is_running():
            remaining = self._job_remaining_seconds()
            if remaining is None:
                return None
            total += remaining
        for job in pending:
            duration = (self._probe_duration(job.get("input_value"))
                        if job.get("input_type") == "file" else None)
            seconds = eta.pending_job_seconds(self._timings, duration)
            if seconds is None:
                return None
            total += seconds
        return total

    def _update_queue_eta(self) -> None:
        """Show the total queue ETA in the queue card, or clear it when there is nothing to wait for."""
        pending = self._pending_queue_jobs()
        if not pending and not self.is_running():
            self.queue_eta_label.clear()
            return
        seconds = self._queue_eta_seconds(pending)
        self.queue_eta_label.setText(self._tr.t(
            "queue.total_eta", eta=eta.format_eta(seconds) if seconds is not None else eta.UNKNOWN))

    def _add_or_update_table_row(self, job: dict, eta: str = "") -> None:
        job_id = job["id"]
        target_row = -1
        for r in range(self.queue_table.rowCount()):
            item = self.queue_table.item(r, 1)
            if item and item.data(Qt.ItemDataRole.UserRole) == job_id:
                target_row = r
                break
        if target_row == -1:
            target_row = self.queue_table.rowCount()
            self.queue_table.insertRow(target_row)

        order_val = job.get("queue_order") or (target_row + 1)
        order_item = QTableWidgetItem(str(order_val))
        order_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.queue_table.setItem(target_row, 0, order_item)

        display_name = ""
        if job.get("queue_config"):
            try:
                cfg = json.loads(job["queue_config"])
                display_name = cfg.get("title", "")
            except Exception:
                pass
        if not display_name:
            val = job.get("input_value", "")
            display_name = Path(val).name if job.get("input_type") == "file" else val
        name_item = QTableWidgetItem(display_name)
        name_item.setData(Qt.ItemDataRole.UserRole, job_id)
        name_item.setToolTip(job.get("input_value", ""))
        self.queue_table.setItem(target_row, 1, name_item)

        st = job.get("status", "pending")
        st_text = self._tr.t(f"queue.status_{st}", default=st.capitalize())
        st_item = QTableWidgetItem(st_text)
        st_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.queue_table.setItem(target_row, 2, st_item)

        if not eta:
            eta = "-" if st in ("pending", "paused", "cancelled", "failed") else ("" if st == "completed" else "--:--")
        eta_item = QTableWidgetItem(eta)
        eta_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.queue_table.setItem(target_row, 3, eta_item)

        open_btn = QPushButton(self._tr.t("queue.open_output"))
        out_dir = job.get("output_dir")
        open_btn.setEnabled(st == "completed" and bool(out_dir))
        if out_dir:
            open_btn.clicked.connect(lambda _, d=out_dir: self._open_dir(d))
        self.queue_table.setCellWidget(target_row, 4, open_btn)
        if hasattr(self, "queue_dialog"):
            self.queue_dialog._update_count_label()

    def start_job(self) -> bool:
        return self.start_queue()

    def start_queue(self) -> bool:
        if self.is_running():
            return False

        next_job = None
        if self._runtime is not None:
            db = Database(self._runtime.db_path)
            try:
                repo = JobsRepo(db)
                next_job = repo.next_pending_in_queue()
            finally:
                db.close()

        if next_job is None:
            error = self.validate_inputs()
            if error:
                self._report(error, level=logging.WARNING)
                return False
            if self._runtime is None:
                self._report(self._tr.t("status.pipeline_unavailable"), level=logging.WARNING)
                return False
            self._on_add_current_to_queue()
            db = Database(self._runtime.db_path)
            try:
                repo = JobsRepo(db)
                next_job = repo.next_pending_in_queue()
            finally:
                db.close()
            if next_job is None:
                self._report(self._tr.t("queue.empty"), level=logging.WARNING)
                return False
        else:
            if self._runtime is None:
                self._report(self._tr.t("status.pipeline_unavailable"), level=logging.WARNING)
                return False

        self._queue_running = True
        self._queue_cancelled = False
        self._refresh_timings()
        return self._run_next_in_queue()

    def cancel_job(self) -> None:
        self.cancel_queue()

    def cancel_queue(self) -> None:
        self._queue_cancelled = True
        self._queue_running = False
        if self.is_running():
            self._runner.request_cancel()
            self.start_button.setEnabled(False)
            self._report(self._tr.t("status.cancelling"))
        else:
            if self._runtime:
                db = Database(self._runtime.db_path)
                try:
                    repo = JobsRepo(db)
                    repo.cancel_queue()
                finally:
                    db.close()
        for r in range(self.queue_table.rowCount()):
            item = self.queue_table.item(r, 2)
            if item and item.text() == self._tr.t("queue.status_pending"):
                item.setText(self._tr.t("queue.status_cancelled"))
                eta_item = self.queue_table.item(r, 3)
                if eta_item:
                    eta_item.setText("-")
        self._update_queue_eta()

    def _run_next_in_queue(self) -> bool:
        if self._queue_cancelled or self._runtime is None:
            self._queue_running = False
            self._set_running_ui(False)
            self._runner = None
            return False

        db = Database(self._runtime.db_path)
        try:
            repo = JobsRepo(db)
            next_job = repo.next_pending_in_queue()
        finally:
            db.close()

        if next_job is None:
            self._queue_running = False
            self._set_running_ui(False)
            self._runner = None
            self.progress_bar.setValue(100)
            return False

        qconfig = {}
        if next_job.get("queue_config"):
            try:
                qconfig = json.loads(next_job["queue_config"])
            except Exception:
                pass
        # Settings saved with the queue entry win; API keys always come from the current settings.
        extras = {**self._current_extras(), **(qconfig.get("extras") or {})}
        extras.update({k: v for k, v in self._current_extras().items() if k.endswith("_api_key")})
        series_dict = qconfig.get("series") or {}
        override = SeriesInfo(
            series_name=series_dict.get("series_name") or next_job.get("series_name"),
            season=series_dict.get("season") or next_job.get("season"),
            episode=series_dict.get("episode") or next_job.get("episode"),
            episode_title=series_dict.get("episode_title") or next_job.get("episode_title"),
            year=series_dict.get("year") or next_job.get("year"),
            source="user",
        )
        input_val = next_job["input_value"]
        url = next_job["input_type"] == "url" or is_url(input_val)
        if url and not js_runtime_available():
            self._report(self._tr.t("status.no_js_runtime"), level=logging.WARNING)

        config = JobConfig(
            input_path=None if url else Path(input_val),
            source_url=input_val if url else None,
            series_override=override,
            source_language=next_job["source_language"],
            target_language=next_job["target_language"],
            mode=Mode(next_job["mode"]),
            output_dir=Path(next_job["output_dir"]) if next_job.get("output_dir") else None,
        )

        self._runner = JobRunner(
            config, self._runtime.db_path, self._runtime.pipeline_factory,
            extras, self, job_id=next_job["id"],
        )
        self._runner.progress.connect(self._on_progress)
        self._runner.succeeded.connect(self._on_succeeded)
        self._runner.failed.connect(self._on_failed)
        self._runner.cancelled.connect(self._on_cancelled)
        self._runner.finished.connect(self._on_runner_finished)
        self._set_running_ui(True)
        # Fresh ETA inputs for the job that starts now (URLs announce their length from the pipeline).
        self._current_stage = ""
        self._stage_fraction = 0.0
        self._last_overall = 0.0
        self._job_media_duration = None if url else self._probe_duration(input_val)
        self._job_started = time.monotonic()
        self._elapsed_timer.start()
        self._update_elapsed()
        next_job["status"] = "running"
        self._add_or_update_table_row(next_job, eta="--:--")
        self._report(self._tr.t("status.job_started", path=input_val))
        self._runner.start()
        return True

    def _set_running_ui(self, running: bool) -> None:
        for widget in (self.input_edit, self.input_browse_button, self.source_combo, self.target_combo,
                       self.mode_combo, self.output_edit, self.output_browse_button, self.series_edit,
                       self.season_spin, self.episode_spin, self.add_to_queue_button,
                       self.queue_add_files_button, self.queue_add_folder_button,
                       self.queue_add_playlist_button, self.queue_clear_button):
            widget.setEnabled(not running)
        # These two have their own rule when idle (a repository must exist; a link is needed for the quality).
        self.series_keep_check.setEnabled(not running and self._series_repo is not None)
        if running:
            self.quality_combo.setEnabled(False)
        else:
            self._update_quality_state()
        self.start_button.setEnabled(True)
        self.start_button.setText(self._tr.t("button.cancel") if running else self._start_label())
        if hasattr(self, "queue_dialog"):
            self.queue_dialog.start_button.setEnabled(not running)
            self.queue_dialog.cancel_button.setEnabled(running)
        if running:
            self.progress_bar.setValue(0)
            self.review_button.setEnabled(False)
            self.burn_button.setEnabled(False)

    @Slot(str, float, float, str)
    def _on_progress(self, stage: str, stage_fraction: float, overall: float, message: str) -> None:
        if message.startswith("media_duration:"):
            # Side channel from the pipeline: the media length, used for ETA maths only.
            try:
                self._job_media_duration = float(message.split(":", 1)[1])
            except ValueError:
                log.warning("Unexpected media duration message: %s", message)
            return
        self._current_stage = stage
        self._stage_fraction = stage_fraction
        self._last_overall = overall
        self.progress_bar.setValue(int(overall * 100))
        if message.startswith("download:"):
            text = self._tr.t("status.downloading", detail=message.split(":", 1)[1])
        elif message == "detecting_language":
            text = self._tr.t("status.detecting_language")
        elif message == "probing":
            text = self._tr.t("status.probing")
        elif message == "enhancing_audio":
            text = self._tr.t(f"status.{message}")
        else:
            stage_name = self._tr.t(f"stage.{stage}")
            percent = int(stage_fraction * 100)
            text = f"{stage_name} {percent}%"
            remaining = eta.stage_remaining(self._timings, stage, self._job_media_duration, stage_fraction)
            if remaining is not None:      # per-stage ETA from this machine's past real-time factors
                text = self._tr.t("status.stage_eta", stage=stage_name, percent=percent,
                                  eta=eta.format_eta(remaining))
        self.stage_label.setText(text)

        # Update ETA in queue table for current running job
        if self._runner and getattr(self._runner, "job_id", None) is not None:
            jid = self._runner.job_id
            remaining = self._job_remaining_seconds()
            eta_str = f"~{eta.format_eta(remaining)}" if remaining is not None else eta.UNKNOWN
            for r in range(self.queue_table.rowCount()):
                it = self.queue_table.item(r, 1)
                if it and it.data(Qt.ItemDataRole.UserRole) == jid:
                    st_it = self.queue_table.item(r, 2)
                    if st_it:
                        st_it.setText(f"{self._tr.t('queue.status_running')} ({int(overall * 100)}%)")
                    eta_it = self.queue_table.item(r, 3)
                    if eta_it:
                        eta_it.setText(eta_str)
                    break

    @Slot(object)
    def _on_succeeded(self, result: JobResult) -> None:
        self._last_output_dir = result.output_dir
        self.progress_bar.setValue(100)
        self._report(self._tr.t("status.completed", path=str(result.output_dir)))
        for path in result.outputs.values():
            self.log_view.appendPlainText(f"  {path}")
        series = {k: v for k, v in result.series.items() if v not in (None, "") and k != "source"}
        if series:
            self._report(self._tr.t("status.series_detected", info=series))
        if any(w.startswith("Translation plan") and "/local/" in w for w in result.warnings):
            self._report(self._tr.t("status.local_model_failed"), level=logging.WARNING)
        refine = result.stats.get("refine", {})
        if refine:
            providers = ", ".join(refine.get("providers", [])) or "-"
            if refine.get("mode") == "correct":
                self._report(self._tr.t("status.correct_summary", checked=refine.get("refined", 0),
                                        total=refine.get("lines", 0), corrected=refine.get("corrected", 0),
                                        providers=providers))
            else:
                self._report(self._tr.t("status.refine_summary", refined=refine.get("refined", 0),
                                        total=refine.get("lines", 0), suggestions=refine.get("suggestions", 0),
                                        providers=providers))
            tokens = refine.get("tokens") or {}
            if tokens:
                inp = sum(v.get("prompt_tokens", 0) for v in tokens.values())
                out = sum(v.get("completion_tokens", 0) for v in tokens.values())
                requests = sum(v.get("requests", 0) for v in tokens.values())
                self._report(self._tr.t("status.ai_tokens", total=f"{inp + out:,}", inp=f"{inp:,}",
                                        out=f"{out:,}", requests=requests))
        sources = result.stats.get("subtitle_sources", [])
        self._report(self._tr.t("status.sources_found", count=len(sources)))
        for src in sources:
            agreement = src.get("audio_agreement")
            score = f" {agreement:.2f}" if agreement is not None else ""
            self.log_view.appendPlainText(
                f"  {src['provider']} {src['kind']} {src['language']}: {src['status']}{score} ({src['cues']} cues)")
        if result.warnings:
            self._report(self._tr.t("status.warnings", count=len(result.warnings)), level=logging.WARNING)
            for warning in result.warnings:
                self.log_view.appendPlainText(f"  ! {warning}")
        self.review_button.setEnabled(True)
        self.burn_button.setEnabled(True)
        if self._settings.get("burn_video"):
            self.burn_video(result.output_dir)

    @Slot(str, str, object)
    def _on_failed(self, technical: str, ui_key: str, ui_args: dict) -> None:
        message = self._tr.t(ui_key, **(ui_args or {}))
        self._report(message, level=logging.ERROR)
        self.log_view.appendPlainText(f"  {technical}")
        if self._runner and getattr(self._runner, "job_id", None) is not None:
            jid = self._runner.job_id
            for r in range(self.queue_table.rowCount()):
                it = self.queue_table.item(r, 1)
                if it and it.data(Qt.ItemDataRole.UserRole) == jid:
                    st_it = self.queue_table.item(r, 2)
                    if st_it:
                        st_it.setText(self._tr.t("queue.status_failed"))
                    eta_it = self.queue_table.item(r, 3)
                    if eta_it:
                        eta_it.setText("-")
                    break
        if not self._queue_running and not self._close_when_done:
            self._show_error(message, technical)

    @Slot()
    def _on_cancelled(self) -> None:
        self._report(self._tr.t("status.cancelled"))

    @Slot()
    def _on_runner_finished(self) -> None:
        sender = self.sender()
        if sender is not None and self._runner is not None and sender is not self._runner:
            return
        self._elapsed_timer.stop()
        self._update_elapsed()
        self._refresh_timings()           # the finished job's timings feed the next job's ETA
        dur = time.monotonic() - self._job_started if self._job_started else 0.0
        self._job_times.append(dur)
        if self._runner and getattr(self._runner, "job_id", None) is not None and self._runtime:
            jid = self._runner.job_id
            db = Database(self._runtime.db_path)
            try:
                repo = JobsRepo(db)
                if self._queue_cancelled:
                    repo.cancel_queue()
                job_row = repo.get(jid)
                if job_row:
                    dur_str = f"{int(dur // 60):02d}:{int(dur % 60):02d}"
                    self._add_or_update_table_row(job_row, eta=dur_str)
            finally:
                db.close()

        if self._queue_running and not self._queue_cancelled:
            if self._run_next_in_queue():
                return

        self._queue_running = False
        self._set_running_ui(False)
        self.stage_label.clear()
        self._runner = None
        if self._close_when_done:
            self.close()

    def _update_elapsed(self) -> None:
        seconds = int(time.monotonic() - self._job_started) if self._job_started else 0
        self.elapsed_label.setText(self._tr.t("status.elapsed", time=time.strftime("%H:%M:%S", time.gmtime(seconds))))
        self._update_queue_eta()          # the running job shrinks every second, so must the queue ETA

    def open_review(self, folder) -> None:
        if not folder:
            return
        from app.core.review import ReviewError
        from app.services.job_manager import RefinerFactory
        from app.ui.review_window import ReviewWindow

        factory = RefinerFactory("translate", False) if self._settings.get("llm_refine") else None
        try:
            window = ReviewWindow(Path(folder), self._tr, factory, self,
                                  asr_factory=self._review_asr_factory())
        except (ReviewError, OSError, ValueError, KeyError) as exc:
            self._report(self._tr.t("error.review_open", error=str(exc)), level=logging.WARNING)
            return
        self._review_windows = [w for w in getattr(self, "_review_windows", []) if w.isVisible()] + [window]
        window.show()

    def _review_asr_factory(self):
        """The real ASR engine for "re-transcribe this span", built from the jobs' own plans; None when
        no local Whisper model is installed, so the review window says so instead of failing."""
        from app.models.model_manager import WHISPER_MODELS, ModelManager
        from app.services.job_manager import AsrEngineFactory

        models_dir = Path(self._settings.get("model_dir") or (default_data_root() / "models"))
        models = ModelManager(models_dir)
        installed = [name for name in WHISPER_MODELS if models.is_installed("whisper", name)]
        if not installed:
            return None
        return AsrEngineFactory(models, installed)

    @Slot()
    def _choose_review_folder(self) -> None:
        start = self.output_edit.text().strip() or str(default_output_dir())
        folder = QFileDialog.getExistingDirectory(self, self._tr.t("dialog.select_episode"), start)
        if folder:
            self.open_review(folder)

    def offer_gpu_download(self) -> None:
        """Once per installation: an NVIDIA GPU without the GPU libraries -> offer the download."""
        if self._settings.get("gpu_offer_shown"):
            return
        from app.services import gpu_runtime
        from app.services.hardware_detection import _nvidia_smi_gpus

        if gpu_runtime.should_offer(_nvidia_smi_gpus()):
            self._settings.set("gpu_offer_shown", True)
            self.download_gpu_libraries(ask=True)

    def download_gpu_libraries(self, ask: bool = True) -> None:
        if self._gpu_runner is not None and self._gpu_runner.isRunning():
            return
        if ask:
            answer = QMessageBox.question(self, self._tr.t("app.title"), self._tr.t("gpu.offer"))
            if answer != QMessageBox.StandardButton.Yes:
                self._report(self._tr.t("gpu.later"))
                return
        from app.ui.gpu_download import GpuDownloadRunner

        runner = GpuDownloadRunner(self)
        runner.progress.connect(lambda f, text: (self.progress_bar.setValue(int(f * 100)),
                                                 self.stage_label.setText(self._tr.t("gpu.downloading", detail=text))))
        runner.succeeded.connect(lambda: self._report(self._tr.t("gpu.done")))
        runner.failed.connect(lambda error: self._report(self._tr.t("gpu.failed", error=error), level=logging.WARNING))
        self._gpu_runner = runner
        self._report(self._tr.t("gpu.started"))
        runner.start()

    def set_theme(self, mode: str) -> None:
        """Switch between following Windows, light and dark (saved in the settings)."""
        self._settings.set("theme", mode)
        self._theme_mode = mode
        app = QApplication.instance()
        if app is not None:
            theme.apply(app, mode)
        for key, action in self._theme_actions.items():
            action.setChecked(key == mode)
        self._update_theme_button()

    def _update_theme_button(self) -> None:
        mode = self._theme_mode
        symbol = {"system": "\u25d0", "light": "\u2600", "dark": "\u263e"}.get(mode, "\u25d0")
        self.theme_button.setText(f"{symbol}  {self._tr.t(f'theme.{mode}')}")
        self.theme_button.setToolTip(self._tr.t("theme.tooltip"))

    @Slot()
    def _open_provider_settings(self) -> None:
        ProviderSettingsDialog(self._tr, self._settings, self,
                               series_repo=self._series_repo,
                               poster_cache_dir=self._poster_cache_dir).open()

    def burn_video(self, episode_dir) -> None:
        """Save a copy of the episode video with the subtitles drawn into the picture."""
        if not episode_dir or (self._burn_runner is not None and self._burn_runner.isRunning()):
            return
        self.burn_button.setEnabled(False)
        self.stage_label.setText(self._tr.t("stage.burn"))
        self.progress_bar.setValue(0)
        runner = BurnRunner(None, None, Path(episode_dir), self)
        runner.progress.connect(lambda f: (self.progress_bar.setValue(int(f * 100)),
                                           self.stage_label.setText(f"{self._tr.t('stage.burn')} {int(f * 100)}%")))
        runner.succeeded.connect(lambda path: self._report(self._tr.t("status.burn_done", path=path)))
        runner.failed.connect(lambda error: self._report(self._tr.t("status.burn_failed", error=self._tr.t(error)),
                                                         level=logging.WARNING))
        runner.finished.connect(lambda: self.burn_button.setEnabled(not self.is_running()))
        self._burn_runner = runner
        runner.start()

    @Slot()
    def _open_output(self) -> None:
        """Open the output folder the user configured, not the folder of one episode (D-108).

        The episode folder (`_last_output_dir`) is one level deeper and already has its own buttons: the queue row's
        "Open Folder" and the completion message both point at it.
        """
        root = Path(self.output_edit.text().strip() or default_output_dir())
        try:
            root.mkdir(parents=True, exist_ok=True)         # nothing has been written yet on a fresh install
        except OSError as exc:
            log.warning("Cannot create the output folder %s: %s", root, exc)
        self._open_dir(root)

    def open_queue(self) -> BatchQueueDialog:
        self.queue_dialog.show()
        self.queue_dialog.raise_()
        self.queue_dialog.activateWindow()
        return self.queue_dialog

    def open_history(self):
        from app.ui.history_window import JobHistoryDialog
        dialog = JobHistoryDialog(self, self._tr, self._runtime.db_path if self._runtime else None, self)
        self._history_dialog = dialog
        dialog.show()
        return dialog

    def resume_job(self, job_id: int) -> bool:
        if self._runtime is None:
            self._report(self._tr.t("status.pipeline_unavailable"), level=logging.WARNING)
            return False
        db = Database(self._runtime.db_path)
        try:
            repo = JobsRepo(db)
            job = repo.prepare_resume(job_id, self._current_extras())
            if not job:
                return False
            self._add_or_update_table_row(job)
        finally:
            db.close()

        val = job.get("input_value", "")
        display_name = job.get("episode_title") or (Path(val).name if job.get("input_type") == "file" else val)
        self._report(self._tr.t("history.resumed", id=job_id, title=display_name))
        self._update_queue_eta()

        if not self.is_running():
            self.start_queue()
        return True

    def _show_error(self, message: str, details: str) -> None:
        box = QMessageBox(QMessageBox.Icon.Critical, self._tr.t("dialog.error_title"), message,
                          QMessageBox.StandardButton.Ok, self)
        box.setDetailedText(details)
        box.open()  # Window-modal but non-blocking.

    def closeEvent(self, event: QCloseEvent) -> None:
        # Review windows first: each asks about unsaved edits and may refuse to close (D-047).
        for window in getattr(self, "_review_windows", []):
            if window.isVisible() and not window.close():
                event.ignore()
                return
        # Lookup threads next: a running QThread must never be destroyed with its owner (D-106).
        for worker in (getattr(self, "series_picker", None), getattr(self, "link_probe", None)):
            if worker is not None:
                worker.stop()
        for runner in (self._burn_runner, self._gpu_runner):
            if runner is not None and runner.isRunning():
                runner.cancel()
                runner.wait(10000)
        if self._update_runner is not None and self._update_runner.isRunning():
            self._update_runner.wait(12000)          # the request has a 10 s timeout
        if self.is_running():
            # Cancel first; completed stages and segments stay cached for the next run.
            self._close_when_done = True
            self.cancel_job()
            event.ignore()
            return
        self._save_geometry()                # remember size and position for the next start (D-109)
        if self._series_db is not None:
            self._series_db.close()          # the series picker's own connection (D-076)
            self._series_db = None
        super().closeEvent(event)

    # -- updates (D-118) --------------------------------------------------------------------

    def check_for_updates(self, manual: bool = False) -> None:
        """Ask GitHub for a newer release in the background. Automatic checks stay silent unless one is found."""
        from app.ui.update_dialog import UpdateCheckRunner

        if self._update_runner is not None and self._update_runner.isRunning():
            return
        if manual:
            self.statusBar().showMessage(self._tr.t("update.checking"), 5000)
        runner = UpdateCheckRunner(parent=self)
        runner.found.connect(lambda info: self._on_update_checked(info, manual))
        runner.failed.connect(lambda error: self._on_update_check_failed(error, manual))
        self._update_runner = runner
        runner.start()

    def _on_update_checked(self, info, manual: bool) -> None:
        from app.services import updater
        from app.ui.update_dialog import UpdateDialog

        t = self._tr.t
        if not updater.is_newer(info.version, __version__):
            if manual:
                QMessageBox.information(self, t("update.title_check"), t("update.up_to_date", current=__version__))
            return
        if not manual and self._settings.get("update_skipped") == info.version:
            return
        dialog = UpdateDialog(self._tr, info, __version__, allow_skip=not manual, busy=self.is_running(), parent=self)
        dialog.exec()
        if dialog.skipped:
            self._settings.set("update_skipped", info.version)
        elif dialog.launched:
            self.close()              # the installer is waiting for this program to exit

    def _on_update_check_failed(self, error: str, manual: bool) -> None:
        if manual:
            QMessageBox.warning(self, self._tr.t("update.title_check"),
                                self._tr.t("update.check_failed", error=error))

    @Slot()
    def _show_about(self) -> None:
        QMessageBox.about(self, self._tr.t("about.title"), self._tr.t("about.text", version=__version__))

    def _report(self, message: str, level: int = logging.INFO) -> None:
        """Show a message in the status bar and the activity log without blocking."""
        self.statusBar().showMessage(message, 10000)
        self.log_view.appendPlainText(message)
        log.log(level, "UI: %s", message)
