"""Application look: Fusion style, light and dark palettes, one stylesheet (D-042).

Theme setting: "system" follows the Windows light/dark choice (and changes with it while the app runs), "light"
and "dark" force a theme. Widgets mark special roles with object names: "primaryButton" (accent),
"headerTitle"/"headerSubtitle" (window header), "card" frames, "mutedLabel".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

from app.utils.paths import resources_dir

THEMES = ("system", "light", "dark")


@dataclass(frozen=True)
class Tokens:
    window: str
    card: str
    card_border: str
    field: str
    field_border: str
    text: str
    muted: str
    accent: str
    accent_hover: str
    accent_text: str
    button: str
    button_hover: str
    disabled_text: str
    selection: str
    log: str


LIGHT = Tokens(window="#f3f4f8", card="#ffffff", card_border="#e2e5ee", field="#ffffff", field_border="#cfd4e0",
               text="#1d2230", muted="#667085", accent="#4f46e5", accent_hover="#4338ca", accent_text="#ffffff",
               button="#eef0f6", button_hover="#e2e6f1", disabled_text="#a0a6b5", selection="#c7c4fb",
               log="#f8f9fc")
DARK = Tokens(window="#14161c", card="#1d2028", card_border="#2b2f3a", field="#252935", field_border="#373c4a",
              text="#e7e9ef", muted="#9aa1b2", accent="#7c74ff", accent_hover="#9089ff", accent_text="#ffffff",
              button="#2a2e3a", button_hover="#343947", disabled_text="#5d6375", selection="#4b46a8",
              log="#171920")


def system_is_dark() -> bool:
    hints = QGuiApplication.styleHints()
    scheme = getattr(hints, "colorScheme", None)
    if scheme is None:
        return False
    return scheme() == Qt.ColorScheme.Dark


def resolve(mode: str) -> str:
    """"light" or "dark" for a theme setting."""
    if mode == "dark" or (mode == "system" and system_is_dark()):
        return "dark"
    return "light"


def next_mode(mode: str) -> str:
    return THEMES[(THEMES.index(mode) + 1) % len(THEMES)] if mode in THEMES else "system"


def _palette(t: Tokens) -> QPalette:
    p = QPalette()
    for role, color in ((QPalette.ColorRole.Window, t.window), (QPalette.ColorRole.WindowText, t.text),
                        (QPalette.ColorRole.Base, t.field), (QPalette.ColorRole.AlternateBase, t.card),
                        (QPalette.ColorRole.Text, t.text), (QPalette.ColorRole.Button, t.button),
                        (QPalette.ColorRole.ButtonText, t.text), (QPalette.ColorRole.Highlight, t.accent),
                        (QPalette.ColorRole.HighlightedText, t.accent_text), (QPalette.ColorRole.ToolTipBase, t.card),
                        (QPalette.ColorRole.ToolTipText, t.text), (QPalette.ColorRole.PlaceholderText, t.muted),
                        (QPalette.ColorRole.Link, t.accent)):
        p.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(t.disabled_text))
    return p


def stylesheet(t: Tokens) -> str:
    icons = (resources_dir() / "icons").as_posix()
    name = "dark" if t is DARK else "light"
    return f"""
QWidget {{ color: {t.text}; }}
QMainWindow, QDialog {{ background: {t.window}; }}
QMenuBar {{ background: {t.window}; border: none; padding: 2px 6px; }}
QMenuBar::item {{ padding: 5px 10px; border-radius: 6px; background: transparent; }}
QMenuBar::item:selected {{ background: {t.button_hover}; }}
QMenu {{ background: {t.card}; border: 1px solid {t.card_border}; border-radius: 8px; padding: 6px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
QMenu::item:selected {{ background: {t.selection}; color: {t.text}; }}
QStatusBar {{ background: {t.window}; color: {t.muted}; }}
QGroupBox {{ background: {t.card}; border: 1px solid {t.card_border}; border-radius: 12px; margin-top: 22px;
            padding: 12px 14px 12px 14px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px 4px 6px;
                   color: {t.muted}; }}
QFrame#header {{ background: transparent; }}
QFrame#card {{ background: {t.card}; border: 1px solid {t.card_border}; border-radius: 12px; }}
QLabel#headerTitle {{ font-size: 20px; font-weight: 700; }}
QLabel#headerSubtitle, QLabel#mutedLabel {{ color: {t.muted}; }}
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit, QTableWidget {{
    background: {t.field}; border: 1px solid {t.field_border}; border-radius: 8px;
    selection-background-color: {t.selection}; selection-color: {t.text}; }}
QLineEdit, QComboBox, QSpinBox {{ min-height: 30px; padding: 0 10px; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus {{ border: 1px solid {t.accent}; }}
QLineEdit#bigInput {{ min-height: 40px; font-size: 14px; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{ image: url({icons}/chevron_down_{name}.png); width: 12px; height: 12px; }}
QSpinBox {{ padding-right: 22px; }}
QSpinBox::up-button, QSpinBox::down-button {{ border: none; background: transparent; width: 22px; }}
QSpinBox::up-arrow {{ image: url({icons}/chevron_up_{name}.png); width: 10px; height: 10px; }}
QSpinBox::down-arrow {{ image: url({icons}/chevron_down_{name}.png); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{ background: {t.card}; border: 1px solid {t.card_border}; border-radius: 8px;
                              selection-background-color: {t.selection}; selection-color: {t.text}; padding: 4px; }}
QPlainTextEdit#logView {{ background: {t.log}; font-family: "Cascadia Mono", "Consolas", monospace;
                          font-size: 12px; padding: 6px; }}
QPushButton, QToolButton {{ background: {t.button}; border: 1px solid {t.card_border}; border-radius: 8px;
                            padding: 7px 16px; min-height: 18px; }}
QPushButton:hover, QToolButton:hover {{ background: {t.button_hover}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t.disabled_text}; background: {t.card};
                                             border-color: {t.card_border}; }}
QPushButton#primaryButton {{ background: {t.accent}; color: {t.accent_text}; border: none; font-weight: 700;
                             padding: 9px 26px; }}
QPushButton#primaryButton:hover {{ background: {t.accent_hover}; }}
QPushButton#primaryButton:disabled {{ background: {t.button}; color: {t.disabled_text}; }}
QToolButton#themeButton {{ padding: 6px 12px; border-radius: 16px; }}
QProgressBar {{ background: {t.button}; border: none; border-radius: 7px; min-height: 14px; max-height: 14px;
               text-align: center; color: {t.text}; font-size: 11px; }}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 7px; }}
QHeaderView::section {{ background: {t.card}; color: {t.muted}; border: none;
                        border-bottom: 1px solid {t.card_border}; padding: 6px; font-weight: 600; }}
QTableWidget {{ gridline-color: {t.card_border}; }}
QTableWidget::item:selected {{ background: {t.selection}; color: {t.text}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {t.field_border}; border-radius: 4px;
                        background: {t.field}; }}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; image: url({icons}/check.png); }}
QCheckBox::indicator:disabled {{ background: {t.card}; border-color: {t.card_border}; }}
QSplitter::handle {{ background: {t.card_border}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t.field_border}; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t.field_border}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QToolTip {{ background: {t.card}; color: {t.text}; border: 1px solid {t.card_border}; padding: 4px 8px; }}
"""


def apply(app: QApplication, mode: str) -> str:
    """Apply a theme setting; returns the resolved "light" or "dark"."""
    resolved = resolve(mode)
    tokens = DARK if resolved == "dark" else LIGHT
    app.setStyle("Fusion")
    if sys.platform == "win32":
        app.setFont(QFont("Segoe UI", 10))
    app.setPalette(_palette(tokens))
    app.setStyleSheet(stylesheet(tokens))
    app.setProperty("resolvedTheme", resolved)
    return resolved


def follow_system(app: QApplication, current_mode) -> None:
    """Re-apply when Windows switches light/dark while the theme setting is "system"."""
    hints = QGuiApplication.styleHints()
    signal = getattr(hints, "colorSchemeChanged", None)
    if signal is not None:
        signal.connect(lambda *_: apply(app, current_mode()) if current_mode() == "system" else None)
