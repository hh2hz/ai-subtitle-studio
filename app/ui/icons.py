"""Application icon (app/resources/icons/app.png, also used for the Windows .ico)."""

from __future__ import annotations

from PySide6.QtGui import QIcon

from app.utils.paths import resources_dir


def app_icon() -> QIcon:
    path = resources_dir() / "icons" / "app.png"
    return QIcon(str(path)) if path.is_file() else QIcon()
