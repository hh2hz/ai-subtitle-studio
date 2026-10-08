"""UI localization: JSON string catalogues and runtime language/direction switching.

Each catalogue is resources/translations/<code>.json with the structure
{"_meta": {"native_name": ..., "direction": "ltr" | "rtl"}, "strings": {key: text}}.
Adding a UI language only requires adding a catalogue file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QGuiApplication

log = logging.getLogger(__name__)

FALLBACK_LANGUAGE = "en"


class TranslationError(RuntimeError):
    pass


class Translator(QObject):
    """Holds the active UI language and resolves string keys."""

    language_changed = Signal(str)

    def __init__(self, translations_dir: Path, fallback: str = FALLBACK_LANGUAGE, parent=None):
        super().__init__(parent)
        self._strings: dict[str, dict[str, str]] = {}
        self._meta: dict[str, dict[str, str]] = {}
        for path in sorted(Path(translations_dir).glob("*.json")):
            self._load(path)
        if fallback not in self._strings:
            raise TranslationError(f"Fallback catalogue {fallback}.json not found in {translations_dir}")
        self._fallback = fallback
        self._language = fallback
        self._reported_missing: set[tuple[str, str]] = set()

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta, strings = data["_meta"], data["strings"]
            if meta.get("direction") not in ("ltr", "rtl") or not meta.get("native_name"):
                raise ValueError("_meta needs native_name and direction ltr|rtl")
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in strings.items()):
                raise ValueError("strings must map str -> str")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TranslationError(f"Invalid translation catalogue {path}: {exc}") from exc
        self._meta[path.stem] = meta
        self._strings[path.stem] = strings

    # -- queries ---------------------------------------------------------------------------

    @property
    def language(self) -> str:
        return self._language

    def available_languages(self) -> list[str]:
        others = sorted(code for code in self._strings if code != self._fallback)
        return [self._fallback, *others]

    def native_name(self, code: str) -> str:
        return self._meta[code]["native_name"]

    def is_rtl(self, code: str | None = None) -> bool:
        return self._meta[code or self._language]["direction"] == "rtl"

    def layout_direction(self, code: str | None = None) -> Qt.LayoutDirection:
        return Qt.LayoutDirection.RightToLeft if self.is_rtl(code) else Qt.LayoutDirection.LeftToRight

    def keys(self, code: str) -> set[str]:
        return set(self._strings[code])

    def raw(self, code: str, key: str) -> str | None:
        return self._strings[code].get(key)

    def has(self, key: str) -> bool:
        return key in self._strings[self._fallback]

    def t(self, key: str, **kwargs) -> str:
        """Return the text for key in the active language (fallback language, then the key)."""
        text = self._strings[self._language].get(key)
        if text is None:
            self._report_missing(self._language, key)
            text = self._strings[self._fallback].get(key)
            if text is None:
                return key
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError, ValueError) as exc:
                log.warning("Formatting failed for key %r: %s", key, exc)
        return text

    def _report_missing(self, code: str, key: str) -> None:
        if (code, key) not in self._reported_missing:
            self._reported_missing.add((code, key))
            log.warning("Missing translation: %s/%s", code, key)

    # -- mutation --------------------------------------------------------------------------

    def set_language(self, code: str) -> None:
        """Activate a UI language and apply its layout direction application-wide."""
        if code not in self._strings:
            raise ValueError(f"Unsupported UI language: {code}")
        changed = code != self._language
        self._language = code
        app = QGuiApplication.instance()
        if app is not None:
            app.setLayoutDirection(self.layout_direction())
        if changed:
            log.info("UI language changed to %s", code)
            self.language_changed.emit(code)
