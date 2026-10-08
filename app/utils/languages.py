"""Subtitle language catalogue (independent of the UI language).

Codes are ISO 639-1. The list will be filtered against ASR/MT backend support in M1.
"""

from __future__ import annotations

AUTO_DETECT = "auto"

SUBTITLE_LANGUAGES: tuple[str, ...] = (
    "ar", "de", "en", "es", "fa", "fr", "hi", "id", "it", "ja",
    "ko", "nl", "pl", "pt", "ru", "tr", "uk", "ur", "zh",
)


_ENGLISH_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fa": "Persian", "fr": "French",
    "hi": "Hindi", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean", "nl": "Dutch",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "zh": "Chinese",
}
RTL_LANGUAGES = {"ar", "fa", "ur", "he"}


def english_name(code: str) -> str:
    """English language name for prompts; unknown codes are returned unchanged."""
    return _ENGLISH_NAMES.get(code.split("-")[0].lower(), code)


def is_supported(code: str) -> bool:
    return code in SUBTITLE_LANGUAGES


def native_name(code: str) -> str:
    """Language name in its own script, provided by Qt locale data."""
    from PySide6.QtCore import QLocale

    locale = QLocale(code)
    if locale.language() == QLocale.Language.C:
        return code
    return locale.nativeLanguageName() or code
