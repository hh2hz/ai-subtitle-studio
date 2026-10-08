"""Typed application settings persisted as JSON values in SQLite."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from app.core.downloader import COOKIE_BROWSERS, QUALITY_BEST
from app.core.modes import DEFAULT_MODE, Mode
from app.database.database import Database
from app.utils import languages

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "ui_language": "en",
    "theme": "system",           # "system" (follow Windows), "light" or "dark"
    "source_language": languages.AUTO_DETECT,
    "target_language": "ar",
    "mode": DEFAULT_MODE.value,
    "output_dir": "",          # empty = default location
    "model_dir": "",           # empty = <data root>/models
    "log_level": "INFO",
    "first_run_completed": False,
    "fetch_platform_subtitles": True,
    "opensubtitles_api_key": "",
    "subdl_api_key": "",
    "translation_engine": "local",                  # "local" (built-in TranslateGemma) or "madlad"
    "local_model": "translategemma-4b-q4km",
    "llm_refine": False,         # cloud LLMs with the user's API keys (off: everything stays on this computer)
    "llm_correct_only": False,   # on: cloud AI only corrects the local translation (fewer tokens, but keeps
                                 # awkward local lines, D-038); off (default): cloud AI translates every line
    "gpu_offer_shown": False,    # the NVIDIA library download was offered once (Settings menu offers it again)
    "burn_video": False,         # also save <title>.<lang>.subtitled.mp4 (subtitles drawn into the picture)
    "llm_review": True,          # a second AI model double-checks only the risky lines (D-104, replaces the reviewer)
    "audio_enhance": "off",      # music/noise removal + quiet speech boost: "auto" (Maximum accuracy), "on", "off";
                                 # off by default until it is shown to help (D-046)
    "diarization": True,         # speaker detection on the CPU after transcription (D-098)
    "snap_to_shots": False,      # snap cue edges to shot changes within 250 ms (D-103); slower, so off by default
    "name_normalization": True,  # deterministic name normalisation (D-058, task 2.6)
    # Site access (D-073). Cookie values are never read, copied, stored or logged by us: yt-dlp opens the
    # browser profile or the exported file itself, and only the choice is remembered here.
    "cookies_source": "",        # "": none, "browser" (the user's own session), "file" (an exported cookies.txt)
    "cookies_browser": "",       # browser name for --cookies-from-browser, e.g. "firefox"
    "cookies_file": "",          # path to the cookies.txt the user exported
    "force_ipv4": False,         # some networks are refused over IPv6; also retried once automatically
    "video_quality": QUALITY_BEST,   # "best" or a height in pixels for link downloads (D-080)
    "window_geometry": "",           # hex of the saved QMainWindow geometry; empty = default size/position (D-109)
    "window_geometry_version": 0,    # layout generation that saved it; an older one is ignored once (D-111)
    "api_keys_prompted": False,      # the first-start API keys window was shown (the keys themselves are never stored here)
    "keep_job_cache": False,         # keep the job's working files after a finished episode (debugging only, D-116)
    "dependency_check": True,        # check, repair and update the libraries before the window opens (D-114)
    "dependency_check_last": 0.0,    # unix time of the last successful PyPI lookup (6-hour pause between lookups)
    "update_check": True,            # look for a newer release on GitHub at every start (D-118)
    "update_skipped": "",            # a release version the user chose to skip ("" = none)
}

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
_MODES = {m.value for m in Mode}

_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "ui_language": lambda v: 2 <= len(v) <= 8,
    "theme": lambda v: v in ("system", "light", "dark"),
    "source_language": lambda v: v == languages.AUTO_DETECT or languages.is_supported(v),
    "target_language": languages.is_supported,
    "mode": lambda v: v in _MODES,
    "log_level": lambda v: v in _LOG_LEVELS,
    "translation_engine": lambda v: v in ("local", "madlad"),
    "local_model": lambda v: bool(v.strip()),
    "audio_enhance": lambda v: v in ("auto", "on", "off"),
    "cookies_source": lambda v: v in ("", "browser", "file"),
    "cookies_browser": lambda v: v == "" or v in COOKIE_BROWSERS,
    "cookies_file": lambda v: v == "" or v.lower().endswith(".txt"),
    "video_quality": lambda v: v == "best" or (v.isdigit() and 144 <= int(v) <= 4320),
    "window_geometry": lambda v: v == "" or (len(v) % 2 == 0 and len(v) <= 1024
                                             and all(c in "0123456789abcdefABCDEF" for c in v)),
    "window_geometry_version": lambda v: 0 <= v <= 1000,
    "dependency_check_last": lambda v: 0.0 <= v <= 1e11,
    "update_skipped": lambda v: v == "" or (len(v) <= 20 and all(c in "0123456789." for c in v)),
}


class Settings:
    def __init__(self, db: Database):
        self._db = db

    @staticmethod
    def _check(key: str, value: Any) -> None:
        if key not in DEFAULTS:
            raise KeyError(f"Unknown setting: {key}")
        expected = type(DEFAULTS[key])
        # Exact type match so that bool and int are not interchangeable.
        if type(value) is not expected:
            raise TypeError(f"Setting {key!r} expects {expected.__name__}, got {type(value).__name__}")
        validator = _VALIDATORS.get(key)
        if validator is not None and not validator(value):
            raise ValueError(f"Invalid value for setting {key!r}: {value!r}")

    def get(self, key: str) -> Any:
        if key not in DEFAULTS:
            raise KeyError(f"Unknown setting: {key}")
        row = self._db.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return DEFAULTS[key]
        try:
            value = json.loads(row["value"])
            self._check(key, value)
        except (ValueError, TypeError) as exc:
            log.warning("Stored setting %r is invalid (%s); using default", key, exc)
            return DEFAULTS[key]
        return value

    def set(self, key: str, value: Any) -> None:
        self._check(key, value)
        with self._db.conn:
            self._db.conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def reset(self, key: str) -> None:
        if key not in DEFAULTS:
            raise KeyError(f"Unknown setting: {key}")
        with self._db.conn:
            self._db.conn.execute("DELETE FROM settings WHERE key = ?", (key,))

    def all(self) -> dict[str, Any]:
        return {key: self.get(key) for key in DEFAULTS}
