"""Application-wide logging configuration.

Named logging_setup (not logging) to avoid confusion with the standard library module.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
APP_LOG_NAME = "app.log"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3
_OWNED_ATTR = "_aiss_owned"


def setup_logging(log_dir: Path, level: str = "INFO") -> Path:
    """Configure root logging to a rotating file (and stderr if present). Idempotent.

    A log file that cannot be written must never stop the application: a second instance holding `app.log`, a
    restricted shell (a sandbox denies writes outside the workspace) or a scanner can all make `open()` fail. The
    handler then falls back to a timestamped file and, if even that fails, logging stays on stderr (D-109).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / APP_LOG_NAME
    root = logging.getLogger()

    # Remove handlers installed by a previous call so re-initialisation does not duplicate output.
    for handler in list(root.handlers):
        if getattr(handler, _OWNED_ATTR, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(LOG_FORMAT)
    file_handler, log_file = _open_log_file(log_file, formatter)
    if file_handler is not None:
        root.addHandler(file_handler)

    # A windowed (pythonw / frozen GUI) process on Windows has no stderr.
    if sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        setattr(stream_handler, _OWNED_ATTR, True)
        root.addHandler(stream_handler)

    set_level(level)
    if file_handler is None:
        logging.getLogger(__name__).warning("Running without a log file; see the messages above")
    return log_file


def _open_log_file(log_file: Path, formatter: logging.Formatter) -> tuple[logging.Handler | None, Path]:
    """Open the rotating log, falling back to a timestamped file and then to no file at all."""
    for candidate in (log_file, log_file.with_name(
            f"{log_file.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}{log_file.suffix}")):
        try:
            handler = RotatingFileHandler(candidate, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT,
                                          encoding="utf-8")
        except OSError as exc:
            _complain(f"cannot write {candidate} ({exc})")
            continue
        handler.setFormatter(formatter)
        setattr(handler, _OWNED_ATTR, True)
        if candidate != log_file:
            _complain(f"{log_file.name} is not writable; using {candidate.name}")
        return handler, candidate
    return None, log_file


def _complain(message: str) -> None:
    """Tell the user about a logging problem without a configured logger (it may have no handler yet)."""
    text = f"AI Subtitle Studio: {message}"
    if sys.stderr is not None:
        print(text, file=sys.stderr, flush=True)


def set_level(level: str) -> None:
    logging.getLogger().setLevel(level.upper())


def install_excepthook() -> None:
    """Log uncaught exceptions (including those raised inside Qt slots)."""

    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.getLogger("app").critical(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
        )

    sys.excepthook = _hook
