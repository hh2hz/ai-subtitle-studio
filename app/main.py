"""Application entry point: python -m app.main"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sqlite3
import sys
from pathlib import Path

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app import __version__
from app.services import overlay

overlay.activate()      # a newer yt-dlp downloaded earlier wins over the bundled copy (D-114); must precede its import

from app.core.downloader import YtDlpAdapter  # noqa: E402
from app.services import series_lookup
from app.database.database import Database
from app.database.jobs import JobsRepo
from app.database.settings import Settings
from app.services.job_manager import default_pipeline_factory
from app.ui import theme
from app.ui.icons import app_icon
from app.ui.main_window import MainWindow, RuntimeContext
from app.utils.i18n import FALLBACK_LANGUAGE, Translator
from app.utils.logging_setup import install_excepthook, set_level, setup_logging
from app.utils.paths import (AppPaths, bundled_bin_dir, default_data_root, resolve_data_root,
                             translations_dir)

log = logging.getLogger("app")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="AISubtitleStudio")
    parser.add_argument("--data-dir", help="Override the user data directory")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Start the GUI, process one event loop cycle, and exit with code 0")
    return parser.parse_args(argv)


# Modules the packaged application must be able to import. Third-party packages first, then the application
# modules that are loaded while a job runs (they are reached through the pipeline, but a packaging mistake such as
# a missing hidden import only shows up here).
SELF_TEST_MODULES = (
    "ctranslate2", "faster_whisper", "av", "numpy", "yt_dlp", "yt_dlp_ejs", "anyascii", "sherpa_onnx",
    "sentencepiece", "huggingface_hub", "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
    "app.core.diarize", "app.core.frames", "app.core.platform_text", "app.core.redecode", "app.core.risk",
    "app.core.windows",
)

# Every file the code loads from app/resources: the UI catalogues, the icon set the stylesheet points at
# (app/ui/theme.py), the application icon, the name stop list, the style guides and the model ranking files.
# tests/test_packaging_pins.py fails when app/resources ships a file that is not listed here.
SELF_TEST_RESOURCES = (
    "icons/app.ico", "icons/app.png", "icons/check.png", "icons/chevron_down_dark.png",
    "icons/chevron_down_light.png", "icons/chevron_up_dark.png", "icons/chevron_up_light.png",
    "model_ranking.json", "model_ranking_estimates.json", "name_stoplist.json", "style_guides.json",
    "translations/ar.json", "translations/en.json",
)


def self_test(report: str | None = None) -> int:
    """Checks that a packaged build contains everything (used by packaging/build_windows.ps1). Exit code 0 = ok.
    The windowed executable has no console, so the result can also be written to a report file."""
    import importlib
    import shutil

    from app.utils.paths import resources_dir

    problems = []

    def progress(text: str) -> None:
        """Write the step that is running, so a hang shows where it stopped."""
        if report:
            try:
                Path(report).write_text(f"running: {text}\n", encoding="utf-8")
            except OSError:
                pass

    for module in SELF_TEST_MODULES:
        progress(f"import {module}")
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            if module.startswith("app."):
                problems.append(f"module missing: {module} ({exc})")
            else:
                problems.append(f"import {module}: {exc}")
    try:
        from faster_whisper.utils import get_assets_path

        if not any(Path(get_assets_path()).glob("*.onnx")):
            problems.append("faster_whisper VAD model missing")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"faster_whisper assets: {exc}")
    try:
        from anyascii import anyascii

        if anyascii("\u041f\u043e\u043b\u0430\u0442") != "Polat":
            problems.append("anyascii data wrong")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"anyascii: {exc}")
    root = resources_dir()
    for rel in SELF_TEST_RESOURCES:
        if not (root / rel).is_file():
            problems.append(f"resource missing: {rel}")
    bundled = bundled_bin_dir()
    if bundled is not None:
        os.environ["PATH"] = f"{bundled}{os.pathsep}{os.environ.get('PATH', '')}"
    lines = [f"{tool}: {shutil.which(tool) or 'NOT FOUND'}" for tool in ("ffmpeg", "ffprobe", "deno")]
    lines += [f"PROBLEM: {problem}" for problem in problems]
    lines.append("self-test " + ("FAILED" if problems else "ok"))
    if report:
        Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 1 if problems else 0


def _relaunch() -> bool:
    """Start this application again (loaded libraries were replaced). Returns False if that was not possible."""
    import subprocess

    from app.services import dependency_check

    env = dict(os.environ, **{dependency_check.RESTART_ENV: "1"})      # a second restart is never requested
    try:
        subprocess.Popen(dependency_check.restart_command(), env=env)
    except OSError as exc:
        log.warning("Cannot restart the application: %s", exc)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--self-test"]:
        return self_test(argv[1] if len(argv) > 1 else None)
    if argv[:1] == ["--overlay-probe"] and len(argv) == 3:
        # Child-process import check of a freshly downloaded yt-dlp release (app.services.dependency_check).
        from app.services.dependency_check import overlay_probe_main

        return overlay_probe_main(argv[1], argv[2])
    if argv[:1] == ["--gpu-probe"]:
        # Child-process GPU check used by the frozen executable (see app.services.gpu_probe).
        from app.services.gpu_probe import run_probe

        return run_probe(*argv[1:4])
    args = parse_args(argv)
    bundled = bundled_bin_dir()
    if bundled is not None:
        # FFmpeg, FFprobe and Deno shipped with the installer come first on PATH (yt-dlp and the burner use them).
        os.environ["PATH"] = f"{bundled}{os.pathsep}{os.environ.get('PATH', '')}"
    preferred_root = Path(args.data_dir) if args.data_dir else default_data_root()
    try:
        root, fallback_note = resolve_data_root(preferred_root)
    except OSError as exc:                      # nothing writable at all: say so instead of a traceback
        print(f"AI Subtitle Studio: {exc}", file=sys.stderr)
        return 2
    if fallback_note:
        print(f"AI Subtitle Studio: {fallback_note}", file=sys.stderr)
    paths = AppPaths.from_root(root).ensure().with_models_from(preferred_root)
    log_file = setup_logging(paths.logs_dir)
    install_excepthook()
    log.info("AI Subtitle Studio %s | Python %s | PySide6 %s | %s",
             __version__, platform.python_version(), pyside_version, platform.platform())
    log.info("Data directory: %s | models: %s | log file: %s", paths.root, paths.models_dir, log_file)
    if fallback_note:
        log.warning("%s", fallback_note)

    try:
        db = Database(paths.db_file)
    except sqlite3.Error as exc:                # a locked or read-only folder must be readable to the user
        message = f"Cannot open the settings database {paths.db_file}: {exc}"
        print(f"AI Subtitle Studio: {message}", file=sys.stderr)
        log.critical("%s", message)
        return 2
    try:
        settings = Settings(db)
        set_level(settings.get("log_level"))
        interrupted = JobsRepo(db).mark_interrupted()
        if interrupted:
            log.warning("%d job(s) were interrupted; starting the same input again resumes from cache",
                        interrupted)

        app = QApplication.instance() or QApplication(sys.argv[:1])
        app.setWindowIcon(app_icon())
        theme.apply(app, settings.get("theme"))
        theme.follow_system(app, lambda: settings.get("theme"))
        translator = Translator(translations_dir())
        ui_language = settings.get("ui_language")
        if ui_language not in translator.available_languages():
            log.warning("Unknown UI language %r in settings; using %s", ui_language, FALLBACK_LANGUAGE)
            ui_language = FALLBACK_LANGUAGE
        translator.set_language(ui_language)

        # Automatic library check before the window opens (D-114): repairs what is broken, updates yt-dlp and
        # shows an error only when something is still wrong afterwards.
        if settings.get("dependency_check") and not args.smoke_test:
            from app.ui.dependency_dialog import OUTCOME_QUIT, OUTCOME_RESTART, run_check_dialog

            outcome = run_check_dialog(translator, settings)
            if outcome == OUTCOME_QUIT:
                return 1
            if outcome == OUTCOME_RESTART and _relaunch():
                return 0

        # First start: ask for the cloud AI keys once (D-117). Settings -> API keys opens the same window later.
        if not args.smoke_test:
            from app.ui.api_keys_dialog import maybe_prompt_first_run

            maybe_prompt_first_run(translator, settings)

        models_dir = settings.get("model_dir") or str(paths.models_dir)
        runtime = RuntimeContext(
            db_path=paths.db_file,
            pipeline_factory=default_pipeline_factory(paths.jobs_dir, Path(models_dir), paths.root),
            cache_dir=paths.cache_dir,
            downloader=YtDlpAdapter(),      # used to read the series/season/episode out of a pasted link
            series_fetch=series_lookup.http_get,   # TVMaze suggestions for the series field
        )
        window = MainWindow(translator, settings, runtime)
        window.show()
        if settings.get("update_check") and not args.smoke_test:
            window.check_for_updates(manual=False)        # silent unless a newer release exists (D-118)
        if not args.smoke_test:
            QTimer.singleShot(1500, window.offer_gpu_download)
        if args.smoke_test:
            QTimer.singleShot(0, app.quit)
        exit_code = app.exec()
        log.info("Exiting with code %d", exit_code)
        return exit_code
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
