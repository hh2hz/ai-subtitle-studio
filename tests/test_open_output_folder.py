"""The "Open output folder" button opens the configured output root, not one episode folder (D-108)."""

from pathlib import Path


def _window(qtbot, translator, settings, db, tmp_path):
    from app.ui.main_window import MainWindow, RuntimeContext

    runtime = RuntimeContext(db_path=db.path, pipeline_factory=lambda *args, **kwargs: None,
                             cache_dir=tmp_path / "cache")
    window = MainWindow(translator, settings, runtime)
    qtbot.addWidget(window)
    return window


def _capture(monkeypatch):
    opened = []
    import app.ui.main_window as ui

    monkeypatch.setattr(ui.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    return opened


def test_opens_the_configured_output_root_not_the_episode_folder(qtbot, translator, settings, db, tmp_path, monkeypatch):
    window = _window(qtbot, translator, settings, db, tmp_path)
    root = tmp_path / "out"
    episode = root / "Kurtlar Vadisi Pusu - 200. Bölüm FULL HD"
    episode.mkdir(parents=True)
    window.output_edit.setText(str(root))
    window._last_output_dir = episode          # what a finished job leaves behind
    opened = _capture(monkeypatch)

    window._open_output()

    assert [Path(p) for p in opened] == [root]


def test_empty_field_opens_the_default_output_folder(qtbot, translator, settings, db, tmp_path, monkeypatch):
    import app.ui.main_window as ui

    window = _window(qtbot, translator, settings, db, tmp_path)
    window.output_edit.setText("")
    default = tmp_path / "default-output"
    default.mkdir()                                # the default folder is created on demand by a job
    monkeypatch.setattr(ui, "default_output_dir", lambda: default)
    opened = _capture(monkeypatch)

    window._open_output()

    assert [Path(p) for p in opened] == [default]


def test_missing_folder_is_created_and_opened(qtbot, translator, settings, db, tmp_path, monkeypatch):
    window = _window(qtbot, translator, settings, db, tmp_path)
    target = tmp_path / "not-there-yet"
    window.output_edit.setText(str(target))
    opened = _capture(monkeypatch)

    window._open_output()

    assert target.is_dir()
    assert [Path(p) for p in opened] == [target]


def test_the_button_is_enabled_from_the_start(qtbot, translator, settings, db, tmp_path):
    window = _window(qtbot, translator, settings, db, tmp_path)

    assert window.open_output_button.isEnabled()


def test_the_button_stays_enabled_while_a_job_runs(qtbot, translator, settings, db, tmp_path):
    window = _window(qtbot, translator, settings, db, tmp_path)

    window._set_running_ui(True)

    assert window.open_output_button.isEnabled()
    assert not window.review_button.isEnabled()


def test_a_folder_that_cannot_be_opened_is_reported(qtbot, translator, settings, db, tmp_path, monkeypatch):
    import app.ui.main_window as ui
    from app.utils import open_folder as of

    window = _window(qtbot, translator, settings, db, tmp_path)
    monkeypatch.setattr(ui.QDesktopServices, "openUrl", lambda url: False)
    monkeypatch.setattr(of, "_diagnosed", True)

    def refused(*args, **kwargs):
        raise OSError("refused")

    monkeypatch.setattr(of.subprocess, "run", refused)

    assert window._open_dir(tmp_path) is False
    assert translator.t("error.open_folder_failed", path=str(tmp_path)) in window.log_view.toPlainText()
    assert ui.QGuiApplication.clipboard().text() == str(tmp_path)


def test_every_method_is_tried_in_turn_on_windows(monkeypatch, tmp_path, caplog):
    from app.utils import open_folder as of

    calls = []
    monkeypatch.setattr(of.sys, "platform", "win32")
    monkeypatch.setattr(of, "_diagnosed", True)

    def denied(path):
        calls.append("startfile")
        raise PermissionError(5, "Access is denied")

    def explore_ok(path):
        calls.append("explore")

    monkeypatch.setattr(of.os, "startfile", denied, raising=False)
    monkeypatch.setattr(of, "_shell_execute_explore", explore_ok)

    assert of.open_folder(tmp_path, lambda p: False) is True
    assert calls == ["startfile", "explore"]
    assert "Access is denied" in caplog.text


def test_the_process_state_is_logged_once_when_everything_fails(monkeypatch, tmp_path, caplog):
    from app.utils import open_folder as of

    monkeypatch.setattr(of.sys, "platform", "linux")
    monkeypatch.setattr(of, "_diagnosed", False)
    monkeypatch.setattr(of.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    monkeypatch.setattr(of, "process_diagnostics", lambda: "restricted_token=True")

    assert of.open_folder(tmp_path) is False
    assert of.open_folder(tmp_path) is False

    assert caplog.text.count("Process state: restricted_token=True") == 1


def test_process_diagnostics_never_raises():
    from app.utils import open_folder as of

    assert isinstance(of.process_diagnostics(), str)
