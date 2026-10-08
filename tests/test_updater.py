"""Update check, verified download and the update window (D-118). No test touches the network."""

import hashlib
import io
import json
import threading
from pathlib import Path

import pytest

from app import __version__
from app.core.errors import JobCancelled
from app.services import updater
from app.services.updater import UpdateError, UpdateInfo
from app.ui import update_dialog
from app.ui.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = b"installer-bytes" * 1000
GOOD_URL = updater.DOWNLOAD_PREFIX + "v9.9.9/" + updater.ASSET_NAME


def _release(tag="v9.9.9", url=GOOD_URL, digest=None, name=updater.ASSET_NAME):
    asset = {"name": name, "browser_download_url": url, "size": len(PAYLOAD)}
    asset["digest"] = digest if digest is not None else "sha256:" + hashlib.sha256(PAYLOAD).hexdigest()
    return json.dumps({"tag_name": tag, "body": "notes", "assets": [asset]}).encode()


def _info(**changes):
    values = dict(version="9.9.9", page_url=updater.RELEASES_PAGE, download_url=GOOD_URL, size=len(PAYLOAD),
                  sha256=hashlib.sha256(PAYLOAD).hexdigest())
    values.update(changes)
    return UpdateInfo(**values)


class _Response(io.BytesIO):
    headers: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# -- versions ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("latest,current,newer", [
    ("1.0.1", "1.0.0", True), ("v1.1.0", "1.0.9", True), ("2.0", "1.9.9", True), ("1.0.0", "1.0.0", False),
    ("1.1", "1.1.0", False), ("1.0.0", "1.2.0", False), ("1.0.0-beta", "0.9.0", False), ("junk", "1.0.0", False),
    ("1.10.0", "1.9.0", True),
])
def test_is_newer(latest, current, newer):
    assert updater.is_newer(latest, current) is newer


def test_the_program_version_is_a_plain_version():
    assert updater.parse_version(__version__) is not None


# -- release parsing --------------------------------------------------------------------------------------------

def test_parse_release_reads_version_asset_and_digest():
    info = updater.parse_release(_release())

    assert info.version == "9.9.9" and info.download_url == GOOD_URL and info.size == len(PAYLOAD)
    assert info.sha256 == hashlib.sha256(PAYLOAD).hexdigest()


@pytest.mark.parametrize("payload", [
    b"not json", b"{}", _release(tag="nightly"), _release(url="https://evil.example/x.exe"),
    _release(url="http://github.com/" + updater.REPO + "/releases/download/v1/x.exe"),
    _release(digest="md5:abc"), _release(name="other.exe"), json.dumps({"tag_name": "v1.0.0", "assets": []}).encode(),
])
def test_unusable_releases_are_refused(payload):
    with pytest.raises(UpdateError):
        updater.parse_release(payload)


def test_check_latest_turns_network_errors_into_update_error():
    def broken(url):
        raise OSError("offline")

    with pytest.raises(UpdateError):
        updater.check_latest(broken)


def test_check_latest_asks_only_the_project_repository():
    asked = []
    updater.check_latest(lambda url: asked.append(url) or _release())

    assert asked == [updater.LATEST_URL] and updater.REPO in asked[0]


# -- download ---------------------------------------------------------------------------------------------------

def test_download_verifies_and_returns_the_file(tmp_path):
    seen = []

    path = updater.download_installer(_info(), seen.append, folder=tmp_path, opener=lambda url: _Response(PAYLOAD))

    assert path.read_bytes() == PAYLOAD and seen and seen[-1] == 1.0
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_a_wrong_hash_is_discarded(tmp_path):
    with pytest.raises(UpdateError, match="SHA-256"):
        updater.download_installer(_info(), folder=tmp_path, opener=lambda url: _Response(PAYLOAD[:-1] + b"X"))

    assert list(tmp_path.iterdir()) == []


def test_a_short_download_is_discarded(tmp_path):
    with pytest.raises(UpdateError):
        updater.download_installer(_info(), folder=tmp_path, opener=lambda url: _Response(PAYLOAD[:100]))

    assert list(tmp_path.iterdir()) == []


def test_cancel_stops_the_download_and_leaves_nothing(tmp_path):
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(JobCancelled):
        updater.download_installer(_info(), cancel=cancel, folder=tmp_path, opener=lambda url: _Response(PAYLOAD))

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("changes", [{"download_url": "https://evil.example/a.exe"}, {"sha256": ""}])
def test_an_unverifiable_update_is_never_downloaded(tmp_path, changes):
    def forbidden(url):
        raise AssertionError("must not download")

    with pytest.raises(UpdateError):
        updater.download_installer(_info(**changes), folder=tmp_path, opener=forbidden)


def test_redirects_must_stay_on_https():
    handler = updater._HttpsOnly()
    with pytest.raises(Exception):
        handler.redirect_request(None, None, 302, "Found", {}, "http://example.com/x")


def test_launch_runs_the_installer_silently_and_detached(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(updater.subprocess, "Popen", lambda args, **kw: calls.append((args, kw)))

    updater.launch_installer(tmp_path / "setup.exe")

    args, kwargs = calls[0]
    assert args[0].endswith("setup.exe") and "/SILENT" in args and "/CLOSEAPPLICATIONS" in args
    assert "/VERYSILENT" not in args and kwargs["close_fds"] is True


def test_a_source_checkout_cannot_replace_itself():
    assert updater.can_self_update() is False


# -- wiring -----------------------------------------------------------------------------------------------------

def test_defaults_and_installer_script(settings):
    assert settings.get("update_check") is True and settings.get("update_skipped") == ""
    iss = (ROOT / "packaging/installer.iss").read_text(encoding="utf-8")
    assert "Check: WizardSilent" in iss and "AppId=" in iss and "CloseApplications=yes" in iss


def test_the_automatic_check_is_the_only_start_hook():
    main = (ROOT / "app/main.py").read_text(encoding="utf-8")
    assert 'settings.get("update_check")' in main and "check_for_updates(manual=False)" in main


def _window(qtbot, translator, settings):
    window = MainWindow(translator, settings)
    qtbot.addWidget(window)
    return window


def test_constructing_the_window_never_checks_the_network(qtbot, translator, settings):
    window = _window(qtbot, translator, settings)

    assert window._update_runner is None and window.update_action.text()


def test_an_automatic_check_is_silent_when_up_to_date_or_skipped(qtbot, translator, settings, monkeypatch):
    shown = []
    monkeypatch.setattr(update_dialog.UpdateDialog, "exec", lambda self: shown.append(self))
    monkeypatch.setattr("app.ui.main_window.QMessageBox.information", lambda *a, **k: shown.append("info"))
    window = _window(qtbot, translator, settings)

    window._on_update_checked(_info(version=__version__), manual=False)
    settings.set("update_skipped", "9.9.9")
    window._on_update_checked(_info(), manual=False)

    assert shown == []


def test_a_manual_check_reports_up_to_date_and_errors(qtbot, translator, settings, monkeypatch):
    messages = []
    monkeypatch.setattr("app.ui.main_window.QMessageBox.information", lambda *a, **k: messages.append(a[2]))
    monkeypatch.setattr("app.ui.main_window.QMessageBox.warning", lambda *a, **k: messages.append(a[2]))
    window = _window(qtbot, translator, settings)

    window._on_update_checked(_info(version=__version__), manual=True)
    window._on_update_check_failed("offline", manual=True)
    window._on_update_check_failed("offline", manual=False)

    assert len(messages) == 2 and __version__ in messages[0] and "offline" in messages[1]


def test_a_newer_release_opens_the_window_and_skip_is_remembered(qtbot, translator, settings, monkeypatch):
    def fake_exec(self):
        self.skipped = True

    monkeypatch.setattr(update_dialog.UpdateDialog, "exec", fake_exec)
    window = _window(qtbot, translator, settings)

    window._on_update_checked(_info(), manual=False)

    assert settings.get("update_skipped") == "9.9.9"


def test_the_check_runner_reports_found_and_failed(qtbot):
    runner = update_dialog.UpdateCheckRunner(fetch=lambda url: _release())
    with qtbot.waitSignal(runner.found, timeout=5000) as found:
        runner.start()
    runner.wait()
    assert found.args[0].version == "9.9.9"

    def broken(url):
        raise OSError("offline")

    runner = update_dialog.UpdateCheckRunner(fetch=broken)
    with qtbot.waitSignal(runner.failed, timeout=5000) as failed:
        runner.start()
    runner.wait()
    assert "offline" in failed.args[0]


def test_dialog_without_self_update_opens_the_release_page(qtbot, translator, monkeypatch):
    opened = []
    monkeypatch.setattr(update_dialog.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    dialog = update_dialog.UpdateDialog(translator, _info(), "1.0.0", allow_skip=True, can_update=False)
    qtbot.addWidget(dialog)

    dialog.update_button.click()

    assert opened == [updater.RELEASES_PAGE]


def test_dialog_blocks_the_update_while_a_job_runs(qtbot, translator):
    dialog = update_dialog.UpdateDialog(translator, _info(), "1.0.0", allow_skip=False, can_update=True, busy=True)
    qtbot.addWidget(dialog)

    assert not dialog.update_button.isEnabled() and dialog.skip_button.isHidden()


def test_dialog_download_then_launch(qtbot, translator, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(updater, "download_installer", lambda info, progress, cancel: tmp_path / "setup.exe")
    monkeypatch.setattr(updater, "launch_installer", lambda path: started.append(path))
    dialog = update_dialog.UpdateDialog(translator, _info(), "1.0.0", allow_skip=True, can_update=True)
    qtbot.addWidget(dialog)

    with qtbot.waitSignal(dialog.accepted, timeout=5000):
        dialog.update_button.click()

    assert dialog.launched and started == [tmp_path / "setup.exe"]


def test_dialog_shows_a_failed_download_and_stays_open(qtbot, translator, monkeypatch):
    def fail(info, progress, cancel):
        raise UpdateError("hash mismatch")

    monkeypatch.setattr(updater, "download_installer", fail)
    dialog = update_dialog.UpdateDialog(translator, _info(), "1.0.0", allow_skip=True, can_update=True)
    qtbot.addWidget(dialog)

    dialog.update_button.click()
    qtbot.waitUntil(lambda: "hash mismatch" in dialog.label.text(), timeout=5000)

    assert not dialog.launched and dialog.update_button.isEnabled()
