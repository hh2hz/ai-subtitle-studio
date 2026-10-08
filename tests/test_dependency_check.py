"""Automatic library check and repair (D-114): offline logic, the yt-dlp overlay and the start-up window."""

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pytest

from app.services import dependency_check as deps
from app.services import overlay

REQ = deps.Requirement("PySide6", "6.11.2", True)
NOW = 1_800_000_000.0


def _wheel(module: str, version: str = "9.9.9", extra: dict | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{module}/__init__.py", f"__version__ = {version!r}\n")
        archive.writestr(f"{module}-{version}.dist-info/METADATA", f"Name: {module}\nVersion: {version}\n")
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _payload(name: str, version: str, data: bytes) -> dict:
    return {"info": {"version": version},
            "urls": [{"packagetype": "bdist_wheel", "filename": f"{name}-{version}-py3-none-any.whl",
                      "url": f"https://files.pythonhosted.org/packages/x/{name}-{version}-py3-none-any.whl",
                      "digests": {"sha256": hashlib.sha256(data).hexdigest()}}]}


class _Net:
    """Fake PyPI: serves one wheel per package and counts calls."""

    def __init__(self, version="2099.1.1", corrupt=False):
        self.version, self.corrupt = version, corrupt
        self.wheels = {n: _wheel(m, version) for n, m in overlay.PACKAGES.items()}
        self.json_calls = 0
        self.byte_calls = 0

    def fetch_json(self, url, timeout):
        self.json_calls += 1
        name = url.split("/pypi/")[1].split("/")[0]
        return _payload(name.replace("-", "_"), self.version, self.wheels[name])

    def fetch_bytes(self, url, timeout):
        self.byte_calls += 1
        data = next(w for n, w in self.wheels.items() if f"/{n.replace('-', '_')}-{self.version}" in url)
        return data + b"x" if self.corrupt else data


@pytest.fixture
def clean_overlay():
    path = list(sys.path)
    modules = {name for name in sys.modules}
    yield
    overlay.activate_dirs({})                                  # never leave a finder or a fake folder behind
    sys.path[:] = path
    for name in set(sys.modules) - modules:
        if name.split(".")[0] in ("yt_dlp", "yt_dlp_ejs"):
            del sys.modules[name]


def _healthy(monkeypatch):
    monkeypatch.setattr(deps.importlib.metadata, "version", lambda name: "6.11.2")
    monkeypatch.setattr(deps, "_find_spec", lambda module: object())
    monkeypatch.setattr(deps, "installed_version", lambda name: "2026.1.1")


def _no_pip(*args):
    raise AssertionError("pip must not run")


# -- parsing and the offline checks ----------------------------------------------------------------

def test_parse_requirements_reads_pins_extras_and_skips_noise():
    parsed = deps.parse_requirements("""
    # runtime dependencies
    PySide6==6.11.2            # LGPL-3.0 / GPL
    yt-dlp[default]==2026.8.19 # extra
    numpy>=2.0
    -r other.txt
    av
    """)

    assert [(r.name, r.version, r.pinned) for r in parsed] == [
        ("PySide6", "6.11.2", True), ("yt-dlp", "2026.8.19", True), ("numpy", "2.0", False), ("av", "", False)]


def test_import_names_cover_the_packages_that_differ():
    assert deps.Requirement("faster-whisper").module == "faster_whisper"
    assert deps.Requirement("yt-dlp").module == "yt_dlp"


def test_a_missing_package_is_an_error(monkeypatch):
    def missing(name):
        raise deps.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(deps.importlib.metadata, "version", missing)
    problems = deps.check_environment([REQ], tools=False)

    assert problems and problems[0].level == "error" and "not installed" in problems[0].title


def test_a_version_that_differs_from_the_pin_is_an_error(monkeypatch):
    monkeypatch.setattr(deps.importlib.metadata, "version", lambda name: "6.10.0")
    monkeypatch.setattr(deps, "_find_spec", lambda module: object())

    problems = deps.check_environment([REQ], tools=False)

    assert len(problems) == 1 and "6.10.0 differs from the pinned 6.11.2" in problems[0].title


def test_yt_dlp_is_floating_so_a_newer_release_is_not_an_error(monkeypatch):
    monkeypatch.setattr(deps.importlib.metadata, "version", lambda name: "2099.1.1")
    monkeypatch.setattr(deps, "_find_spec", lambda module: object())

    assert deps.check_environment([deps.Requirement("yt-dlp", "2026.8.19", True)], tools=False) == []


def test_an_unpinned_requirement_accepts_any_version(monkeypatch):
    monkeypatch.setattr(deps.importlib.metadata, "version", lambda name: "9.9.9")
    monkeypatch.setattr(deps, "_find_spec", lambda module: object())

    assert deps.check_environment([deps.Requirement("numpy", "2.0", False)], tools=False) == []


def test_a_missing_tool_is_a_warning_from_sources_and_an_error_when_installed(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda tool: None)

    source = deps.check_environment([], tools=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    installed = deps.check_environment([], tools=True)

    assert {p.title.split()[0] for p in source} == {"ffmpeg", "ffprobe", "deno"}
    assert all(p.level == "warning" for p in source) and all(p.level == "error" for p in installed)


def test_python_version_is_checked(monkeypatch):
    monkeypatch.setattr(deps.sys, "version_info", (3, 9, 0))

    problems = deps.check_environment([], tools=False)

    assert problems and problems[0].level == "error" and "3.12" in problems[0].title


def test_missing_modules_are_reported_for_the_installed_build(monkeypatch):
    monkeypatch.setattr(deps, "_find_spec", lambda module: None if module == "numpy" else object())

    problems = deps.check_environment([], tools=False, modules=("numpy", "av"))

    assert [p.title for p in problems] == ["The component numpy is missing from this installation"]


def test_requirements_path_points_at_the_project_file():
    path = deps.requirements_path()
    assert path is not None and path.is_file() and path.name == "requirements.txt"


# -- repair of pinned libraries ------------------------------------------------------------------

def test_a_healthy_environment_runs_no_pip_and_reports_no_error(monkeypatch, tmp_path):
    _healthy(monkeypatch)

    result = deps.perform_check(requirements=[REQ], runner=_no_pip, root=tmp_path, last_lookup=NOW, now=NOW,
                                tools=False, frozen=False)

    assert result.ok and not result.restart and not result.repaired and result.checked == 1


def test_a_missing_pin_is_repaired_with_pip_without_upgrading(monkeypatch, tmp_path):
    state = {"broken": True}
    calls = []

    def version(name):
        if state["broken"]:
            raise deps.importlib.metadata.PackageNotFoundError(name)
        return "6.11.2"

    def pip(command, cwd, progress):
        calls.append(command)
        progress("Installing...")
        state["broken"] = False
        return 0

    monkeypatch.setattr(deps.importlib.metadata, "version", version)
    monkeypatch.setattr(deps, "_find_spec", lambda module: object())
    monkeypatch.setattr(deps, "installed_version", lambda name: "2026.1.1")
    lines = []

    result = deps.perform_check(requirements=[REQ], runner=pip, root=tmp_path, last_lookup=NOW, now=NOW,
                                tools=False, frozen=False, restarted=False, progress=lines.append)

    assert result.ok and result.repaired and result.restart
    assert calls and calls[0][1:4] == ["-m", "pip", "install"] and "-r" in calls[0]
    assert "--upgrade" not in calls[0] and "-U" not in calls[0]            # never past the pins
    assert "Installing..." in lines


def test_no_second_restart_is_requested_after_a_restart(monkeypatch, tmp_path):
    state = {"broken": True}

    def version(name):
        if state["broken"]:
            raise deps.importlib.metadata.PackageNotFoundError(name)
        return "6.11.2"

    monkeypatch.setattr(deps.importlib.metadata, "version", version)
    monkeypatch.setattr(deps, "_find_spec", lambda module: object())
    monkeypatch.setattr(deps, "installed_version", lambda name: "2026.1.1")

    result = deps.perform_check(requirements=[REQ], runner=lambda c, d, p: state.update(broken=False) or 0,
                                root=tmp_path, last_lookup=NOW, now=NOW, tools=False, frozen=False, restarted=True)

    assert result.ok and result.repaired and not result.restart


def test_a_failed_repair_is_an_error_the_user_sees(monkeypatch, tmp_path):
    monkeypatch.setattr(deps.importlib.metadata, "version",
                        lambda name: (_ for _ in ()).throw(deps.importlib.metadata.PackageNotFoundError(name)))
    monkeypatch.setattr(deps, "_find_spec", lambda module: object())
    monkeypatch.setattr(deps, "installed_version", lambda name: "2026.1.1")

    result = deps.perform_check(requirements=[REQ], runner=lambda c, d, p: 1, root=tmp_path, last_lookup=NOW,
                                now=NOW, tools=False, frozen=False)

    assert not result.ok and "PySide6 is not installed" in result.errors[0]
    assert any("pip exited with code 1" in note for note in result.notes)


def test_the_installed_build_never_calls_pip(monkeypatch, tmp_path):
    monkeypatch.setattr(deps, "_find_spec", lambda module: None if module == "numpy" else object())
    monkeypatch.setattr(deps, "installed_version", lambda name: "2026.1.1")

    result = deps.perform_check(modules=("numpy", "av"), frozen=True, runner=_no_pip, root=tmp_path,
                                last_lookup=NOW, now=NOW, tools=False)

    assert not result.ok and "numpy" in result.errors[0] and result.checked == 2


def test_the_installed_build_with_everything_present_checks_something(monkeypatch, tmp_path):
    monkeypatch.setattr(deps, "_find_spec", lambda module: object())
    monkeypatch.setattr(deps, "installed_version", lambda name: "2026.1.1")

    result = deps.perform_check(modules=("numpy", "av"), frozen=True, runner=_no_pip, root=tmp_path,
                                last_lookup=NOW, now=NOW, tools=False)

    assert result.ok and result.checked == 2


# -- the yt-dlp overlay ----------------------------------------------------------------------------

def test_a_newer_yt_dlp_is_downloaded_verified_and_activated(monkeypatch, tmp_path, clean_overlay):
    _healthy(monkeypatch)
    net = _Net()

    result = deps.perform_check(requirements=[], runner=_no_pip, root=tmp_path, last_lookup=0.0, now=NOW,
                                tools=False, frozen=False, fetch_json=net.fetch_json, fetch_bytes=net.fetch_bytes,
                                probe=lambda folder, name: (True, "ok"))

    assert result.ok and result.looked_up
    assert result.updated == ["yt-dlp 2026.1.1 -> 2099.1.1", "yt-dlp-ejs 2026.1.1 -> 2099.1.1"]
    active = overlay.valid_overlays(tmp_path)
    assert active["yt-dlp"][0][0] == "2099.1.1" and (active["yt-dlp"][0][1] / "yt_dlp" / "__init__.py").is_file()


def test_the_overlay_wins_over_an_already_installed_copy(tmp_path, clean_overlay):
    folder = overlay.install_wheel("yt-dlp-ejs", "9.9.9", _wheel("yt_dlp_ejs"),
                                   hashlib.sha256(_wheel("yt_dlp_ejs")).hexdigest(), tmp_path)
    overlay.mark_ok(folder)

    assert overlay.activate(tmp_path) == {"yt-dlp-ejs": "9.9.9"}
    assert isinstance(sys.meta_path[0], overlay._OverlayFinder)
    spec = sys.meta_path[0].find_spec("yt_dlp_ejs")
    assert spec is not None and Path(spec.origin).is_relative_to(folder)


def test_a_checksum_mismatch_is_rejected_and_nothing_is_kept(monkeypatch, tmp_path, clean_overlay):
    _healthy(monkeypatch)
    net = _Net(corrupt=True)

    result = deps.perform_check(requirements=[], runner=_no_pip, root=tmp_path, last_lookup=0.0, now=NOW,
                                tools=False, frozen=False, fetch_json=net.fetch_json, fetch_bytes=net.fetch_bytes,
                                probe=lambda folder, name: (True, "ok"))

    assert result.ok and not result.updated                          # the installed copy keeps working
    assert any("SHA-256" in note for note in result.notes)
    assert overlay.valid_overlays(tmp_path) == {}
    site = tmp_path / "site"
    assert not site.exists() or not [p for p in site.iterdir() if p.is_dir()]


def test_a_release_that_fails_the_import_probe_is_rolled_back_and_not_retried(monkeypatch, tmp_path, clean_overlay):
    _healthy(monkeypatch)
    net = _Net()
    kwargs = dict(requirements=[], runner=_no_pip, root=tmp_path, last_lookup=0.0, now=NOW, tools=False,
                  frozen=False, fetch_json=net.fetch_json, fetch_bytes=net.fetch_bytes,
                  probe=lambda folder, name: (False, "boom"))

    first = deps.perform_check(**kwargs)
    downloads = net.byte_calls
    second = deps.perform_check(**kwargs)

    assert first.ok and not first.updated and overlay.valid_overlays(tmp_path) == {}
    assert overlay.is_known_bad("yt-dlp", "2099.1.1", tmp_path)
    assert net.byte_calls == downloads and second.ok                 # the bad release is not downloaded again


def test_offline_is_silent_and_does_not_start_the_pause(monkeypatch, tmp_path, clean_overlay):
    _healthy(monkeypatch)

    def offline(url, timeout):
        raise OSError("no route")

    result = deps.perform_check(requirements=[], runner=_no_pip, root=tmp_path, last_lookup=0.0, now=NOW,
                                tools=False, frozen=False, fetch_json=offline)

    assert result.ok and not result.looked_up and not result.updated
    assert any("Could not look up yt-dlp" in note for note in result.notes)


def test_the_lookup_is_paused_for_six_hours(monkeypatch, tmp_path, clean_overlay):
    _healthy(monkeypatch)
    net = _Net()

    deps.perform_check(requirements=[], runner=_no_pip, root=tmp_path, last_lookup=NOW - 3600, now=NOW,
                       tools=False, frozen=False, fetch_json=net.fetch_json, fetch_bytes=net.fetch_bytes)
    assert net.json_calls == 0

    deps.perform_check(requirements=[], runner=_no_pip, root=tmp_path, last_lookup=NOW - 7 * 3600, now=NOW,
                       tools=False, frozen=False, fetch_json=net.fetch_json, fetch_bytes=net.fetch_bytes,
                       probe=lambda folder, name: (True, "ok"))
    assert net.json_calls == 2


def test_the_lookup_budget_is_shared_by_all_packages(monkeypatch, tmp_path, clean_overlay):
    _healthy(monkeypatch)
    seen = []
    ticks = iter([0.0, 0.1, 100.0, 100.0, 100.0])

    def fetch(url, timeout):
        seen.append(timeout)
        raise ValueError("no wheel")

    result = deps.Result()
    deps.update_floating(lambda text: None, result, root=tmp_path, fetch_json=fetch, clock=lambda: next(ticks))

    assert len(seen) == 1 and seen[0] <= 4.0
    assert any("ran out of time" in note for note in result.notes)


def test_an_equal_or_older_release_changes_nothing(monkeypatch, tmp_path, clean_overlay):
    _healthy(monkeypatch)
    net = _Net(version="2026.1.1")

    result = deps.perform_check(requirements=[], runner=_no_pip, root=tmp_path, last_lookup=0.0, now=NOW,
                                tools=False, frozen=False, fetch_json=net.fetch_json, fetch_bytes=net.fetch_bytes)

    assert not result.updated and net.byte_calls == 0


def test_unsafe_archives_and_foreign_download_hosts_are_refused(tmp_path):
    evil = _wheel("yt_dlp", extra={"../escape.txt": "x"})
    with pytest.raises(ValueError):
        overlay.install_wheel("yt-dlp", "9.9.9", evil, hashlib.sha256(evil).hexdigest(), tmp_path)
    assert not (tmp_path / "escape.txt").exists()
    assert not overlay.allowed_url("https://example.com/yt_dlp.whl")
    assert overlay.allowed_url("https://files.pythonhosted.org/packages/a/b.whl")


def test_only_the_newest_two_releases_are_kept(tmp_path):
    for version in ("1.0.0", "1.1.0", "1.2.0"):
        data = _wheel("yt_dlp", version)
        overlay.mark_ok(overlay.install_wheel("yt-dlp", version, data, hashlib.sha256(data).hexdigest(), tmp_path))

    overlay.prune(tmp_path)

    assert [v for v, _ in overlay.valid_overlays(tmp_path)["yt-dlp"]] == ["1.2.0", "1.1.0"]


def test_the_real_probe_accepts_a_good_release_and_rejects_a_broken_one(tmp_path):
    data = _wheel("yt_dlp_ejs", "9.9.9")
    good = overlay.install_wheel("yt-dlp-ejs", "9.9.9", data, hashlib.sha256(data).hexdigest(), tmp_path)
    ok, detail = deps.probe_overlay(good, "yt-dlp-ejs")
    broken = tmp_path / "broken"
    (broken / "yt_dlp_ejs").mkdir(parents=True)
    (broken / "yt_dlp_ejs" / "__init__.py").write_text("raise RuntimeError('broken')\n", encoding="utf-8")
    bad, bad_detail = deps.probe_overlay(broken, "yt-dlp-ejs")

    assert ok, detail
    assert not bad and "broken" in bad_detail


def test_a_module_that_is_not_loaded_from_the_overlay_fails_the_probe(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert deps.overlay_probe_main(str(empty), "json") == 1          # json comes from the standard library


# -- the start-up window -----------------------------------------------------------------------------

def _dialog(qtbot, translator, settings, result=None, error=None):
    from app.ui.dependency_dialog import DependencyDialog

    def perform(progress):
        progress("Checking the libraries...")
        if error:
            raise error
        return result or deps.Result()

    dialog = DependencyDialog(translator, settings, perform=perform)
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: dialog.report is not None, timeout=5000)
    return dialog


def _forbid_message(dialog):
    def fail(errors):
        raise AssertionError("no error message expected")

    dialog._ask_continue = fail


def test_the_window_says_what_it_is_doing(qtbot, translator, settings):
    from app.ui.dependency_dialog import DependencyDialog

    dialog = DependencyDialog(translator, settings, perform=lambda progress: deps.Result())
    qtbot.addWidget(dialog)

    assert dialog.status_label.text() == translator.t("deps.checking") == "Checking libraries for updates…"
    qtbot.waitUntil(lambda: dialog.report is not None, timeout=5000)


def test_the_window_closes_by_itself_when_everything_is_fine(qtbot, translator, settings):
    from app.ui.dependency_dialog import OUTCOME_OK

    dialog = _dialog(qtbot, translator, settings)

    assert dialog.outcome == OUTCOME_OK and not dialog.isVisible()


def test_a_remaining_problem_shows_an_error_with_quit_or_continue(qtbot, translator, settings, monkeypatch):
    from app.ui.dependency_dialog import OUTCOME_CONTINUE, OUTCOME_QUIT, DependencyDialog

    shown = []
    for answer, expected in ((True, OUTCOME_CONTINUE), (False, OUTCOME_QUIT)):
        monkeypatch.setattr(DependencyDialog, "_ask_continue",
                            lambda self, errors, answer=answer: shown.append(errors) or answer)
        dialog = _dialog(qtbot, translator, settings, result=deps.Result(errors=["PySide6 is not installed"]))
        assert dialog.outcome == expected

    assert shown == [["PySide6 is not installed"]] * 2


def test_a_crash_inside_the_check_becomes_an_error_message(qtbot, translator, settings, monkeypatch):
    from app.ui.dependency_dialog import OUTCOME_QUIT, DependencyDialog

    seen = []
    monkeypatch.setattr(DependencyDialog, "_ask_continue", lambda self, errors: seen.append(errors) or False)

    dialog = _dialog(qtbot, translator, settings, error=RuntimeError("disk exploded"))

    assert dialog.outcome == OUTCOME_QUIT and "disk exploded" in seen[0][0]


def test_a_repair_that_replaced_loaded_libraries_asks_for_a_restart(qtbot, translator, settings):
    from app.ui.dependency_dialog import OUTCOME_RESTART

    dialog = _dialog(qtbot, translator, settings, result=deps.Result(repaired=True, restart=True))

    assert dialog.outcome == OUTCOME_RESTART


def test_only_a_successful_lookup_is_stamped(qtbot, translator, settings):
    _dialog(qtbot, translator, settings, result=deps.Result(looked_up=False))
    assert settings.get("dependency_check_last") == 0.0

    _dialog(qtbot, translator, settings, result=deps.Result(looked_up=True))
    assert settings.get("dependency_check_last") > 1_700_000_000


def test_stop_is_safe_twice(qtbot, translator, settings):
    dialog = _dialog(qtbot, translator, settings)

    dialog.stop()
    dialog.stop()

    assert dialog._thread is None


def test_arabic_strings_exist_for_every_message(translator):
    keys = ("deps.title", "deps.checking", "deps.failed", "deps.restarting", "deps.error_title", "deps.error_body",
            "deps.continue_anyway", "deps.quit")
    english = {k: translator.t(k) for k in keys}
    translator.set_language("ar")
    try:
        assert all(translator.t(k) not in (k, english[k]) for k in keys)
    finally:
        translator.set_language("en")


def test_the_default_job_never_touches_the_settings_database_from_the_worker_thread(qtbot, translator, settings,
                                                                                      monkeypatch):
    """Regression: SQLite connections are bound to their thread; the worker must not call Settings.get."""
    from app.ui.dependency_dialog import DependencyDialog

    seen = {}
    monkeypatch.setattr(DependencyDialog, "_ask_continue", lambda self, errors: False)   # never block a test
    monkeypatch.setattr(deps, "perform_check", lambda **kw: seen.update(kw) or deps.Result())
    settings.set("dependency_check_last", 123.0)

    dialog = DependencyDialog(translator, settings)             # the real default job, a real Settings
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: dialog.report is not None, timeout=5000)

    assert dialog.report.errors == [] and seen["last_lookup"] == 123.0
