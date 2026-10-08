"""Cookie policy (probe without cookies first) and the retrying move out of the work folder (D-085).

No network: yt-dlp's YoutubeDL is replaced by a fake that returns scripted results per cookie mode.
"""

import pytest

from app.core import downloader as dl
from app.core.downloader import YtDlpAdapter, access_options
from app.core.errors import PipelineError

URL = "https://example.invalid/v"
BOT = "ERROR: [youtube] x: Sign in to confirm you're not a bot. Use --cookies-from-browser"


def _info(height: int, **extra) -> dict:
    return {"id": "x", "height": height, "format_id": str(height),
            "formats": [{"vcodec": "avc1", "height": height}], **extra}


def _fake(monkeypatch, plain, cookie, written=None):
    """`plain` / `cookie`: info dict or Exception, used for the cookie-less / cookie calls."""
    calls: list[dict] = []

    class FakeYDL:
        def __init__(self, options):
            self.options = options
            calls.append(options)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            outcome = cookie if self.options.get("cookiesfrombrowser") else plain
            if download and written is not None:
                written()
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def sanitize_info(self, info):
            return info

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    return calls


def _adapter():
    extras = access_options({"cookies_source": "browser", "cookies_browser": "firefox"})
    return YtDlpAdapter(retry_delays=(), sleep=lambda _: None, extra_options=extras)


def test_cookie_less_probe_wins_when_cookies_hide_the_hd_formats(monkeypatch):
    calls = _fake(monkeypatch, _info(1080), _info(360))
    adapter = _adapter()

    assert dl.available_qualities(adapter.probe(URL)) == [1080]
    assert len(calls) == 1 and not calls[0].get("cookiesfrombrowser")     # HD found: cookies never touched
    assert adapter.access_note and "Cookies skipped" in adapter.access_note


def test_cookies_are_used_when_the_cookie_less_probe_needs_a_sign_in(monkeypatch):
    calls = _fake(monkeypatch, Exception(BOT), _info(720))
    adapter = _adapter()

    assert dl.available_qualities(adapter.probe(URL)) == [720]
    assert calls[-1].get("cookiesfrombrowser")
    assert adapter._cookies_for[URL] is True


def test_the_higher_of_two_reduced_probes_wins(monkeypatch):
    _fake(monkeypatch, _info(480), _info(720))
    adapter = _adapter()
    assert dl.available_qualities(adapter.probe(URL)) == [720]
    assert adapter._cookies_for[URL] is True


def test_a_failing_cookie_probe_does_not_break_a_working_cookie_less_one(monkeypatch):
    _fake(monkeypatch, _info(480), Exception("ERROR: could not copy chrome cookie database"))
    adapter = _adapter()
    assert dl.available_qualities(adapter.probe(URL)) == [480]


def test_other_cookie_less_errors_are_not_hidden(monkeypatch):
    _fake(monkeypatch, Exception("ERROR: Unsupported URL: x"), _info(1080))
    with pytest.raises(PipelineError) as excinfo:
        _adapter().probe(URL)
    assert excinfo.value.ui_key == "error.url_unsupported"


def test_download_uses_the_probe_decision_and_falls_back_to_cookies(monkeypatch, tmp_path):
    out = tmp_path / "out"
    work = tmp_path / "work"
    dest = out / "ep"

    def write():
        (work / "ep.mp4").write_bytes(b"x" * 10)

    calls = _fake(monkeypatch, _info(1080), _info(360), written=write)
    adapter = _adapter()
    adapter.probe(URL)
    path = adapter.download_video(URL, dest, work_dir=work)
    assert path == out / "ep.mp4" and path.read_bytes() == b"x" * 10
    assert not (work / "ep.mp4").exists()                       # moved, not copied
    assert not calls[-1].get("cookiesfrombrowser")
    assert calls[-1]["file_access_retries"] >= 10

    # A cookie-less download that is refused with a sign-in demand is retried with the cookies.
    calls.clear()
    adapter2 = _adapter()
    adapter2._cookies_for[URL] = False
    _fake_signin = _fake(monkeypatch, Exception(BOT), _info(720), written=write)
    adapter2.download_video(URL, dest, work_dir=work)
    assert not _fake_signin[0].get("cookiesfrombrowser") and _fake_signin[-1].get("cookiesfrombrowser")


def test_quality_warning_names_the_cookie_cause(monkeypatch, tmp_path):
    adapter = _adapter()
    warnings = []
    adapter._report_quality(_info(360, formats=[{"vcodec": "avc1", "height": 1080}]), "best", warnings.append,
                            cookies=True)
    assert warnings and "cookie setting is on" in warnings[0]


# -- moving the finished file ----------------------------------------------------------------------

def _locked(monkeypatch, failures):
    state = {"n": 0}
    real = dl.os.replace

    def replace(src, dst):
        if state["n"] < failures:
            state["n"] += 1
            err = PermissionError(13, "locked")
            err.winerror = 32
            raise err
        return real(src, dst)

    monkeypatch.setattr(dl.os, "replace", replace)
    return state


def test_move_retries_while_the_file_is_locked(monkeypatch, tmp_path):
    src, dst = tmp_path / "a.mp4", tmp_path / "out" / "a.mp4"
    dst.parent.mkdir()
    src.write_bytes(b"data")
    sleeps = []
    adapter = YtDlpAdapter(sleep=sleeps.append)
    _locked(monkeypatch, 3)       # rename fails, copy-then-replace fails, ... until the lock is gone

    assert adapter._move_finished(src, dst) == dst
    assert dst.read_bytes() == b"data" and not src.exists()
    assert sleeps and max(sleeps) <= dl.MOVE_MAX_BACKOFF_S


def test_move_gives_a_clear_error_and_keeps_the_file(monkeypatch, tmp_path):
    src, dst = tmp_path / "a.mp4", tmp_path / "a_out.mp4"
    src.write_bytes(b"data")
    adapter = YtDlpAdapter(sleep=lambda _: None)
    _locked(monkeypatch, 10_000)

    with pytest.raises(PipelineError) as excinfo:
        adapter._move_finished(src, dst)
    assert "could not be moved" in str(excinfo.value) and src.read_bytes() == b"data"
    assert not (tmp_path / "a_out.mp4.copying").exists()


# -- subtitles withheld by YouTube (PO token) -------------------------------------------------------

def test_missing_po_token_is_unavailable_not_an_error(monkeypatch, tmp_path, caplog):
    from app.providers.base import ProviderUnavailable
    from app.providers.youtube import PlatformSubtitleProvider
    from app.providers.base import SubtitleRequest

    class FakeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            self.options["logger"].warning("There are missing subtitles languages because a PO token was not "
                                           "provided. Automatic captions for 1 languages are missing.")

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    adapter = YtDlpAdapter(retry_delays=(), sleep=lambda _: None)
    with pytest.raises(ProviderUnavailable):
        adapter.download_subtitle(URL, "tr-orig", True, tmp_path)

    provider = PlatformSubtitleProvider(adapter, sleep=lambda _: None)
    request = SubtitleRequest(url=URL, media_info={"automatic_captions": {"tr-orig": [{"ext": "vtt"}]}},
                              source_language="tr", target_language="ar", series=None, work_dir=str(tmp_path))
    with caplog.at_level("WARNING"):
        assert provider.fetch(request) == []
    assert "failed" not in caplog.text
