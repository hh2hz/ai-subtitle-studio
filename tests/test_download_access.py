"""Site access: error classes, cookie/network options and the single IPv4 retry (D-073).

No network: yt-dlp's YoutubeDL is replaced by a fake that records the options it was given.
"""

import pytest

from app.core.downloader import COOKIE_BROWSERS, YtDlpAdapter, _classify, access_options
from app.core.errors import PipelineError

BOT_CHECK = ("ERROR: [youtube] hFTH4t66b4E: Sign in to confirm you're not a bot. Use --cookies-from-browser "
             "or --cookies for the authentication.")
VIMEO_LOGIN = ("ERROR: [vimeo] 76979871: The web client only works when logged-in. Use --cookies, "
               "--cookies-from-browser, --username and --password, --netrc-cmd, or --netrc (vimeo) to provide "
               "account credentials.")


def _fake_yt_dlp(monkeypatch, results):
    """Replace yt_dlp.YoutubeDL. Each call consumes one entry of `results`; an Exception raises."""
    outcomes = list(results)
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
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def sanitize_info(self, info):
            return info

        def download(self, urls):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    return calls


@pytest.mark.parametrize("message,expected", [
    (BOT_CHECK, "error.youtube_bot_check"),
    (VIMEO_LOGIN, "error.login_required"),
    ("ERROR: [youtube] x: Sign in to confirm your age. This video may be inappropriate for some users.",
     "error.age_restricted"),
    ("ERROR: [youtube] x: This video is not available in your country", "error.geo_blocked"),
    ("ERROR: [youtube] x: Failed to extract any player response", "error.po_token"),
    ("ERROR: Unsupported URL: https://example.invalid/watch", "error.url_unsupported"),
    ("ERROR: [youtube] x: Video unavailable", "error.video_unavailable"),
    ("ERROR: HTTP Error 429: Too Many Requests", "error.rate_limited"),
    ("ERROR: unable to download webpage: timed out", "error.download_failed"),
])
def test_classify_maps_site_messages(message, expected):
    assert _classify(Exception(message)).ui_key == expected


def test_access_options_from_settings():
    assert access_options({}) == {}
    assert access_options({"cookies_source": "browser", "cookies_browser": "firefox"}) == {
        "cookiesfrombrowser": ("firefox", None, None, None)}
    # A source without its detail is ignored instead of producing a broken yt-dlp option.
    assert access_options({"cookies_source": "browser"}) == {}
    assert access_options({"cookies_source": "file", "cookies_file": r"C:\x\cookies.txt"}) == {
        "cookiefile": r"C:\x\cookies.txt"}
    assert access_options({"force_ipv4": True}) == {"force_ipv4": True}
    assert COOKIE_BROWSERS[0] == "firefox"


def test_probe_passes_cookie_options_to_yt_dlp(monkeypatch):
    # The cookie-less probe comes first (D-085); it offers no video here, so the cookie probe follows.
    calls = _fake_yt_dlp(monkeypatch, [{"id": "x", "_type": "video"}, {"id": "x", "_type": "video"}])
    adapter = YtDlpAdapter(extra_options=access_options({"cookies_source": "browser", "cookies_browser": "brave"}))

    assert adapter.probe("https://example.invalid/v")["id"] == "x"
    assert calls[0]["cookiesfrombrowser"] is None
    assert calls[1]["cookiesfrombrowser"] == ("brave", None, None, None)
    assert calls[1]["skip_download"] is True


def test_probe_retries_once_over_ipv4_after_a_bot_check(monkeypatch):
    calls = _fake_yt_dlp(monkeypatch, [Exception(BOT_CHECK), {"id": "x", "_type": "video"}])
    adapter = YtDlpAdapter(retry_delays=(), sleep=lambda _: None)

    assert adapter.probe("https://example.invalid/v")["id"] == "x"
    assert len(calls) == 2
    assert calls[0].get("force_ipv4") is None
    assert calls[1]["force_ipv4"] is True


def test_probe_does_not_retry_other_errors(monkeypatch):
    calls = _fake_yt_dlp(monkeypatch, [Exception("ERROR: Unsupported URL: https://example.invalid/x")])
    adapter = YtDlpAdapter(retry_delays=(), sleep=lambda _: None)

    with pytest.raises(PipelineError) as excinfo:
        adapter.probe("https://example.invalid/x")
    assert excinfo.value.ui_key == "error.url_unsupported"
    assert len(calls) == 1


def test_no_second_ipv4_attempt_when_it_was_already_forced(monkeypatch):
    calls = _fake_yt_dlp(monkeypatch, [Exception(BOT_CHECK)])
    adapter = YtDlpAdapter(retry_delays=(), sleep=lambda _: None, extra_options={"force_ipv4": True})

    with pytest.raises(PipelineError) as excinfo:
        adapter.probe("https://example.invalid/v")
    assert excinfo.value.ui_key == "error.youtube_bot_check"
    assert len(calls) == 1


def test_download_video_keeps_cookie_options(monkeypatch, tmp_path):
    calls = _fake_yt_dlp(monkeypatch, [{"id": "x"}])
    adapter = YtDlpAdapter(extra_options=access_options({"cookies_source": "file",
                                                        "cookies_file": str(tmp_path / "cookies.txt")}))
    dest = tmp_path / "ep"
    (tmp_path / "ep.mp4").write_bytes(b"x")          # what the real yt-dlp would have written
    adapter._cookies_for["https://example.invalid/v"] = True      # the probe decided that cookies are needed

    assert adapter.download_video("https://example.invalid/v", dest) == tmp_path / "ep.mp4"
    assert calls[0]["cookiefile"] == str(tmp_path / "cookies.txt")
    assert calls[0]["merge_output_format"] == "mp4"
