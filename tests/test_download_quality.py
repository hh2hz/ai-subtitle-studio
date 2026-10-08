"""Download quality: offered heights, the format chain, and the warning when the result is lower (D-080).

No network: yt-dlp's YoutubeDL is replaced by a fake that returns a canned info dict.
"""

import logging
from pathlib import Path

import pytest

from app.core.downloader import QUALITY_BEST, YtDlpAdapter, available_qualities, format_selector

INFO = {
    "id": "x", "title": "Episode", "height": 360, "format_id": "18",
    "formats": [
        {"format_id": "18", "vcodec": "avc1.42001E", "acodec": "mp4a.40.2", "height": 360, "ext": "mp4"},
        {"format_id": "137", "vcodec": "avc1.640028", "acodec": "none", "height": 1080, "ext": "mp4"},
        {"format_id": "136", "vcodec": "avc1.4D401F", "acodec": "none", "height": 720, "ext": "mp4"},
        {"format_id": "135", "vcodec": "avc1.4D401E", "acodec": "none", "height": 480, "ext": "mp4"},
        {"format_id": "137b", "vcodec": "avc1.640028", "acodec": "none", "height": 1080, "ext": "mp4"},
        {"format_id": "140", "vcodec": "none", "acodec": "mp4a.40.2", "height": None, "ext": "m4a"},
        {"format_id": "160", "vcodec": "avc1.4D400C", "acodec": "none", "height": 144, "ext": "mp4"},
        {"format_id": "231", "vcodec": "avc1.4D401E", "acodec": "none", "ext": "mp4"},        # no height
    ],
}


def test_available_qualities_lists_distinct_video_heights_descending():
    assert available_qualities(INFO) == [1080, 720, 480, 360, 144]
    assert available_qualities({}) == []
    assert available_qualities(None) == []
    # audio-only links have nothing to choose from
    assert available_qualities({"formats": [{"vcodec": "none", "acodec": "mp4a", "height": None}]}) == []


@pytest.mark.parametrize("quality,expected", [
    (QUALITY_BEST, "bv*[protocol!*=m3u8]+ba[protocol!*=m3u8]/bv*+ba/b"),
    ("best", "bv*[protocol!*=m3u8]+ba[protocol!*=m3u8]/bv*+ba/b"),
    (None, "bv*[protocol!*=m3u8]+ba[protocol!*=m3u8]/bv*+ba/b"),
    ("1080", "bv*[height<=1080][protocol!*=m3u8]+ba[protocol!*=m3u8]"
             "/bv*[height<=1080]+ba/b[height<=1080]"),
    (720, "bv*[height<=720][protocol!*=m3u8]+ba[protocol!*=m3u8]"
          "/bv*[height<=720]+ba/b[height<=720]"),
])
def test_format_selector_prefers_direct_streams(quality, expected):
    """Direct (https/DASH) streams first, HLS (m3u8) only in the fallback chain (D-083)."""
    assert format_selector(quality, ffmpeg=True) == expected


@pytest.mark.parametrize("quality,expected", [
    (QUALITY_BEST, "b[protocol!*=m3u8]/b"),
    ("720", "b[height<=720][protocol!*=m3u8]/b[height<=720]/b"),
])
def test_format_selector_without_ffmpeg_stays_single_file(quality, expected):
    assert format_selector(quality, ffmpeg=False) == expected


# -- the chains as yt-dlp itself evaluates them (fake formats, no network) --------------------------

HLS_1080 = {"format_id": "hls1080", "height": 1080, "width": 1920, "fps": 25, "vcodec": "av01.0.08M.08",
            "acodec": "none", "protocol": "m3u8_native", "ext": "mp4", "tbr": 5724,
            "manifest_url": "https://x/hls.m3u8", "fragments": [{"url": "https://x/hls/f1", "duration": 5}]}
HTTPS_1080 = {"format_id": "https1080", "height": 1080, "width": 1920, "fps": 25, "vcodec": "av01.0.08M.08",
              "acodec": "none", "protocol": "https", "ext": "mp4", "tbr": 474}
HTTPS_720 = {"format_id": "https720", "height": 720, "width": 1280, "fps": 25, "vcodec": "av01.0.05M.08",
             "acodec": "none", "protocol": "https", "ext": "mp4", "tbr": 300}
AUDIO_HTTPS = {"format_id": "aud251", "vcodec": "none", "acodec": "opus", "protocol": "https", "ext": "webm",
               "abr": 102}


def _yt_dlp_selection(chain: str, formats: list) -> list:
    """Run the real yt-dlp format selector over a fake format list (no network, no download)."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "noprogress": True}) as ydl:
        return [fmt["format_id"] for fmt in ydl.build_format_selector(chain)(
            {"formats": formats, "incomplete_formats": False})]


def test_selector_chain_picks_the_progressive_stream_over_hls():
    """With an HLS and a direct 1080p stream plus audio, the chain must report the direct one."""
    chosen = _yt_dlp_selection(format_selector(QUALITY_BEST), [HLS_1080, HTTPS_1080, AUDIO_HTTPS])

    assert chosen == ["https1080+aud251"]
    assert all("hls" not in part for part in chosen[0].split("+"))


def test_selector_falls_back_to_hls_when_it_is_the_only_option():
    """The generic fallback keeps an HLS-only link downloadable instead of selecting nothing."""
    chosen = _yt_dlp_selection(format_selector(QUALITY_BEST), [HLS_1080, AUDIO_HTTPS])

    assert chosen == ["hls1080+aud251"]



def test_a_cache_key_includes_the_requested_quality(tmp_path, monkeypatch):
    """A cached 360p file must never satisfy a 1080p request."""
    from app.core.modes import Mode
    from app.core.pipeline import JobConfig, Pipeline

    def make(quality):
        config = JobConfig(None, "tr", "ar", Mode.FAST, source_url="https://example.invalid/v")
        return Pipeline(config, tmp_path / "jobs", [object()], [object()],
                        lambda p, o, c: None, lambda p, b, c: None, video_quality=quality)

    identity = {"url": "https://example.invalid/v"}
    keys = {make(q)._download_key(identity) for q in ("best", "720", "1080")}
    assert len(keys) == 3
    assert make("720")._download_key(identity) == make("720")._download_key(identity)

    # The chain version is part of the key too: a file fetched with an older chain is downloaded again (D-083).
    import app.core.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "CHAIN_VERSION", pipeline_module.CHAIN_VERSION + 1)
    assert make("720")._download_key(identity) not in keys


def _fake_yt_dlp(monkeypatch, info, written):
    """Replace yt_dlp.YoutubeDL: it writes the file the caller expects and returns `info`."""
    calls = []

    class FakeYDL:
        def __init__(self, options):
            self.options = options
            calls.append(options)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            Path(written["path"]).write_bytes(b"x")
            return info

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    return calls


def test_download_uses_the_height_chain_and_reports_a_lower_result(monkeypatch, tmp_path, caplog):
    dest = tmp_path / "ep"
    calls = _fake_yt_dlp(monkeypatch, INFO, {"path": str(dest) + ".mp4"})
    warnings = []
    adapter = YtDlpAdapter(retry_delays=(), sleep=lambda _: None)

    with caplog.at_level(logging.INFO):
        path = adapter.download_video("https://example.invalid/v", dest, quality="1080", warn=warnings.append)

    assert path == tmp_path / "ep.mp4"
    assert calls[0]["format"] == format_selector("1080")
    assert "protocol!*=m3u8" in calls[0]["format"]        # direct streams first (D-083)
    assert calls[0]["merge_output_format"] == "mp4"
    assert "Requested 1080p, got 360p (format_id 18)" in caplog.text
    assert warnings and "Requested 1080p, got 360p" in warnings[0]


def test_download_for_best_warns_when_a_better_height_was_offered(monkeypatch, tmp_path, caplog):
    dest = tmp_path / "ep"
    _fake_yt_dlp(monkeypatch, INFO, {"path": str(dest) + ".mp4"})
    warnings = []
    adapter = YtDlpAdapter(retry_delays=(), sleep=lambda _: None)

    with caplog.at_level(logging.INFO):
        adapter.download_video("https://example.invalid/v", dest, warn=warnings.append)

    assert "Requested bestp, got 360p" in caplog.text
    assert warnings and "Best offered was 1080p, got 360p" in warnings[0]


def test_download_says_nothing_when_the_best_was_downloaded(monkeypatch, tmp_path):
    dest = tmp_path / "ep"
    info = dict(INFO, height=1080, format_id="137+140")
    _fake_yt_dlp(monkeypatch, info, {"path": str(dest) + ".mp4"})
    warnings = []
    adapter = YtDlpAdapter(retry_delays=(), sleep=lambda _: None)

    adapter.download_video("https://example.invalid/v", dest, quality="1080", warn=warnings.append)

    assert warnings == []


def test_download_without_video_reports_no_height(monkeypatch, tmp_path, caplog):
    """An audio-only page has no height: no warning, and the log says so instead of inventing one."""
    dest = tmp_path / "ep"
    info = {"id": "x", "format_id": "140", "formats": [{"vcodec": "none", "acodec": "mp4a", "height": None}]}
    _fake_yt_dlp(monkeypatch, info, {"path": str(dest) + ".m4a"})
    warnings = []
    adapter = YtDlpAdapter(retry_delays=(), sleep=lambda _: None)

    with caplog.at_level(logging.INFO):
        adapter.download_video("https://example.invalid/v", dest, warn=warnings.append)

    assert warnings == []
    assert "got no videop" in caplog.text


# -- a transient 403 on the media data must be retried, not fatal (D-107) ---------------------------

def test_media_403_is_classified_as_retryable_download_error():
    from app.core.downloader import _classify

    error = _classify(RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden"))

    assert error.ui_key == "error.download_forbidden"
    assert error.ui_key in __import__("app.core.downloader", fromlist=["_TRANSIENT_KEYS"])._TRANSIENT_KEYS


def test_download_retries_a_media_403_with_a_fresh_attempt(monkeypatch, tmp_path, caplog):
    """The 200. Bölüm failure: the first attempt 403s, a later one works. The job must survive that."""
    dest = tmp_path / "ep"
    attempts = []

    def fake_extract(self, url, download=False):
        attempts.append(url)
        if len(attempts) == 1:
            Path(str(dest) + ".f399.mp4.part").write_bytes(b"stale")   # a leftover from the failed attempt
            raise RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
        assert not (tmp_path / "ep.f399.mp4.part").exists()            # cleared before the retry
        Path(str(dest) + ".mp4").write_bytes(b"x")
        return {"format_id": "399+251", "height": 1080, "formats": [{"vcodec": "av01", "height": 1080}]}

    import yt_dlp

    class FakeYDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        extract_info = fake_extract

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    slept = []
    adapter = YtDlpAdapter(retry_delays=(5.0, 20.0), sleep=slept.append)

    with caplog.at_level(logging.INFO):
        path = adapter.download_video("https://example.invalid/v", dest)

    assert len(attempts) == 2 and slept == [5.0]
    assert path == tmp_path / "ep.mp4"
    assert "Site refused the download (error.download_forbidden); retrying in 5 s" in caplog.text
    assert "Removed 1 unfinished download file(s) before retrying" in caplog.text


def test_a_media_403_after_every_attempt_still_fails(tmp_path, monkeypatch):
    """Retrying is bounded: three attempts, then the error reaches the user."""
    dest = tmp_path / "ep"
    calls = []

    def always_403(self, url, download=False):
        calls.append(url)
        raise RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")

    import yt_dlp

    class FakeYDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        extract_info = always_403

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    adapter = YtDlpAdapter(retry_delays=(5.0, 20.0), sleep=lambda _: None)

    with pytest.raises(Exception) as excinfo:
        adapter.download_video("https://example.invalid/v", dest)

    assert len(calls) == 3
    assert "403" in str(excinfo.value)


def test_partial_files_are_discarded_before_a_retry(tmp_path):
    from app.core.downloader import _discard_partials

    base = tmp_path / "episode"
    (tmp_path / "episode.f137.mp4.part").write_bytes(b"x")
    (tmp_path / "episode.mp4.ytdl").write_bytes(b"x")
    (tmp_path / "episode.mp4").write_bytes(b"finished")     # a finished file must never be touched
    (tmp_path / "episode.ar.srt").write_bytes(b"subtitle")

    assert _discard_partials(base) == 2
    assert (tmp_path / "episode.mp4").read_bytes() == b"finished"
    assert (tmp_path / "episode.ar.srt").is_file()


# -- the main window combo (A3) -------------------------------------------------------------------

def _window(qtbot, translator, settings, db, tmp_path):
    from app.ui.main_window import MainWindow, RuntimeContext

    runtime = RuntimeContext(db_path=db.path, pipeline_factory=lambda *args, **kwargs: None,
                             cache_dir=tmp_path / "cache")
    window = MainWindow(translator, settings, runtime)
    qtbot.addWidget(window)
    return window


def test_quality_combo_lists_the_heights_of_this_link(qtbot, translator, settings, db, tmp_path):
    window = _window(qtbot, translator, settings, db, tmp_path)

    window._on_qualities_probed([1080, 720, 480, 360])

    items = [window.quality_combo.itemText(i) for i in range(window.quality_combo.count())]
    assert items == ["Best available", "1080p", "720p", "480p", "360p"]
    assert window.quality_combo.currentData() == "best"
    window.quality_combo.setCurrentIndex(window.quality_combo.findData("720"))
    assert settings.get("video_quality") == "720"
    window.close()


def test_quality_combo_is_disabled_for_a_local_file(qtbot, translator, settings, db, tmp_path):
    window = _window(qtbot, translator, settings, db, tmp_path)
    media = tmp_path / "ep.mp4"
    media.write_bytes(b"x")

    window.input_edit.setText(str(media))
    assert not window.quality_combo.isEnabled()
    assert window.quality_combo.toolTip() == translator.t("quality.local_tip")

    window.input_edit.setText("https://www.youtube.com/watch?v=x")
    assert window.quality_combo.isEnabled()
    window.close()


def test_low_offered_qualities_are_reported_not_hidden(qtbot, translator, settings, db, tmp_path):
    window = _window(qtbot, translator, settings, db, tmp_path)

    window._on_qualities_probed([360])

    assert "360p" in window.quality_combo.itemText(1)
    assert translator.t("quality.low_offered", height=360) in window.log_view.toPlainText()
    window.close()


def test_a_saved_height_survives_until_the_link_is_probed(qtbot, translator, settings, db, tmp_path):
    settings.set("video_quality", "1080")
    window = _window(qtbot, translator, settings, db, tmp_path)

    assert window.quality_combo.currentData() == "1080"

    window._on_qualities_probed([720, 360])          # this link does not offer 1080
    assert window.quality_combo.currentData() == "best"
    window.close()

