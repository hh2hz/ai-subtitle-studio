import sys
import types

import pytest

from app.core.downloader import YtDlpAdapter
from app.core.errors import PipelineError


def test_retry_on_rate_limit_then_success():
    sleeps = []
    calls = {"n": 0}

    def action():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("HTTP Error 429: Too Many Requests")
        return "ok"

    adapter = YtDlpAdapter(retry_delays=(1, 2), sleep=sleeps.append)
    assert adapter._with_retry(action) == "ok"
    assert sleeps == [1, 2]


def test_rate_limit_gives_up_with_key():
    adapter = YtDlpAdapter(retry_delays=(0,), sleep=lambda s: None)
    with pytest.raises(PipelineError) as info:
        adapter._with_retry(lambda: (_ for _ in ()).throw(RuntimeError("HTTP Error 429")))
    assert info.value.ui_key == "error.rate_limited"


@pytest.mark.parametrize("message,key", [
    ("ERROR: Unsupported URL: https://x", "error.url_unsupported"),
    ("ERROR: [youtube] abc: Video unavailable", "error.video_unavailable"),
    ("Sign in to confirm you're not a bot", "error.youtube_bot_check"),
    ("Something else", "error.download_failed"),
])
def test_error_classification_without_retry(message, key):
    sleeps = []
    adapter = YtDlpAdapter(retry_delays=(1,), sleep=sleeps.append)
    with pytest.raises(PipelineError) as info:
        adapter._with_retry(lambda: (_ for _ in ()).throw(RuntimeError(message)))
    assert info.value.ui_key == key and sleeps == []


def test_probe_rejects_playlists(monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            return {"_type": "playlist"}

        def sanitize_info(self, info):
            return info

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    with pytest.raises(PipelineError) as info:
        YtDlpAdapter(retry_delays=()).probe("https://youtube.com/playlist?list=x")
    assert info.value.ui_key == "error.url_unsupported"


def test_download_video_picks_merged_mp4(monkeypatch, tmp_path):
    base = tmp_path / "Show [4K] 61. Bolum"

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            assert self.opts["merge_output_format"] == "mp4"
            (tmp_path / "Show [4K] 61. Bolum.f137.mp4").write_bytes(b"x" * 10)     # intermediate
            (tmp_path / "Show [4K] 61. Bolum.mp4").write_bytes(b"x" * 5)
            (tmp_path / "Other.mp4").write_bytes(b"x" * 50)
            return {}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    path = YtDlpAdapter(retry_delays=()).download_video("https://x", base)
    assert path.name == "Show [4K] 61. Bolum.mp4"
