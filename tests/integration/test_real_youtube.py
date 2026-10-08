"""Real yt-dlp run against a URL (network). Set AISS_TEST_URL to a video you have the rights to process.

    set AISS_TEST_URL=https://www.youtube.com/watch?v=...
    python -m pytest tests/integration/test_real_youtube.py -m integration -s
"""

import os

import pytest

from app.core.downloader import YtDlpAdapter, js_runtime_available
from app.core.metadata import detect
from app.providers.youtube import select_tracks

pytestmark = pytest.mark.integration
URL = os.environ.get("AISS_TEST_URL")


@pytest.mark.skipif(not URL, reason="set AISS_TEST_URL")
def test_probe_and_download_audio(tmp_path):
    adapter = YtDlpAdapter()
    print("\nyt-dlp", adapter.version(), "| JS runtime:", js_runtime_available())
    info = adapter.probe(URL)
    print("title:", info.get("title"), "| duration:", info.get("duration"))
    print("series:", detect(title=info.get("title"), ytdlp_info=info).to_dict())
    print("manual subtitles:", sorted(info.get("subtitles") or {}))
    print("auto originals:", [c for c in (info.get("automatic_captions") or {}) if c.endswith("-orig")])
    print("selected tracks tr->ar:", select_tracks(info, "tr", "ar"))
    path = adapter.download_audio(URL, tmp_path)
    assert path.is_file() and path.stat().st_size > 0
