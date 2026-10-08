import io
import json
import zipfile

import pytest

from app.core.metadata import SeriesInfo
from app.providers.base import (
    AUTO, AUTO_TRANSLATED, COMMUNITY, MANUAL, ProviderError, ProviderUnavailable, SubtitleRequest, collect_evidence,
)
from app.providers.opensubtitles import OpenSubtitlesProvider
from app.providers.subdl import SubdlProvider
from app.providers.youtube import PlatformSubtitleProvider, select_tracks
from tests.fakes import FakeDownloader, vtt_from

SRT = b"1\r\n00:00:01,000 --> 00:00:02,000\r\nMerhaba\r\n"


def _request(**kw):
    defaults = dict(source_language="tr", target_language="ar", series=SeriesInfo("Dizi", 1, 2), title="Dizi")
    defaults.update(kw)
    return SubtitleRequest(**defaults)


# -- platform (YouTube) ---------------------------------------------------------------------------

def test_select_tracks_prefers_manual_then_orig_auto():
    info = {"subtitles": {"tr-TR": [], "en": [], "ar": []},
            "automatic_captions": {"en-orig": [], "en": [], "ar": [], "tr": []}}
    assert select_tracks(info, "tr", "ar") == [("tr-TR", False, MANUAL), ("ar", False, MANUAL),
                                               ("en", False, MANUAL), ("en-orig", True, AUTO)]


def test_select_tracks_auto_only():
    info = {"subtitles": {}, "automatic_captions": {"tr": [], "ar": []}}
    assert select_tracks(info, "tr", "ar") == [("tr", True, AUTO)]      # no platform auto-translation
    assert select_tracks({"subtitles": {}, "automatic_captions": {}}, "tr", "ar") == []


def _platform_request(tmp_path, info):
    return _request(url="https://example.com/v", media_info=info, work_dir=str(tmp_path))


def test_platform_provider_downloads_and_parses(tmp_path, media=None):
    info = {"subtitles": {"tr": []}, "automatic_captions": {"en-orig": []}}
    dl = FakeDownloader(None, subtitle_texts={
        ("tr", False): vtt_from([(1, 2, "Merhaba")]),
        ("en-orig", True): vtt_from([(1, 2, "hello"), (2, 2.01, "hello")]),
    })
    evidence = PlatformSubtitleProvider(dl, sleep=lambda s: None).fetch(_platform_request(tmp_path, info))
    assert [(e.language, e.kind, e.machine_generated) for e in evidence] == [("tr", MANUAL, False), ("en", AUTO, True)]
    assert evidence[1].cues[0].text == "hello" and len(evidence[1].cues) == 1


def test_platform_provider_stops_after_rate_limit(tmp_path):
    info = {"subtitles": {"tr": [], "ar": [], "en": []}, "automatic_captions": {}}
    dl = FakeDownloader(None, subtitle_texts={("tr", False): vtt_from([(1, 2, "a")])},
                        fail_subtitles={"ar": ProviderError("HTTP Error 429", "error.rate_limited", retryable=True)})
    evidence = PlatformSubtitleProvider(dl, sleep=lambda s: None).fetch(_platform_request(tmp_path, info))
    assert [e.language for e in evidence] == ["tr"]
    assert dl.subtitle_requests == [("tr", False), ("ar", False)]     # "en" not attempted after 429


def test_platform_provider_all_failed_raises(tmp_path):
    info = {"subtitles": {"tr": []}, "automatic_captions": {}}
    dl = FakeDownloader(None, fail_subtitles={"tr": ProviderError("boom")})
    with pytest.raises(ProviderError):
        PlatformSubtitleProvider(dl, sleep=lambda s: None).fetch(_platform_request(tmp_path, info))


def test_platform_provider_not_for_files():
    with pytest.raises(ProviderUnavailable):
        PlatformSubtitleProvider(None).fetch(_request())


# -- OpenSubtitles / SubDL with a fake HTTP transport ----------------------------------------------

class FakeHttp:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def _answer(self, kind, url, payload):
        self.calls.append((kind, url, payload))
        for prefix, value in self.responses.items():
            if url.startswith(prefix):
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unexpected request {url}")

    def get_json(self, url, params=None, headers=None):
        return self._answer("get_json", url, params)

    def post_json(self, url, body, headers=None):
        return self._answer("post_json", url, body)

    def get_bytes(self, url, headers=None):
        return self._answer("get_bytes", url, None)


def test_opensubtitles_requires_key():
    with pytest.raises(ProviderUnavailable):
        OpenSubtitlesProvider("").fetch(_request())


def test_opensubtitles_search_and_download():
    search = {"data": [
        {"attributes": {"language": "tr", "download_count": 5, "files": [{"file_id": 1}], "release": "a"}},
        {"attributes": {"language": "tr", "download_count": 50, "files": [{"file_id": 2}], "release": "b"}},
        {"attributes": {"language": "ar", "download_count": 9, "files": [{"file_id": 3}], "machine_translated": True}},
        {"attributes": {"language": "ar", "download_count": 99, "files": []}},
    ]}
    http = FakeHttp({"https://api.opensubtitles.com/api/v1/subtitles": search,
                     "https://api.opensubtitles.com/api/v1/download": {"link": "https://dl.example/x.srt"},
                     "https://dl.example/": SRT})
    evidence = OpenSubtitlesProvider("key", http=http).fetch(_request())
    assert [(e.language, e.kind, e.machine_generated) for e in evidence] == [("tr", COMMUNITY, False), ("ar", COMMUNITY, True)]
    downloads = [c[2]["file_id"] for c in http.calls if c[0] == "post_json"]
    assert downloads == [2, 3]
    params = http.calls[0][2]
    assert params["season_number"] == 1 and params["episode_number"] == 2 and params["languages"] == "ar,tr"


def test_subdl_search_and_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "x")
        z.writestr("ep.srt", SRT)
    http = FakeHttp({"https://api.subdl.com/": {"status": True, "subtitles": [
                        {"url": "/subtitle/1.zip", "lang": "TR", "release_name": "r1"},
                        {"url": "/subtitle/2.zip", "lang": "TR", "release_name": "r2"}]},
                     "https://dl.subdl.com/subtitle/": buf.getvalue()})
    evidence = SubdlProvider("key", http=http).fetch(_request())
    assert len(evidence) == 1 and evidence[0].cues[0].text == "Merhaba"


def test_subdl_reports_api_error():
    http = FakeHttp({"https://api.subdl.com/": {"status": False, "error": "invalid api key"}})
    with pytest.raises(ProviderError, match="invalid api key"):
        SubdlProvider("bad", http=http).fetch(_request())


# -- failure isolation ----------------------------------------------------------------------------

class _Boom:
    name = "boom"

    def fetch(self, request):
        raise RuntimeError("unexpected crash")


class _Skip:
    name = "skip"

    def fetch(self, request):
        raise ProviderUnavailable("no key")


def test_collect_evidence_isolates_failures():
    http = FakeHttp({"https://api.opensubtitles.com/": ProviderError("HTTP 429", "error.rate_limited")})
    providers = [_Boom(), _Skip(), OpenSubtitlesProvider("key", http=http)]
    evidence, warnings = collect_evidence(providers, _request())
    assert evidence == []
    assert len(warnings) == 2 and any("unexpected crash" in w for w in warnings) and any("429" in w for w in warnings)
