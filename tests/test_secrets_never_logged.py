"""A4 (D-106): a fake API key and a fake cookie must reach no log, report, warning or database.

The app never reads cookie values (cookiesfrombrowser carries a browser name, cookiefile a path), so the cookie
half locks that design in; the key half exercises the real HTTP error path used by the subtitle providers.
"""

import json
import logging
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from app.core.modes import Mode
from app.core.pipeline import JobConfig, Pipeline
from app.providers.base import ProviderError
from app.providers.http import HttpClient
from app.providers.subdl import SEARCH_URL, SubdlProvider
from tests.fakes import CPU_ASR, CPU_MT, Factory, FakeAsrEngine, FakeTranslator
from tests.media import make_tone_file

FAKE_KEY = "sk-FAKEKEY-0123456789abcdef-DO-NOT-LOG"
FAKE_COOKIE_VALUE = "FAKECOOKIE-0123456789abcdef-DO-NOT-LOG"
SECRETS = (FAKE_KEY, FAKE_COOKIE_VALUE)


@pytest.fixture
def media(tmp_path):
    return make_tone_file(tmp_path / "Episode 1.m4a", seconds=12.0)


def test_auth_error_keeps_the_host_and_drops_the_api_key(monkeypatch, caplog):
    """SubDL puts api_key in the query string; the error message must not carry it."""
    seen = []

    def boom(url, *args, **kwargs):
        seen.append(url.full_url if hasattr(url, "full_url") else str(url))
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ProviderError) as excinfo:
            HttpClient().get_json(SEARCH_URL, {"api_key": FAKE_KEY, "film_name": "Episode"})

    assert FAKE_KEY in seen[0]                      # the key really was in the request URL
    assert FAKE_KEY not in str(excinfo.value)       # ... and is not in the message we show
    assert FAKE_KEY not in caplog.text
    assert "access denied" in str(excinfo.value)    # the user gets the key hint, not the raw URL


def test_server_error_keeps_the_host_and_drops_the_api_key(monkeypatch, caplog):
    """The 5xx branch does name the host, so it is the branch that could leak a query string."""
    seen = []

    def boom(url, *args, **kwargs):
        seen.append(url.full_url if hasattr(url, "full_url") else str(url))
        raise urllib.error.HTTPError(url, 500, "Server Error", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ProviderError) as excinfo:
            HttpClient().get_json(SEARCH_URL, {"api_key": FAKE_KEY, "film_name": "Episode"})

    assert FAKE_KEY in seen[0]
    assert FAKE_KEY not in str(excinfo.value) and FAKE_KEY not in caplog.text
    assert "api.subdl.com" in str(excinfo.value)    # the useful part survives
    assert "?" not in str(excinfo.value)            # and no query string at all


def test_network_error_message_has_no_key(monkeypatch, caplog):
    def boom(url, *args, **kwargs):
        raise urllib.error.URLError("[Errno 11001] getaddrinfo failed")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ProviderError) as excinfo:
            HttpClient().get_json(SEARCH_URL, {"api_key": FAKE_KEY, "film_name": "Episode"})

    assert FAKE_KEY not in str(excinfo.value) and FAKE_KEY not in caplog.text


def test_cookie_file_content_never_enters_the_app(tmp_path):
    """The cookie setting passes a path to yt-dlp; the app itself must never look inside the file."""
    from app.core.downloader import access_options

    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text(
        "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSESSION\t" + FAKE_COOKIE_VALUE + "\n",
        encoding="utf-8")

    options = access_options({"cookies_source": "file", "cookies_file": str(cookie_file)})
    assert options == {"cookiefile": str(cookie_file)}
    assert FAKE_COOKIE_VALUE not in json.dumps(options)
    browser = access_options({"cookies_source": "browser", "cookies_browser": "firefox"})
    assert browser == {"cookiesfrombrowser": ("firefox", None, None, None)}


def _scan(root: Path) -> dict[str, list[str]]:
    """Which files under `root` contain which secret."""
    found: dict[str, list[str]] = {secret: [] for secret in SECRETS}
    for path in [root] if root.is_file() else list(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        text = blob.decode("utf-8", errors="ignore")
        for secret in SECRETS:
            if secret in text:
                found[secret].append(str(path))
    return found


def test_full_job_writes_no_secret_to_any_artifact(tmp_path, media, monkeypatch, caplog):
    """Run a real (fake-engine) job with a keyed provider that fails, then scan everything it wrote."""
    original = urllib.request.urlopen

    def maybe_boom(url, *args, **kwargs):
        target = url.full_url if hasattr(url, "full_url") else str(url)
        if "subdl" in target:
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
        return original(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", maybe_boom)

    config = JobConfig(media, "tr", "ar", Mode.BALANCED, output_dir=tmp_path / "out")
    pipe = Pipeline(config, tmp_path / "jobs", [CPU_ASR], [CPU_MT],
                    Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine()}),
                    Factory({(CPU_MT.model, "cpu"): FakeTranslator()}),
                    providers=[SubdlProvider(FAKE_KEY)])
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text(f"SESSION\t{FAKE_COOKIE_VALUE}\n", encoding="utf-8")
    from app.core.downloader import access_options
    json.dumps(access_options({"cookies_source": "file", "cookies_file": str(cookie_file)}))

    with caplog.at_level(logging.DEBUG):
        result = pipe.run()

    # The job really ran and really analysed the failing provider, so the scan below is not vacuous.
    assert "subdl" in " ".join(result.warnings)
    assert any("subdl" in record.getMessage() for record in caplog.records)
    work = result.output_dir / "work"
    assert (work / "ProcessingLog.txt").stat().st_size > 0
    assert (pipe.job_dir / "job.log").stat().st_size > 0

    for secret, hits in _scan(pipe.job_dir).items():
        assert hits == [], (secret, hits)
    for secret, hits in _scan(result.output_dir).items():
        assert hits == [], (secret, hits)
    assert FAKE_KEY not in caplog.text and FAKE_COOKIE_VALUE not in caplog.text
    assert FAKE_KEY not in " ".join(result.warnings)
    assert FAKE_KEY not in str(result.stats)


def test_job_extras_never_store_api_keys(tmp_path):
    """Queued jobs must not carry `*_api_key` into the database."""
    from app.database.database import Database
    from app.database.jobs import JobsRepo

    db = Database(tmp_path / "studio.db")
    repo = JobsRepo(db)
    job_id = repo.enqueue(input_type="url", input_value="https://example.invalid/v", source_language="tr",
                          target_language="ar", mode="balanced",
                          extras={"subdl_api_key": FAKE_KEY, "video_quality": "720"})
    queued = repo.get_queue()
    entry = next(item for item in queued if item["id"] == job_id)
    stored = json.dumps(entry)
    assert FAKE_KEY not in stored
    assert "video_quality" in stored
    assert FAKE_KEY.encode() not in (tmp_path / "studio.db").read_bytes()
