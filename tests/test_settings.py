import pytest

from app.database.database import Database
from app.database.settings import DEFAULTS, Settings


def test_defaults(settings):
    assert settings.all() == DEFAULTS


def test_round_trip_across_reopen(tmp_path):
    path = tmp_path / "s.db"
    values = {
        "ui_language": "ar", "theme": "dark", "gpu_offer_shown": True, "source_language": "tr", "target_language": "ar",
        "mode": "maximum_accuracy", "output_dir": "C:\\Users\\x\\Subs", "model_dir": "D:/models",
        "log_level": "DEBUG", "first_run_completed": True,
        "fetch_platform_subtitles": False, "opensubtitles_api_key": "k1", "subdl_api_key": "k2",
        "llm_refine": True, "llm_correct_only": True, "llm_review": True, "burn_video": True,
        "translation_engine": "madlad", "local_model": "gemma3:4b", "audio_enhance": "on",
        "name_normalization": False, "diarization": False, "snap_to_shots": True, "parallel_jobs": 2,
        "cookies_source": "browser", "cookies_browser": "firefox",
        "cookies_file": "C:\\Users\\x\\cookies.txt", "force_ipv4": True,
        "video_quality": "720", "window_geometry": "01d9d0cb0003", "window_geometry_version": 2,
        "dependency_check": False, "keep_job_cache": True, "dependency_check_last": 1.0,
        "update_check": False, "update_skipped": "1.2.0",
        "api_keys_prompted": True,
    }
    assert set(values) == set(DEFAULTS)
    with Database(path) as db:
        s = Settings(db)
        for k, v in values.items():
            s.set(k, v)
    with Database(path) as db:
        assert Settings(db).all() == values


def test_overwrite_and_reset(settings):
    settings.set("mode", "fast")
    settings.set("mode", "balanced")
    assert settings.get("mode") == "balanced"
    settings.reset("mode")
    assert settings.get("mode") == DEFAULTS["mode"]


@pytest.mark.parametrize("key,value,exc", [
    ("nonexistent", 1, KeyError),
    ("first_run_completed", 1, TypeError),      # int is not bool
    ("output_dir", None, TypeError),
    ("mode", "turbo", ValueError),
    ("target_language", "auto", ValueError),    # auto only valid for source
    ("source_language", "xx", ValueError),
    ("log_level", "TRACE", ValueError),
    ("cookies_source", "chrome", ValueError),    # a source, not a browser name
    ("cookies_browser", "netscape", ValueError),
    ("cookies_file", "cookies.json", ValueError),  # must be a cookies.txt
    ("video_quality", "4k", ValueError),           # "best" or a height in pixels
    ("video_quality", "80", ValueError),           # below any real height
])
def test_rejects_invalid(settings, key, value, exc):
    with pytest.raises(exc):
        settings.set(key, value)


def test_corrupt_stored_value_falls_back(settings, db):
    with db.conn:
        db.conn.execute("INSERT INTO settings (key, value) VALUES ('mode', 'not json')")
    assert settings.get("mode") == DEFAULTS["mode"]
