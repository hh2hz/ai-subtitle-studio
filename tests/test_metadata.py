from pathlib import Path

import pytest

from app.core.metadata import SeriesInfo, detect, parse_title

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("title,series,season,episode", [
    ("Kurtlar.Vadisi.S01E05.720p", "Kurtlar Vadisi", 1, 5),
    ("Show Name 2x07 - The Return", "Show Name", 2, 7),
    ("Show - Season 3 Episode 12", "Show", 3, 12),
    ("Dizi Adi 2. Sezon 14. B\u00f6l\u00fcm", "Dizi Adi", 2, 14),
    ("Dizi Adi 45. Bolum", "Dizi Adi", None, 45),
    ("Dizi Adi | B\u00f6l\u00fcm 3", "Dizi Adi", None, 3),
    ("My Show Ep.9 (2021)", "My Show", None, 9),
])
def test_parse_title(title, series, season, episode):
    info = parse_title(title, "title")
    assert (info.series_name, info.season, info.episode) == (series, season, episode)


def test_episode_title_and_year():
    info = parse_title("Show S02E03 - The Heist (2019)", "title")
    assert info.episode_title == "The Heist (2019)"
    assert info.year == 2019


def test_arabic_title_fixture():
    for line in (FIXTURES / "arabic_titles.txt").read_text(encoding="utf-8").splitlines():
        title, season, episode = line.split("|")
        info = parse_title(title, "title")
        assert info.episode == int(episode)
        assert info.season == (int(season) if season else None)


def test_no_match():
    assert parse_title("Random vlog about cats", "title").episode is None


def test_detect_prefers_structured_metadata_and_user_override():
    info = detect(title="Show S01E02", ytdlp_info={"series": "Real Show", "season_number": 1, "episode_number": 3})
    assert (info.series_name, info.episode, info.source) == ("Real Show", 3, "metadata")
    corrected = info.merged_with(SeriesInfo(episode=4, source="user"))
    assert corrected.episode == 4 and corrected.series_name == "Real Show"
