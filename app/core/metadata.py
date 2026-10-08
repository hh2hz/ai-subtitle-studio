"""Series / season / episode detection from titles, descriptions and file names."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_AR_EPISODE = "\u0627\u0644\u062d\u0644\u0642\u0629"   # Arabic word for "episode"
_AR_SEASON = "\u0627\u0644\u0645\u0648\u0633\u0645"    # Arabic word for "season"

_PATTERNS = [
    # S01E02, s1e2, S01 E02
    re.compile(r"\bS(?P<season>\d{1,2})\s*[ ._-]?\s*E(?P<episode>\d{1,4})\b", re.I),
    # 1x02
    re.compile(r"\b(?P<season>\d{1,2})x(?P<episode>\d{1,4})\b", re.I),
    # Season 1 Episode 2
    re.compile(r"\bSeason\s*(?P<season>\d{1,2}).{0,10}?\bEpisode\s*(?P<episode>\d{1,4})\b", re.I),
    # Turkish: "1. Sezon 2. Bolum", "2. Bolum", "Bolum 2" (with or without Turkish letters)
    re.compile(r"\b(?P<season>\d{1,2})\s*\.?\s*Sezon\s*(?P<episode>\d{1,4})\s*\.?\s*B[o\u00f6]l[u\u00fc]m", re.I),
    re.compile(r"\b(?P<episode>\d{1,4})\s*\.?\s*B[o\u00f6]l[u\u00fc]m\b", re.I),
    re.compile(r"\bB[o\u00f6]l[u\u00fc]m\s*(?P<episode>\d{1,4})\b", re.I),
    # Arabic: "<season word> 1 <episode word> 2" or "<episode word> 2"
    re.compile(rf"{_AR_SEASON}\s*(?P<season>\d{{1,2}}).{{0,10}}?{_AR_EPISODE}\s*(?P<episode>\d{{1,4}})"),
    re.compile(rf"{_AR_EPISODE}\s*(?P<episode>\d{{1,4}})"),
    # Episode 12, Ep 12, Ep.12
    re.compile(r"\b(?:Episode|Ep\.?)\s*(?P<episode>\d{1,4})\b", re.I),
]
_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")
_SEPARATORS = re.compile(r"\s*[|\u2013\u2014-]\s*")


@dataclass
class SeriesInfo:
    series_name: str | None = None
    season: int | None = None
    episode: int | None = None
    episode_title: str | None = None
    year: int | None = None
    source: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def merged_with(self, override: "SeriesInfo") -> "SeriesInfo":
        """Fields set in override (user corrections) win."""
        data = self.to_dict()
        data.update({k: v for k, v in override.to_dict().items() if v not in (None, "")})
        return SeriesInfo(**data)


def parse_title(text: str, source: str) -> SeriesInfo:
    info = SeriesInfo(source=source)
    if not text:
        return info
    cleaned = re.sub(r"[._]+", " ", text).strip()
    for pattern in _PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        groups = match.groupdict()
        info.episode = int(groups["episode"])
        if groups.get("season"):
            info.season = int(groups["season"])
        name = _SEPARATORS.sub(" ", cleaned[: match.start()]).strip(" -|:([")
        if name:
            info.series_name = name
        rest = cleaned[match.end():].strip(" -|:")
        if rest:
            info.episode_title = _SEPARATORS.split(rest)[0].strip() or None
        break
    year = _YEAR.search(cleaned)
    if year:
        info.year = int(year.group(1))
    return info


def detect(title: str | None = None, filename: str | None = None, ytdlp_info: dict | None = None) -> SeriesInfo:
    """Combine structured yt-dlp fields (most reliable), then the title, then the file name."""
    result = SeriesInfo()
    for candidate in (parse_title(filename or "", "filename"), parse_title(title or "", "title")):
        if candidate.episode is not None:
            result = candidate
    if ytdlp_info:
        structured = SeriesInfo(
            series_name=ytdlp_info.get("series"),
            season=ytdlp_info.get("season_number"),
            episode=ytdlp_info.get("episode_number"),
            episode_title=ytdlp_info.get("episode"),
            year=ytdlp_info.get("release_year"),
            source="metadata",
        )
        if structured.episode is not None or structured.series_name:
            result = result.merged_with(structured)
    return result
