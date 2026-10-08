"""Subtitle provider interface and evidence records."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Protocol

from app.core.exporter import Cue
from app.core.metadata import SeriesInfo

log = logging.getLogger(__name__)

# Evidence kinds. Only "manual" can supply final target-language text (after verification, DECISIONS D-018).
MANUAL = "manual"
AUTO = "auto"                      # automatic speech recognition by the platform
AUTO_TRANSLATED = "auto_translated"
COMMUNITY = "community"            # subtitle sites: human-made, unknown sync/release


class ProviderError(RuntimeError):
    def __init__(self, message: str, ui_key: str = "provider.failed", retryable: bool = False):
        super().__init__(message)
        self.ui_key = ui_key
        self.retryable = retryable


class ProviderUnavailable(ProviderError):
    """Provider not configured (e.g. missing API key); skipped without being an error."""


@dataclass
class SubtitleRequest:
    source_language: str             # may be "auto" before detection
    target_language: str
    series: SeriesInfo
    title: str | None = None
    media_info: dict | None = None   # yt-dlp info for URL inputs
    url: str | None = None
    work_dir: str | None = None


@dataclass
class SubtitleEvidence:
    provider: str
    language: str
    kind: str
    cues: list[Cue]
    reference: str                   # where it came from (track name, file id, release name)
    machine_generated: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cues"] = [asdict(c) for c in self.cues]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SubtitleEvidence":
        known = {f for f in cls.__dataclass_fields__}
        fields = {k: v for k, v in data.items() if k in known}
        return cls(**{**fields, "cues": [Cue(**c) for c in data["cues"]]})


class SubtitleProvider(Protocol):
    name: str

    def fetch(self, request: SubtitleRequest) -> list[SubtitleEvidence]:
        ...


def collect_evidence(providers: list[SubtitleProvider], request: SubtitleRequest) -> tuple[list[SubtitleEvidence], list[str]]:
    """Run every provider; a failing provider is logged and skipped, never fatal."""
    evidence: list[SubtitleEvidence] = []
    warnings: list[str] = []
    for provider in providers:
        try:
            found = provider.fetch(request)
            log.info("Provider %s returned %d subtitle track(s)", provider.name, len(found))
            evidence.extend(found)
        except ProviderUnavailable as exc:
            log.info("Provider %s skipped: %s", provider.name, exc)
        except Exception as exc:  # Failure isolation: optional sources never abort a job.
            message = f"Subtitle provider {provider.name} failed: {exc}"
            log.warning(message)
            warnings.append(message)
    return evidence, warnings
