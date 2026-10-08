"""M2 acceptance: URL input and subtitle sources; failures of optional sources never stop the job."""

import pytest

from app.core.errors import PipelineError
from app.core.exporter import read_srt
from app.core.metadata import SeriesInfo
from app.core.modes import Mode
from app.core.pipeline import JobConfig, Pipeline
from app.providers.base import ProviderError
from app.providers.youtube import PlatformSubtitleProvider
from app.utils.atomic import read_json
from tests.fakes import CPU_ASR, CPU_MT, Factory, FakeAsrEngine, FakeDownloader, FakeTranslator, matching_cues, vtt_from
from tests.media import make_tone_file

URL = "https://www.youtube.com/watch?v=test"


@pytest.fixture
def media(tmp_path):
    return make_tone_file(tmp_path / "src.m4a", seconds=20.0)


def _run(tmp_path, downloader, providers=None, override=None):
    config = JobConfig(None, "tr", "ar", Mode.BALANCED, output_dir=tmp_path / "out", source_url=URL,
                       series_override=override)
    pipe = Pipeline(config, tmp_path / "jobs", [CPU_ASR], [CPU_MT],
                    Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine()}),
                    Factory({(CPU_MT.model, "cpu"): FakeTranslator()}),
                    downloader=downloader,
                    providers=providers if providers is not None else [PlatformSubtitleProvider(downloader, sleep=lambda s: None)])
    return pipe.run()


def test_url_without_any_subtitles(tmp_path, media):
    result = _run(tmp_path, FakeDownloader(media))
    assert read_srt(result.outputs["target_srt"])
    assert result.stats["subtitle_sources"] == []
    assert result.series["series_name"] == "Show" and result.series["episode"] == 2
    assert result.output_dir.name == "Show S01E02 - Pilot"


def test_rate_limited_and_garbage_sources_do_not_stop_the_job(tmp_path, media):
    info = {"id": "x", "title": "Clip", "duration": 20, "subtitles": {"tr": [], "ar": []},
            "automatic_captions": {"en-orig": []}}
    dl = FakeDownloader(media, info=info, subtitle_texts={
        ("tr", False): vtt_from(matching_cues()),
        ("en-orig", True): vtt_from([(0, 20, "forh. Uhhuh. helicopter")]),
    }, fail_subtitles={"ar": ProviderError("HTTP Error 429", "error.rate_limited", retryable=False)})
    result = _run(tmp_path, dl)
    sources = {(s["language"], s["kind"]): s for s in result.stats["subtitle_sources"]}
    assert sources[("tr", "manual")]["status"] == "matches_audio"
    assert sources[("tr", "manual")]["audio_agreement"] == 1.0
    # The platform's ASR track is labelled English although the speech is Turkish (seen on the real test video).
    assert sources[("en", "auto")]["status"] == "rejected"
    assert any("transcribed as en, speech is tr" in w for w in result.warnings)
    assert read_srt(result.outputs["target_srt"])
    assert any(k.startswith("evidence_") for k in result.outputs)


def test_same_language_garbage_is_rejected(tmp_path, media):
    info = {"id": "x", "title": "Clip", "subtitles": {}, "automatic_captions": {"tr": []}}
    dl = FakeDownloader(media, info=info, subtitle_texts={("tr", True): vtt_from([(0, 20, "lorem ipsum dolor")])})
    result = _run(tmp_path, dl)
    assert result.stats["subtitle_sources"][0]["status"] == "rejected"
    assert any("Ignored youtube auto subtitles" in w for w in result.warnings)


def test_provider_crash_is_isolated(tmp_path, media):
    class Crashing:
        name = "crashing"

        def fetch(self, request):
            raise ConnectionError("network down")

    result = _run(tmp_path, FakeDownloader(media), providers=[Crashing()])
    assert any("network down" in w for w in result.warnings)
    assert read_srt(result.outputs["target_srt"])


def test_audio_download_failure_is_reported(tmp_path, media):
    dl = FakeDownloader(media, fail_audio=PipelineError("Video unavailable", "error.video_unavailable"))
    with pytest.raises(PipelineError) as info:
        _run(tmp_path, dl)
    assert info.value.ui_key == "error.video_unavailable"


def test_download_is_cached_and_override_applies(tmp_path, media):
    dl = FakeDownloader(media)
    _run(tmp_path, dl)
    result = _run(tmp_path, dl, override=SeriesInfo(series_name="Correct Name", episode=5))
    assert dl.audio_downloads == 1
    assert result.stats["stages"]["download"]["cached"]
    assert result.series["series_name"] == "Correct Name" and result.series["episode"] == 5
    assert result.series["season"] == 1     # detected value kept where the user gave none


def test_file_input_skips_download(tmp_path, media):
    config = JobConfig(media, "tr", "ar", Mode.FAST, output_dir=tmp_path / "out")
    result = Pipeline(config, tmp_path / "jobs", [CPU_ASR], [CPU_MT],
                      Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine()}),
                      Factory({(CPU_MT.model, "cpu"): FakeTranslator()})).run()
    assert result.stats["stages"]["download"] == {"skipped": "local file"}
    assert result.stats["stages"]["subtitles"] == {"skipped": "no providers"}


def test_config_requires_exactly_one_input(tmp_path):
    with pytest.raises(ValueError):
        Pipeline(JobConfig(None, "tr", "ar", Mode.FAST), tmp_path, [CPU_ASR], [CPU_MT], None, None)
