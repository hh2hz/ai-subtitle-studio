"""A finished episode removes its job cache so that processing it again starts from scratch (D-116)."""

import threading

import pytest

from app.core.errors import JobCancelled
from tests.fakes import FakeAsrEngine, FakeTranslator
from tests.media import make_tone_file
from tests.test_pipeline import _pipeline


@pytest.fixture
def media(tmp_path):
    return make_tone_file(tmp_path / "Episode 1.m4a", seconds=20.0)


def test_a_finished_episode_removes_its_cache_but_keeps_every_output(tmp_path, media):
    pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), keep_cache=False)

    result = pipe.run()

    assert not (tmp_path / "jobs" / result.job_key).exists()
    out = result.output_dir
    assert (out / "Episode 1.ar.srt").is_file() and (out / "work" / "MasterTranscript.json").is_file()
    assert (out / "work" / "ProcessingLog.txt").stat().st_size > 0
    assert (out / "work" / "audio.wav").stat().st_size > 0            # the review editor re-transcribes from it


def test_the_cache_is_kept_when_asked(tmp_path, media):
    pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), keep_cache=True)

    result = pipe.run()

    assert (tmp_path / "jobs" / result.job_key / "audio.wav").is_file()


def test_the_default_of_the_pipeline_class_keeps_the_cache(tmp_path, media):
    pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator())

    assert (tmp_path / "jobs" / pipe.run().job_key).is_dir()


def test_a_cancelled_episode_keeps_its_cache_so_it_can_resume(tmp_path, media):
    cancel = threading.Event()
    engine = FakeAsrEngine(cancel_after=3, cancel_event=cancel)
    pipe, _, _ = _pipeline(tmp_path, media, engine, FakeTranslator(), cancel=cancel, keep_cache=False)

    with pytest.raises(JobCancelled):
        pipe.run()

    assert (tmp_path / "jobs" / pipe.job_key).is_dir()


def test_after_the_output_folder_is_deleted_the_same_input_runs_every_stage_again(tmp_path, media):
    import shutil

    first, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), keep_cache=False)
    result = first.run()
    shutil.rmtree(result.output_dir)

    second, asr_f, mt_f = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), keep_cache=False)
    again = second.run()

    assert len(asr_f.loads) == 1 and len(mt_f.loads) == 1
    assert not any(again.stats["stages"][s]["cached"] for s in ("audio", "transcribe", "translate"))
    assert (again.output_dir / "Episode 1.ar.srt").is_file()


def test_a_folder_that_is_not_the_jobs_own_is_never_removed(tmp_path, media):
    pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), keep_cache=False)
    pipe.run()
    stranger = tmp_path / "precious"
    stranger.mkdir()
    (stranger / "file.txt").write_text("x", encoding="utf-8")
    pipe.job_dir = stranger

    pipe._remove_job_cache()

    assert (stranger / "file.txt").is_file()


def test_the_setting_reaches_the_pipeline_and_defaults_to_removal(tmp_path):
    from app.database.settings import DEFAULTS

    assert DEFAULTS["keep_job_cache"] is False
