"""Task 5.5: frame snapping and optional shot-change snapping (D-103)."""

import json
import subprocess

import pytest

from app.core import frames
from app.core.errors import JobCancelled
from app.core.exporter import Cue
from tests.fakes import FakeAsrEngine, FakeTranslator
from tests.test_pipeline import _pipeline, media  # noqa: F401

FPS = 25.0
FRAME = 1 / FPS


def on_frame(seconds, fps=FPS):
    return abs(seconds * fps - round(seconds * fps)) < 1e-6 * fps + 0.002 * fps


def test_starts_and_ends_land_on_frames_with_a_two_frame_gap():
    cues = frames.snap_cues([Cue(1.013, 2.507, "a", 0), Cue(2.52, 4.0, "b", 1), Cue(4.04, 5.0, "c", 2)], FPS)
    assert all(on_frame(c.start) and on_frame(c.end) for c in cues)
    for a, b in zip(cues, cues[1:]):
        assert b.start - a.end >= 2 * FRAME - 0.002 and a.end > a.start
    assert [c.uid for c in cues] == [0, 1, 2]
    assert cues[0].start == pytest.approx(1.0, abs=0.001)


def test_a_cue_is_never_shorter_than_one_frame_and_keeps_its_other_fields():
    cue = frames.snap_cues([Cue(1.0, 1.001, "x", 7, "S1", [3])], FPS)[0]
    assert cue.end - cue.start == pytest.approx(FRAME, abs=0.001)
    assert (cue.uid, cue.speaker, cue.absorbed) == (7, "S1", [3])


def test_the_later_cue_moves_when_the_earlier_one_is_too_short_to_end_sooner():
    cues = frames.snap_cues([Cue(1.0, 1.04, "a"), Cue(1.04, 3.0, "b")], FPS)
    assert cues[1].start - cues[0].end >= 2 * FRAME - 0.002 and cues[0].end > cues[0].start


def test_edges_snap_to_a_shot_change_within_250_ms_only():
    shots = [10.0, 20.0]
    cues = [Cue(9.9, 14.0, "a", 0), Cue(16.0, 20.2, "b", 1), Cue(25.0, 28.0, "c", 2)]
    out = frames.snap_to_shots(cues, shots, FPS)
    assert out[0].start == pytest.approx(10.0, abs=0.001)           # 100 ms before the cut
    assert out[1].end == pytest.approx(20.0, abs=0.001)             # 200 ms after the cut
    assert (out[2].start, out[2].end) == (25.0, 28.0)               # no shot nearby
    assert frames.snap_to_shots(cues, [], FPS) == cues


def test_shot_snapping_respects_the_minimum_duration_and_neighbours():
    out = frames.snap_to_shots([Cue(9.9, 10.5, "a"), Cue(10.6, 13.0, "b")], [10.4], FPS, min_duration=0.833)
    assert out[0].end - out[0].start >= 0.55 and out[0].end <= out[1].start    # not shortened below what it was
    short = frames.snap_to_shots([Cue(9.9, 10.5, "a")], [10.2], FPS, min_duration=0.833)
    assert short[0].start == pytest.approx(9.88, abs=0.03) or short[0].start == pytest.approx(9.9, abs=0.03)


def test_video_fps_is_none_for_audio_and_garbage(tmp_path):
    from tests.media import make_tone_file

    assert frames.video_fps(make_tone_file(tmp_path / "a.m4a", seconds=1.0)) is None
    (tmp_path / "bad.mp4").write_bytes(b"not a video")
    assert frames.video_fps(tmp_path / "bad.mp4") is None
    assert frames.video_fps(tmp_path / "missing.mp4") is None


class _FakeProcess:
    def __init__(self, lines, code=0):
        self.stderr = iter(lines)
        self.returncode = code

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        pass


def test_shot_changes_are_parsed_and_cached_per_video(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 100)
    calls = []

    def popen(cmd, **kw):
        calls.append(cmd)
        return _FakeProcess(["[Parsed_showinfo_1 @ 0x1] n:   0 pts:  1 pts_time:12.48 pos: 1",
                             "frame=1", "[Parsed_showinfo_1 @ 0x1] n: 1 pts: 2 pts_time:30.5 pos: 2"])
    monkeypatch.setattr(frames.subprocess, "Popen", popen)
    cache = tmp_path / "shots.json"
    assert frames.shot_changes(video, cache, ffmpeg="ffmpeg") == [12.48, 30.5]
    assert "select='gt(scene,0.3)'" in calls[0][calls[0].index("-vf") + 1]
    assert frames.shot_changes(video, cache, ffmpeg="ffmpeg") == [12.48, 30.5] and len(calls) == 1
    video.write_bytes(b"y" * 100)                                   # another file: not served from the cache
    frames.shot_changes(video, cache, ffmpeg="ffmpeg")
    assert len(calls) == 2


def test_shot_detection_failure_and_cancel(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(frames.subprocess, "Popen", lambda cmd, **kw: _FakeProcess([], code=1))
    assert frames.shot_changes(video, tmp_path / "s.json", ffmpeg="ffmpeg") == []
    assert not (tmp_path / "s.json").exists()
    monkeypatch.setattr(frames.subprocess, "Popen", lambda cmd, **kw: _FakeProcess(["x"]))
    import threading
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(JobCancelled):
        frames.shot_changes(video, tmp_path / "s2.json", ffmpeg="ffmpeg", cancel=cancel)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert frames.shot_changes(video, tmp_path / "s3.json") == []


def test_pipeline_snaps_to_frames_when_the_video_has_a_frame_rate(tmp_path, media, monkeypatch):  # noqa: F811
    import app.core.pipeline as pipeline

    monkeypatch.setattr(pipeline.frames, "video_fps", lambda path: 25.0)
    result = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator())[0].run()
    doc = json.loads(open(result.outputs["master_transcript"], encoding="utf-8").read())
    assert doc["media"]["fps"] == 25.0 and "shot_changes" not in doc["media"]
    from app.core.exporter import read_srt
    cues = read_srt(result.outputs["target_srt"])
    assert all(abs(c.start * 25 - round(c.start * 25)) < 0.06 for c in cues)


def test_pipeline_uses_shot_changes_only_when_enabled(tmp_path, media, monkeypatch):  # noqa: F811
    import app.core.pipeline as pipeline

    monkeypatch.setattr(pipeline.frames, "video_fps", lambda path: 25.0)
    seen = []
    monkeypatch.setattr(pipeline.frames, "shot_changes", lambda *a, **k: seen.append(1) or [4.0])
    _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator())[0].run()
    assert seen == []
    result = _pipeline(tmp_path / "b", media, FakeAsrEngine(), FakeTranslator(), snap_shots=True)[0].run()
    assert seen == [1] and json.loads(open(result.outputs["master_transcript"], encoding="utf-8").read())[
        "media"]["shot_changes"] == [4.0]


def test_the_review_session_keeps_snapping_after_an_edit(tmp_path, media, monkeypatch):  # noqa: F811
    import app.core.pipeline as pipeline
    from app.core.review import ReviewSession

    monkeypatch.setattr(pipeline.frames, "video_fps", lambda path: 25.0)
    result = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator())[0].run()
    session = ReviewSession(result.output_dir)
    session.set_text(0, "edited")
    session.save()
    from app.core.exporter import read_srt
    assert all(abs(c.start * 25 - round(c.start * 25)) < 0.06 for c in read_srt(session.target_srt))
