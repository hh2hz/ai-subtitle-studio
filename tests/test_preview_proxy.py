"""The compatible preview copy used when the review window cannot play the original video (D-119)."""

import shutil
import subprocess

import pytest

from app.core import preview_proxy
from app.core.errors import JobCancelled, PipelineError

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")


def test_proxy_path_changes_with_the_file(tmp_path):
    video = tmp_path / "a.mkv"
    video.write_bytes(b"1")
    first = preview_proxy.proxy_path(video, tmp_path)
    video.write_bytes(b"22")

    assert preview_proxy.proxy_path(video, tmp_path) != first
    assert first.suffix == ".mp4" and first.parent == tmp_path


def test_an_existing_copy_is_reused_without_ffmpeg(tmp_path, monkeypatch):
    video = tmp_path / "a.mkv"
    video.write_bytes(b"1")
    target = preview_proxy.proxy_path(video, tmp_path / "p")
    target.parent.mkdir()
    target.write_bytes(b"ready")
    monkeypatch.setattr(preview_proxy.shutil, "which", lambda name: None)

    assert preview_proxy.make_preview(video, folder=tmp_path / "p") == target


def test_missing_ffmpeg_is_a_clear_error(tmp_path, monkeypatch):
    video = tmp_path / "a.mkv"
    video.write_bytes(b"1")
    monkeypatch.setattr(preview_proxy.shutil, "which", lambda name: None)

    with pytest.raises(PipelineError):
        preview_proxy.make_preview(video, folder=tmp_path / "p")


@needs_ffmpeg
def test_a_real_video_is_converted_to_h264_540p_with_the_same_length(tmp_path):
    src = tmp_path / "in.mkv"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2", "-c:v", "mpeg4", "-c:a", "libvorbis",
                    str(src)], check=True)

    out = preview_proxy.make_preview(src, folder=tmp_path / "p")

    info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,height:format=duration",
                           "-of", "default=nw=1", str(out)], capture_output=True, text=True).stdout
    assert "h264" in info and "height=540" in info and "aac" in info
    assert abs(float(info.split("duration=")[1].split()[0]) - 2.0) < 0.3
    assert [p.name for p in out.parent.iterdir()] == [out.name]


def test_cancel_leaves_no_partial_file(tmp_path, monkeypatch):
    video = tmp_path / "a.mkv"
    video.write_bytes(b"1")

    def cancelled(*args, **kwargs):
        raise JobCancelled()

    monkeypatch.setattr(preview_proxy.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(preview_proxy.burn, "video_info", lambda path: (1, 1, 1.0))
    monkeypatch.setattr(preview_proxy.burn, "_run", cancelled)

    with pytest.raises(JobCancelled):
        preview_proxy.make_preview(video, folder=tmp_path / "p")

    assert list((tmp_path / "p").glob("*.part.mp4")) == []


@needs_ffmpeg
def test_the_software_surface_receives_and_paints_frames(qtbot, tmp_path):
    from PySide6.QtCore import QUrl
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

    from app.ui.video_surface import VideoSurface

    video = tmp_path / "t.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10", "-t", "2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    surface = VideoSurface()
    surface.resize(240, 135)
    qtbot.addWidget(surface)
    surface.show()
    player = QMediaPlayer()
    audio = QAudioOutput()
    player.setAudioOutput(audio)
    player.setVideoSink(surface.sink)
    player.setSource(QUrl.fromLocalFile(str(video)))
    player.play()

    qtbot.waitUntil(lambda: surface.frames > 3, timeout=10000)

    assert surface.pixmap() is not None and not surface.pixmap().isNull()
    assert surface.pixmap().width() <= 240 and surface.pixmap().height() <= 135
    player.stop()
