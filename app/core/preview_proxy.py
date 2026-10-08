"""A small H.264/AAC copy of a video for the review window's preview (D-119).

Qt's media backend plays what the Windows codecs support; downloads often arrive as AV1, VP9 or 10-bit HEVC, which
then show a black picture or nothing at all. When that happens the review window asks for this copy: 540 p, 8-bit
H.264 and AAC, which every installation can play. Timing is identical to the original, so the cues line up.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Callable

from app.core import burn
from app.core.errors import JobCancelled, PipelineError

log = logging.getLogger(__name__)

HEIGHT = 540
FOLDER_NAME = "AISubtitleStudio-preview"


def proxy_path(video: Path, folder: Path | None = None) -> Path:
    """Where the copy of this exact file (path, size, modification time) lives."""
    video = Path(video).resolve()
    stat = video.stat()
    key = hashlib.sha256(f"{video}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")).hexdigest()[:16]
    return (folder or Path(tempfile.gettempdir()) / FOLDER_NAME) / f"{key}.mp4"


def make_preview(video: Path, progress: Callable[[float], None] | None = None,
                 cancel: threading.Event | None = None, folder: Path | None = None,
                 ffmpeg: str | None = None) -> Path:
    """Return the preview copy of `video`, creating it when needed (an existing one is reused)."""
    video = Path(video)
    target = proxy_path(video, folder)
    if target.is_file() and target.stat().st_size > 0:
        return target
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise PipelineError("FFmpeg was not found", "error.no_ffmpeg")
    target.parent.mkdir(parents=True, exist_ok=True)
    for stale in target.parent.glob("*"):                 # copies of other or changed videos are not needed
        if stale != target:
            stale.unlink(missing_ok=True)
    try:
        duration = burn.video_info(video)[2]
    except Exception:                                  # noqa: BLE001 - only used for the progress bar
        duration = 0.0
    tmp = target.with_name(target.stem + ".part.mp4")
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "error", "-y", "-i", str(video.resolve()),
           "-map", "0:v:0", "-map", "0:a:0?", "-vf", f"scale=-2:'min({HEIGHT},ih)'",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-progress", "pipe:1", str(tmp)]
    log.info("Creating a compatible preview copy of %s", video.name)
    try:
        code, error = burn._run(cmd, target.parent, duration, progress, cancel)
    except JobCancelled:
        tmp.unlink(missing_ok=True)
        raise
    if code != 0:
        tmp.unlink(missing_ok=True)
        raise PipelineError(f"FFmpeg could not create the preview copy: {error[-300:]}", "error.preview_failed")
    tmp.replace(target)
    return target
