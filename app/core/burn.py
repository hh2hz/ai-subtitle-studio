"""A copy of the video with the subtitles drawn into the picture ("burned in"), playable anywhere.

The final SRT is converted to an ASS file with an explicit style (bottom centre, white text, black outline, size
relative to the video height) and drawn by FFmpeg's libass-based `subtitles` filter, which shapes Arabic and lays
out right-to-left text itself, so the result does not depend on a player's subtitle renderer (D-040).
Video is re-encoded (NVIDIA NVENC when available, otherwise libx264); audio is copied.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

from app.core.errors import JobCancelled, PipelineError
from app.core.exporter import Cue, read_srt

log = logging.getLogger(__name__)

Progress = Callable[[float], None]
SUFFIX = ".subtitled.mp4"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ENCODERS = (
    ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"]),
    ("libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]),
)


def episode_files(episode_dir: Path) -> tuple[Path, Path]:
    """(video, final subtitle) of an episode folder created by the app."""
    from app.core.review import VIDEO_EXTENSIONS

    episode_dir = Path(episode_dir)
    srts = sorted(episode_dir.glob("*.srt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not srts:
        raise PipelineError(f"No subtitle file in {episode_dir}", "error.burn_no_files")
    srt = srts[0]
    base = srt.name[: -len(".srt")].rsplit(".", 1)[0]
    videos = [episode_dir / f"{base}{ext}" for ext in VIDEO_EXTENSIONS]
    videos += sorted(p for p in episode_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    for video in videos:
        if video.is_file() and not video.name.endswith(SUFFIX):
            return video, srt
    raise PipelineError(f"No video file in {episode_dir}", "error.burn_no_files")


def output_path(target_srt: Path) -> Path:
    """<title>.<lang>.srt -> <title>.<lang>.subtitled.mp4 next to it."""
    return target_srt.with_name(target_srt.name[: -len(".srt")] + SUFFIX)


def _ass_time(seconds: float) -> str:
    centis = max(0, round(seconds * 100))
    h, rem = divmod(centis, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_text(text: str) -> str:
    text = text.replace("\\", "⧵").replace("{", "(").replace("}", ")")
    return r"\N".join(line.strip() for line in text.split("\n"))


def write_ass(cues: list[Cue], path: Path, width: int, height: int, font: str = "Arial") -> None:
    """Bottom-centre style scaled to the video: text about 5.5 % of the height, outline for any background."""
    size = max(16, round(height * 0.055))
    outline = max(1.5, round(height / 400, 1))
    margin_v = max(10, round(height * 0.06))
    margin_h = max(10, round(width * 0.05))
    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\nPlayResY: {height}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,"
        f"{outline},{max(1, round(outline / 2))},2,{margin_h},{margin_h},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = [f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Default,,0,0,0,,{_ass_text(c.text)}"
              for c in cues if c.text.strip()]
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")


def video_info(video: Path) -> tuple[int, int, float]:
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        duration = float(container.duration / 1_000_000) if container.duration else 0.0
        return stream.codec_context.width, stream.codec_context.height, duration


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise PipelineError("FFmpeg was not found; install it to create a video with subtitles", "error.no_ffmpeg")
    filters = subprocess.run([exe, "-hide_banner", "-filters"], capture_output=True, text=True, errors="replace",
                             creationflags=_NO_WINDOW).stdout
    if not re.search(r"\ssubtitles\s", filters):
        raise PipelineError("This FFmpeg build has no subtitles filter (libass)", "error.no_libass")
    return exe


def burn(video: Path, target_srt: Path, progress: Progress | None = None,
         cancel: threading.Event | None = None, font: str = "Arial", ffmpeg: str | None = None) -> Path:
    """Write <title>.<lang>.subtitled.mp4 next to the subtitle and return its path."""
    ffmpeg = ffmpeg or find_ffmpeg()
    width, height, duration = video_info(video)
    work = target_srt.parent / "work"
    work.mkdir(parents=True, exist_ok=True)
    ass = work / "burn.ass"
    write_ass(read_srt(target_srt), ass, width, height, font)
    out = output_path(target_srt)
    tmp = out.with_name(out.stem + ".part.mp4")
    last_error = ""
    for name, args in ENCODERS:
        # The filter reads the ASS file by a plain relative name (cwd = work/): Windows drive letters and quotes
        # in titles need no filter escaping that way.
        cmd = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "error", "-y", "-i", str(video.resolve()),
               "-vf", "subtitles=burn.ass", *args, "-pix_fmt", "yuv420p", "-c:a", "copy",
               "-movflags", "+faststart", "-progress", "pipe:1", str(tmp.resolve())]
        log.info("Burning subtitles with %s: %s", name, out.name)
        code, last_error = _run(cmd, work, duration, progress, cancel)
        if code == 0:
            tmp.replace(out)
            if progress:
                progress(1.0)
            return out
        tmp.unlink(missing_ok=True)
        log.warning("Encoder %s failed: %s", name, last_error[-300:])
    raise PipelineError(f"FFmpeg could not create the video: {last_error[-300:]}", "error.burn_failed")


def _run(cmd: list[str], cwd: Path, duration: float, progress: Progress | None,
         cancel: threading.Event | None) -> tuple[int, str]:
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            errors="replace", creationflags=_NO_WINDOW)
    errors: list[str] = []
    reader = threading.Thread(target=lambda: errors.extend(proc.stderr), daemon=True)
    reader.start()
    for line in proc.stdout:
        if cancel is not None and cancel.is_set():
            proc.kill()
            proc.wait()
            raise JobCancelled()
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and duration > 0 and progress:
            try:
                progress(min(0.99, int(value) / 1_000_000 / duration))
            except ValueError:
                pass
    code = proc.wait()
    reader.join(timeout=5)
    return code, "".join(errors)
