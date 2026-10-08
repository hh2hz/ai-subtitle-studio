"""yt-dlp adapter: metadata, audio download and platform subtitles.

yt-dlp breaks when sites change, so all yt-dlp usage is isolated here. Users update it with
`pip install -U "yt-dlp[default]"`. YouTube extraction needs a JavaScript runtime (deno by default).
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from app.core.errors import JobCancelled, PipelineError
from app.providers.base import ProviderError, ProviderUnavailable

log = logging.getLogger(__name__)

RETRY_DELAYS_S = (5.0, 20.0)

# Browser names yt-dlp accepts for --cookies-from-browser. The user must be signed in there; the app only
# passes the name, so no cookie value ever passes through our code.
COOKIE_BROWSERS = ("firefox", "brave", "chrome", "chromium", "edge", "opera", "vivaldi")

QUALITY_BEST = "best"

# Chain fragment that keeps HLS/m3u8 streams out of the first attempt (D-083).
_DIRECT = "[protocol!*=m3u8]"
# Bumped whenever the format chain changes: it is part of the download cache key, so a file downloaded with an
# older chain is fetched again instead of being reused (1 = the old `[ext=mp4]` chain, 2 = direct-first,
# 3 = cookies are no longer applied by default, because they hid every HD format, D-085).
CHAIN_VERSION = 3


def available_qualities(info: dict | None) -> list[int]:
    """Distinct heights of the video formats the site offers for this link, highest first (D-080).

    A format without a video codec ("none" or missing) does not count, so an audio-only page yields [].
    """
    heights: set[int] = set()
    for fmt in (info or {}).get("formats") or []:
        codec = fmt.get("vcodec")
        if not codec or codec == "none":
            continue
        height = fmt.get("height")
        if isinstance(height, (int, float)) and height > 0:
            heights.add(int(height))
    return sorted(heights, reverse=True)


def format_selector(quality: str | int | None, ffmpeg: bool = True) -> str:
    """The yt-dlp `-f` value for a requested quality ("best" or a height in pixels).

    Direct (https/DASH) streams come first and HLS (m3u8) last, because a fragment download failed on this machine
    with a Windows file lock (`Unable to remove file: [WinError 32] ... .f616.mp4.part-Frag1119`, job
    d7786157b0216533: the old chain had picked the 1080p "Premium" HLS format). The generic chain stays as the
    fallback, so an HLS-only link still downloads. No extension filter is used: the old `[ext=mp4]` restriction
    could drop every HD stream and land on the progressive 360p one (measured, D-080).
    """
    if quality in (None, "", QUALITY_BEST):
        if not ffmpeg:
            return f"b{_DIRECT}/b"
        return f"bv*{_DIRECT}+ba{_DIRECT}/bv*+ba/b"
    height = int(quality)
    if not ffmpeg:
        return f"b[height<={height}]{_DIRECT}/b[height<={height}]/b"
    return (f"bv*[height<={height}]{_DIRECT}+ba{_DIRECT}"
            f"/bv*[height<={height}]+ba/b[height<={height}]")

# Options that switch the browser/file cookies on; None is yt-dlp's "unset" value for both.
_COOKIE_KEYS = ("cookiesfrombrowser", "cookiefile")
# Errors that mean "this video needs a signed-in session": only then are cookies worth using (D-085).
_SIGNIN_KEYS = ("error.login_required", "error.age_restricted", "error.youtube_bot_check")
# A cookie-less probe that already offers this height is taken as it is; below it the cookie probe is compared too.
_HD_ENOUGH = 1080
# Moving the finished file out of the work folder: another process (antivirus, OneDrive, indexer) may hold it.
MOVE_ATTEMPTS = 10
MOVE_MAX_BACKOFF_S = 5.0

# Refusals that are often caused by the IPv6 route: one silent IPv4 retry is worth it, nothing more.
_IPV4_RETRY_KEYS = ("error.youtube_bot_check", "error.po_token")


def js_runtime_available() -> str | None:
    """Name of a JavaScript runtime yt-dlp can use without extra configuration, if any."""
    return "deno" if shutil.which("deno") else None


def access_options(extras: dict) -> dict:
    """yt-dlp options for the user's site-access settings (D-073).

    Values are never secrets for us: `cookiesfrombrowser` carries a browser name and `cookiefile` a path.
    The cookie data itself stays in the browser profile or the user's own file.
    """
    options: dict = {}
    source = extras.get("cookies_source", "")
    if source == "browser" and extras.get("cookies_browser"):
        options["cookiesfrombrowser"] = (extras["cookies_browser"], None, None, None)
    elif source == "file" and extras.get("cookies_file"):
        options["cookiefile"] = extras["cookies_file"]
    if extras.get("force_ipv4"):
        options["force_ipv4"] = True
    return options


def _classify(exc: Exception) -> PipelineError:
    text = str(exc)
    lowered = text.lower()
    if "unsupported url" in lowered:
        return PipelineError(text, "error.url_unsupported")
    # Age-gated videos ask for a sign-in too ("Sign in to confirm your age"), so this must come first.
    if any(s in lowered for s in ("confirm your age", "age-restricted", "age restricted",
                                  "inappropriate for some users", "age gate")):
        return PipelineError(text, "error.age_restricted")
    # A sign-in demand comes in several shapes; the bot check must be recognised before the generic
    # "use --cookies" hint, because YouTube's bot-check text contains that hint too.
    if "sign in to confirm" in lowered:
        return PipelineError(text, "error.youtube_bot_check")
    if any(s in lowered for s in ("only works when logged-in", "login required", "requires authentication",
                                  "sign in to view", "log in to view", "account credentials",
                                  "use --cookies", "use --username", "authentication")):
        return PipelineError(text, "error.login_required")
    if any(s in lowered for s in ("po token", "po_token", "failed to extract any player response")):
        return PipelineError(text, "error.po_token")
    if any(s in lowered for s in ("not available in your country", "geo restricted", "geo-restricted",
                                  "blocked in your country", "available in your country")):
        return PipelineError(text, "error.geo_blocked")
    if any(s in lowered for s in ("video unavailable", "private video", "has been removed", "not available")):
        return PipelineError(text, "error.video_unavailable")
    if "429" in lowered or "too many requests" in lowered:
        return PipelineError(text, "error.rate_limited")
    # A 403 on the media data ("unable to download video data: HTTP Error 403: Forbidden") is what YouTube returns
    # for an expired or session-bound media URL. Measured on the 200. Bölüm link (2026-10-07): the identical
    # command succeeded a couple of minutes later, so the job must retry with a freshly extracted URL instead of
    # dying on the first answer (D-107).
    if "403" in lowered or "forbidden" in lowered or "unable to download video data" in lowered:
        return PipelineError(text, "error.download_forbidden")
    return PipelineError(text, "error.download_failed")


#: Keys worth another attempt with a freshly extracted URL.
_TRANSIENT_KEYS = ("error.rate_limited", "error.download_forbidden")


def _discard_partials(target_base: Path) -> int:
    """Delete unfinished downloads for one base name before a retry; returns how many files were removed.

    A `.part`/fragment file keeps the offset of a media URL that has since expired, and resuming against it is what
    turns a transient 403 into a permanent failure (D-107).
    """
    removed = 0
    for path in target_base.parent.glob(target_base.name + ".*"):
        if not path.is_file():
            continue
        suffix = path.name[len(target_base.name):].lower()
        if ".part" in suffix or ".ytdl" in suffix or suffix.endswith((".part", ".temp")):
            try:
                path.unlink()
                removed += 1
            except OSError:                     # a scanner may hold it; the retry will overwrite it anyway
                log.info("Could not remove the partial file %s", path.name)
    if removed:
        log.info("Removed %d unfinished download file(s) before retrying", removed)
    return removed


class YtDlpAdapter:
    def __init__(self, retry_delays: tuple[float, ...] = RETRY_DELAYS_S, sleep: Callable[[float], None] = time.sleep,
                 extra_options: dict | None = None):
        self._retry_delays = retry_delays
        self._sleep = sleep
        self._extra = dict(extra_options or {})
        self._cookies_for: dict[str, bool] = {}    # url -> whether the probe that won used the cookies
        self.access_note: str | None = None        # set by probe() when the cookie setting was skipped or needed

    def _has_cookies(self) -> bool:
        return any(self._extra.get(k) for k in _COOKIE_KEYS)

    def _cookie_less(self) -> dict:
        return {**self._extra, **{k: None for k in _COOKIE_KEYS}}

    @staticmethod
    def version() -> str:
        from yt_dlp.version import __version__

        return __version__

    def _options(self, **extra) -> dict:
        # noplaylist: a "watch?v=X&list=Y" link means the video X, not the whole playlist.
        opts = {"quiet": True, "no_warnings": False, "noprogress": True, "logger": _YtDlpLogger(), "noplaylist": True}
        opts.update(self._extra)
        opts.update(extra)
        return opts

    def _run(self, action: Callable[[dict], object], base: dict | None = None) -> object:
        """Run one attempt. A refusal that is often the IPv6 route is retried once over IPv4 (D-073)."""
        base = self._extra if base is None else base
        try:
            return action(base)
        except (JobCancelled, KeyboardInterrupt):
            raise
        except Exception as exc:
            error = exc if isinstance(exc, PipelineError) else _classify(exc)
            if error.ui_key not in _IPV4_RETRY_KEYS or base.get("force_ipv4"):
                raise error from exc
            log.info("Site refused the request (%s); retrying once over IPv4", error.ui_key)
            return action({**base, "force_ipv4": True})

    def _with_retry(self, action: Callable[[], object]) -> object:
        attempts = len(self._retry_delays) + 1
        for attempt in range(attempts):
            try:
                return action()
            except (JobCancelled, KeyboardInterrupt):
                raise
            except Exception as exc:
                error = exc if isinstance(exc, PipelineError) else _classify(exc)
                if error.ui_key not in _TRANSIENT_KEYS or attempt == attempts - 1:
                    raise error from exc
                delay = self._retry_delays[attempt]
                log.warning("Site refused the download (%s); retrying in %.0f s", error.ui_key, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")

    def _probe_once(self, url: str, base: dict) -> dict:
        import yt_dlp

        def run(attempt: dict):
            with yt_dlp.YoutubeDL(self._options(skip_download=True, **attempt)) as ydl:
                return ydl.sanitize_info(ydl.extract_info(url, download=False))

        info = self._with_retry(lambda: self._run(run, base))
        if info.get("_type") == "playlist":
            raise PipelineError("Playlists are not supported; use a single video URL", "error.url_unsupported")
        return info

    def probe(self, url: str) -> dict:
        """Metadata and available subtitle tracks, without downloading media.

        With a cookie setting the probe runs WITHOUT cookies first: measured on this machine, browser cookies made
        YouTube offer only the 360p format 18 where the cookie-less probe offered 1080p (D-085). Cookies are used when
        the cookie-less probe is refused with a sign-in demand, or offers no video, or offers less than
        `_HD_ENOUGH` and the cookie probe offers more. The winner is remembered for `download_video`.
        """
        if not js_runtime_available():
            log.warning("No JavaScript runtime (deno) found; YouTube extraction may be incomplete")
        self.access_note = None
        if not self._has_cookies():
            return self._probe_once(url, self._extra)

        plain: dict | None = None
        try:
            plain = self._probe_once(url, self._cookie_less())
        except PipelineError as exc:
            if exc.ui_key not in _SIGNIN_KEYS:
                raise
            log.info("Cookie-less probe needs a sign-in (%s); using the cookies", exc.ui_key)
        plain_best = (available_qualities(plain) or [0])[0] if plain is not None else 0
        if plain is not None and plain_best >= _HD_ENOUGH:
            self._cookies_for[url] = False
            self.access_note = "Cookies skipped: they are not needed for this video and can hide HD formats"
            return plain
        try:
            with_cookies = self._probe_once(url, self._extra)
        except PipelineError:
            if plain is None:
                raise
            log.warning("Probe with cookies failed; using the cookie-less result", exc_info=True)
            self._cookies_for[url] = False
            return plain
        cookie_best = (available_qualities(with_cookies) or [0])[0]
        if plain is not None and plain_best >= cookie_best:
            log.info("Probe: cookie-less offers up to %sp, with cookies %sp; cookies skipped", plain_best, cookie_best)
            self._cookies_for[url] = False
            self.access_note = "Cookies skipped: they did not offer a higher quality"
            return plain
        log.info("Probe: with cookies up to %sp, cookie-less %sp; cookies used", cookie_best, plain_best)
        self._cookies_for[url] = True
        return with_cookies

    def extract_playlist(self, url: str) -> list[dict]:
        """Extract individual video URLs and titles from a playlist URL without downloading."""
        import yt_dlp

        if not js_runtime_available():
            log.warning("No JavaScript runtime (deno) found; YouTube extraction may be incomplete")

        def run(attempt: dict):
            opts = self._options(skip_download=True, extract_flat="in_playlist", noplaylist=False, **attempt)
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.sanitize_info(ydl.extract_info(url, download=False))

        info = self._with_retry(lambda: self._run(run))
        entries = info.get("entries") or []
        results = []
        for entry in entries:
            if not entry:
                continue
            entry_url = entry.get("webpage_url") or entry.get("url")
            video_id = entry.get("id")
            if entry_url and not entry_url.startswith("http"):
                entry_url = f"https://www.youtube.com/watch?v={entry_url}"
            elif not entry_url and video_id:
                entry_url = f"https://www.youtube.com/watch?v={video_id}"
            if entry_url:
                title = entry.get("title") or video_id or entry_url
                results.append({"url": entry_url, "title": title})
        return results


    def download_video(self, url: str, dest_base: Path, cancel: threading.Event | None = None,
                       progress: Callable[[float], None] | None = None, quality: str = QUALITY_BEST,
                       warn: Callable[[str], None] | None = None, work_dir: Path | None = None) -> Path:
        """Download the video as MP4 to <dest_base>.mp4 (audio is extracted from it later).

        Merging separate video/audio streams needs the FFmpeg command-line tool; without it a single-file
        format is used, which on YouTube is usually low resolution. `quality` is "best" or a height in pixels;
        the height that was really downloaded is always logged and reported through `warn` when it is lower than
        what was asked for or lower than what the site offered (D-080, a silent 360p download hid the problem).

        With `work_dir` the media is downloaded and merged there and only the finished file is moved into
        <dest_base>'s folder, with retries: on Windows another process may hold a freshly written large file for
        a while, which failed the final rename with WinError 32 (D-085).
        """
        import yt_dlp

        dest_base.parent.mkdir(parents=True, exist_ok=True)
        if work_dir is not None:
            work_dir.mkdir(parents=True, exist_ok=True)
        target_base = dest_base if work_dir is None else work_dir / dest_base.name
        use_cookies = True
        if self._has_cookies():
            if url not in self._cookies_for:
                try:
                    self.probe(url)                 # normally done by the caller already; decides cookies or not
                except PipelineError:
                    self._cookies_for[url] = True   # keep the old behaviour when the probe cannot decide
            use_cookies = self._cookies_for.get(url, True)
        base = self._extra if use_cookies else self._cookie_less()
        ffmpeg = bool(shutil.which("ffmpeg"))
        if not ffmpeg:
            log.warning("FFmpeg not found: downloading a single-file format (often lower resolution)")
        fmt = format_selector(quality, ffmpeg)
        log.info("Download format for quality %s: %s", quality, fmt)

        def hook(status: dict) -> None:
            if cancel is not None and cancel.is_set():
                raise JobCancelled()
            if progress and status.get("status") == "downloading":
                total = status.get("total_bytes") or status.get("total_bytes_estimate")
                if total:
                    progress(min(status.get("downloaded_bytes", 0) / total, 1.0))

        selected: dict = {}
        state = {"attempt": 0}

        def run(attempt: dict):
            state["attempt"] += 1
            if state["attempt"] > 1:
                # Resume would replay an expired URL and 403 again: start the retry from a clean slate (D-107).
                _discard_partials(target_base)
            opts = self._options(
                format=fmt,
                merge_output_format="mp4",
                outtmpl=str(target_base).replace("%", "%%") + ".%(ext)s",   # a "%(...)s" in a title is literal text
                progress_hooks=[hook],
                continuedl=True,           # resume partial downloads after a crash
                overwrites=False,
                file_access_retries=10,    # a scanner may hold a fragment for a moment (default is 3)
                **attempt,
            )
            with yt_dlp.YoutubeDL(opts) as ydl:
                selected["info"] = ydl.extract_info(url, download=True) or {}

        try:
            self._with_retry(lambda: self._run(run, base))
        except PipelineError as exc:
            if use_cookies or not self._has_cookies() or exc.ui_key not in _SIGNIN_KEYS:
                raise
            log.info("Download without cookies needs a sign-in (%s); retrying with the cookies", exc.ui_key)
            use_cookies = True
            self._cookies_for[url] = True
            self._with_retry(lambda: self._run(run, self._extra))
        # Titles can contain dots and glob characters, so match "<base>.<ext>" exactly; this also skips
        # yt-dlp intermediate files such as "<base>.f137.mp4" and "*.part".
        prefix = dest_base.name + "."
        candidates = [p for p in target_base.parent.iterdir()
                      if p.is_file() and p.name.startswith(prefix)
                      and p.name[len(prefix):].lower() in ("mp4", "mkv", "webm", "m4a")]
        candidates.sort(key=lambda p: (not p.name.lower().endswith(".mp4"), -p.stat().st_size))
        if not candidates:
            raise PipelineError(f"Download finished but no media file was found for {dest_base}", "error.download_failed")
        self._report_quality(selected.get("info") or {}, quality, warn,
                             cookies=use_cookies and self._has_cookies())
        chosen = candidates[0]
        if work_dir is not None:
            chosen = self._move_finished(chosen, dest_base.parent / chosen.name)
        return chosen

    def _move_finished(self, src: Path, dst: Path) -> Path:
        """Move the finished file into the output folder, retrying while another process holds it (D-085)."""
        last: OSError | None = None
        for attempt in range(MOVE_ATTEMPTS):
            try:
                _place(src, dst)
                return dst
            except OSError as exc:
                last = exc
                delay = min(0.5 * 2 ** attempt, MOVE_MAX_BACKOFF_S)
                log.warning("Could not place %s (%s); retry %d/%d in %.1f s", dst.name, exc, attempt + 1,
                            MOVE_ATTEMPTS, delay)
                self._sleep(delay)
        raise PipelineError(f"The downloaded file could not be moved into the output folder after {MOVE_ATTEMPTS} "
                            f"attempts ({last}). Another program (antivirus, cloud sync) may be holding it; the "
                            f"complete file is kept at {src}", "error.download_failed") from last

    def _report_quality(self, info: dict, quality: str, warn: Callable[[str], None] | None,
                        cookies: bool = False) -> None:
        """Log the height that was really downloaded and warn when it is lower than needed (D-080)."""
        got = info.get("height")
        got = int(got) if isinstance(got, (int, float)) and got else None
        offered = available_qualities(info)
        best_offered = offered[0] if offered else None
        requested = None if quality in (None, "", QUALITY_BEST) else int(quality)
        format_id = info.get("format_id") or "?"
        log.info("Requested %sp, got %sp (format_id %s)",
                 requested or QUALITY_BEST, got or "no video", format_id)
        if got is None:
            return                                  # audio-only source: nothing to compare
        reason = None
        if requested is not None and got < requested:
            reason = f"Requested {requested}p, got {got}p (format_id {format_id})"
        elif best_offered is not None and got < best_offered:
            reason = f"Best offered was {best_offered}p, got {got}p (format_id {format_id})"
        if reason and cookies:
            reason += "; the cookie setting is on and can hide HD formats"
        if reason and not js_runtime_available():
            reason += "; no JavaScript runtime (deno) found"
        if reason and warn is not None:
            warn(reason)

    def download_subtitle(self, url: str, language: str, automatic: bool, dest_dir: Path) -> Path:
        """Download one subtitle track as WebVTT. Raises ProviderError on failure."""
        import yt_dlp

        dest_dir.mkdir(parents=True, exist_ok=True)
        kind = "auto" if automatic else "manual"
        logger = _YtDlpLogger()

        def run(attempt: dict):
            opts = self._options(
                logger=logger,
                skip_download=True,
                writesubtitles=not automatic,
                writeautomaticsub=automatic,
                subtitleslangs=[language],
                subtitlesformat="vtt",
                outtmpl=str(dest_dir / f"yt.{kind}.%(ext)s"),
                **attempt,
            )
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])

        base = self._cookie_less() if self._cookies_for.get(url) is False else self._extra   # same access as the download
        try:
            self._with_retry(lambda: self._run(run, base))
        except PipelineError as exc:
            raise ProviderError(str(exc), exc.ui_key, retryable=exc.ui_key == "error.rate_limited") from exc
        path = dest_dir / f"yt.{kind}.{language}.vtt"
        if not path.is_file():
            if logger.po_token_missing:
                raise ProviderUnavailable(f"YouTube withheld the {language} {kind} subtitles (a PO token is required)")
            raise ProviderError(f"Subtitle {language} ({kind}) was not written")
        return path


def _place(src: Path, dst: Path) -> None:
    """Rename `src` to `dst`; when that fails (locked file, other volume) copy it, verify the size, then remove `src`."""
    try:
        os.replace(src, dst)
        return
    except OSError:
        pass
    partial = dst.with_name(dst.name + ".copying")
    try:
        shutil.copyfile(src, partial)
        if partial.stat().st_size != src.stat().st_size:
            raise OSError("copied file has a different size than the source")
        os.replace(partial, dst)
    finally:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass
    try:
        src.unlink()
    except OSError:
        log.warning("Could not remove the work copy %s", src)


class _YtDlpLogger:
    """Route yt-dlp messages into our logging. yt-dlp reports some errors only through the logger."""

    po_token_missing = False

    def debug(self, msg: str) -> None:
        if not msg.startswith("[debug] "):
            log.debug(msg)

    def info(self, msg: str) -> None:
        log.debug(msg)

    def warning(self, msg: str) -> None:
        if "PO token was not provided" in msg:
            # YouTube withholds some subtitle tracks without a PO token; the pipeline falls back to Whisper (D-086).
            self.po_token_missing = True
            log.info("yt-dlp: %s", msg)
            return
        log.warning("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        log.error("yt-dlp: %s", msg)
