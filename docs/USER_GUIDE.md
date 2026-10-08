# AI Subtitle Studio - User Guide

A practical guide to installing, running and reviewing subtitles with AI Subtitle Studio (Windows, PySide6). Everything runs on your computer except the optional cloud AI pass. The names, paths and defaults below come from the application code.

## Contents

1. Install and start the app
2. Where the app keeps its files
3. The main window
4. The Settings window: every option
5. Cookies and download access
6. Downloads come out at 360p
7. AI providers, quota and timeouts
8. Double check, ReviewRequired.txt and the review window
9. Diarization and speakers
10. Caches, resume and maintenance
11. Troubleshooting
12. Known limitations

**1. Install and start the app**

Installed build: run the installer (a per-user install, no administrator rights needed; it installs to `%LOCALAPPDATA%\Programs\AI Subtitle Studio`, ships `ffmpeg`, `ffprobe` and `deno` in its `bin\` folder, which are put first on `PATH` at startup, and keeps your data folder on uninstall), then start it from the Start menu. On the first run an NVIDIA GPU without the CUDA libraries triggers a one-time offer to download them (about 1.2 GB); you can postpone it and use "Settings > Download NVIDIA GPU acceleration...".

From source (Python 3.12-3.14):

```bat
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -r requirements-gpu.txt   REM optional, NVIDIA GPU only (~1.2 GB)
python -m app.main
```

`launch.bat` starts the same app without a console window. Options: `--data-dir <path>` or the environment variable `AISS_DATA_DIR` moves the data folder; `--smoke-test` starts the GUI and exits; at start-up a small window checks, repairs and updates the libraries by itself and shows an error only if that fails (setting `dependency_check`); `--self-test` checks that a packaged build is complete.

First-run facts:
- Models download on first use: Whisper `large-v3-turbo` about 1.6 GB, MADLAD-400 3B about 3 GB, the built-in TranslateGemma 4B about 2.4 GB, the llama.cpp runtime (about 263 MB CUDA build, about 19 MB CPU build) and the diarization models. Without the GPU packages everything runs on the CPU.
- `ffmpeg` is needed to merge high-resolution video and audio and to burn subtitles into a video. Without it a single-file format is downloaded, which is usually low resolution (section 6).
- YouTube needs a JavaScript runtime: `winget install DenoLand.Deno`. Without it the log warns "No JavaScript runtime (deno) found; YouTube extraction may be incomplete".
- You are responsible for having the rights to any content you process.

**2. Where the app keeps its files**

The data root is `%LOCALAPPDATA%\AISubtitleStudio` unless you pass `--data-dir` or set `AISS_DATA_DIR`.

| What | Location |
| --- | --- |
| Settings database | `<data root>\studio.db` (SQLite; settings are stored as JSON values) |
| Log file | `<data root>\logs\app.log` (rotates at 5 MB, 3 backups) |
| Models | `<data root>\models\whisper\<name>`, `...\translation\<key>`, `...\llama.cpp\<build>`, `...\diarization\<model>` |
| Per-job caches | `<data root>\jobs\<job key>\` (section 10) |
| Provider memory | `<data root>\provider_health.json` (section 7) |
| API keys | `<data root>\api_keys.json`, or `api_keys.local.json` next to the app when running from source |
| GPU libraries | `<data root>\cuda` |
| Name glossaries | `<data root>\glossaries\<series>.<lang>.json`, written per series and target language |
| Default output folder | from source the project's `output\`; installed `Videos\AI Subtitle Studio` |

Notes: a model folder counts as installed only when it contains `.download-complete`, so an interrupted download is fetched again. API key files are files you create yourself from the shape shown in `README.md`; no key file and no example key file ships with the app. Keys are never written to the database or the log; the OpenSubtitles and SubDL keys entered in Settings are stored unencrypted in the local database.

**3. The main window**

- **Input**: paste a video URL (YouTube, Vimeo and the thousands of other sites yt-dlp supports) or use Browse... for a local file. The button reads "Translate" for a file and "Download & Translate" for a link.
- **Source language**: default "Auto Detect". Pick the spoken language when detection is wrong.
- **Target language**: default "Arabic". The list has 19 languages (Arabic, Chinese, Dutch, English, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, Portuguese, Russian, Spanish, Turkish, Ukrainian, Urdu); source and target must differ.
- **Mode**: default "Balanced". Fast = one ASR pass with minimal context (ASR beam 1, translation beam 2); Balanced = beam 5/4 with subtitle sources, re-transcription and verification; Maximum Accuracy = beam 5/5 with stricter thresholds and deeper verification, and it is the slowest.
- **Video quality**: default "Best available". For a link the list holds the heights that link offers right now, and the choice is part of the download cache key; it is disabled for local files. A saved height that is no longer offered shows "not offered for this link (best: ...p); the best available will be used instead". A link that only offers low heights says so and names the cookie setting as the likely cause.
- **Series (optional)**: series name, season and episode. They are auto-detected from the link or the file name when empty (0 shows "Auto") and never overwrite what you typed. "Keep this series" saves the name and its poster.
- **Output folder**: empty means the default from section 2; each job gets its own `<title>\` folder.
- **Buttons**: Translate / Download & Translate, Cancel (finished work is kept and can be resumed), Open output folder, Review, "Save video with subtitles", Queue, History, Settings, and the theme menu.

While a job runs the input, language, mode, series and quality controls are disabled.

**4. The Settings window: every option**

Speech recognition:
- "Difficult audio (music, noise, quiet speech)" = `audio_enhance`, default `off`; the other choices are "Automatic (on in Maximum accuracy)" and "Always clean the audio". Cleaning costs 15-20 % more time and no accuracy gain is proven, so it stays off by default; turn it on for music or heavy noise.
- "Detect who is speaking (runs on the CPU)" = `diarization`, default on (section 9).
- "Snap subtitle times to scene changes (slower)" = `snap_to_shots`, default off; it moves a start or end that is within 0.25 s of a cut onto the cut and adds minutes per episode. Frame snapping is always on.

Translation:
- "Translation engine" = `translation_engine`, default `local` (built-in TranslateGemma 4B, offline); `madlad` is the basic MADLAD-400 engine and is also the automatic fallback when the local model cannot run.
- "Local model" = `local_model`, default `translategemma-4b-q4km`. Change it only to point at another GGUF model you placed in the translation models folder.
- "Improve with cloud AI models (needs internet and API keys)" = `llm_refine`, default off. Only text, never audio or video, is sent when it is on.
- "Cloud AI only corrects the local translation (fewer tokens, lower quality)" = `llm_correct_only`, default off; with it off the cloud models translate every line from scratch.
- "Double-check risky lines with a second AI model (recommended)" = `llm_review`, default on (section 8).
- The dialog shows "Providers with keys: ..." and the API keys file path; with no keys the status line is "No API keys found: the basic translation was used."

Subtitle sources:
- "Fetch subtitles from the video page (YouTube)" = `fetch_platform_subtitles`, default on. Those subtitles are evidence for the ASR text, never the final output.
- "API keys..." button (Settings): opens the window with one field per cloud AI provider (with a link to get a key); an empty provider is skipped and can be filled later; the keys are written to the key file only. The GitHub build also opens this window once on the first start.
- OpenSubtitles and SubDL API keys: optional, free from the providers' websites, stored in the local database.

Download access: see section 5. Kept series: remove saved series here; removing keeps them in the job history but drops them from the suggestions.

Settings with no dialog control: `name_normalization` (default on, deterministic name normalisation), `log_level` ("INFO"), `model_dir` (empty = `<data root>\models`), `burn_video` (off; also on the main window), `theme` and `ui_language` (both menus).

**5. Cookies and download access**

Some sites refuse a download unless the request carries a signed-in session: YouTube's "confirm you are not a bot" check, Vimeo, age-restricted videos. "Settings > Download access" controls this:
- `cookies_source`: "Do not use cookies" (default), "Use my browser session", "Use a cookies.txt file".
- `cookies_browser`: the browser you are signed in with: firefox, brave, chrome, chromium, edge, opera, vivaldi.
- `cookies_file`: a `cookies.txt` you exported yourself (it must end in `.txt`).
- "Force IPv4 (some networks are refused over IPv6)" = `force_ipv4`, default off.

How it is used: the cookie-less probe always runs first. Cookies are used only when that probe is refused with a sign-in demand (login required, age restriction, bot check), or offers no video, or offers less than 1080p while the cookie probe offers more. A failing cookie probe never breaks a working cookie-less one. The winner is remembered for that URL and the download uses the same access; a cookie-less download refused with a sign-in demand is retried with cookies. When cookies are skipped you get a warning such as "Cookies skipped: they are not needed for this video and can hide HD formats".

The measured fact behind that order (D-080, D-085): on a link that offered 144p-1080p without cookies, YouTube with the user's browser cookies offered only format 18, the progressive 640x360 file, and the old format chain fell through to it silently. Browser cookies can therefore restrict YouTube to 360p for some links while a cookie-less probe offers 1080p.

Privacy: yt-dlp opens the browser profile or your `cookies.txt` itself. The app never copies, stores or logs cookie values; the database remembers only the choice, the browser name and the file path. "Force IPv4" helps when a refusal comes from the network route; after a bot-check or PO-token refusal the app already retries once over IPv4 automatically. Measured limitation: on the video that caused D-073 the IPv4 retry did not help either. If a site still refuses, download the file in your browser and open it here - a local file always works and needs none of this.

**6. Downloads come out at 360p**

The app logs the height that was really downloaded and warns when it is below the request or below what the site offered. Work through this list:
1. Look at the "Video quality" combo right after pasting the link: it lists the heights offered now. If it only shows 360p, the site is restricting this request.
2. Settings > Download access > "Do not use cookies", then paste the link again. If the combo now offers 1080p, cookies were the cause (section 5).
3. Install the JavaScript runtime: `winget install DenoLand.Deno`; a missing Deno is named in the warning.
4. yt-dlp updates itself: at every start (at most every 6 hours) the app downloads a newer verified release into `%LOCALAPPDATA%\AISubtitleStudio\site\`. If you have been offline, start the app once with internet. Sites change and a new release is usually the fix.
5. Check `ffmpeg`. Without it the app cannot merge separate video and audio streams and falls back to a single-file format: "FFmpeg not found: downloading a single-file format (often lower resolution)".
6. Choose the height explicitly instead of "Best available".

Cache behaviour: the download cache key contains the requested quality and `CHAIN_VERSION` (currently 3). A cached 360p file can never satisfy a later 1080p request, and a file fetched with an older format chain is downloaded again. So change the setting and re-run the same link: only the download is redone.

**7. AI providers, quota and timeouts**

Model choice is automatic: every usable model of every provider that has a key is ranked, measured benchmark scores first and name-based estimates kept below the quality floor, with the provider order only breaking ties. Real measurements win over estimates and are written to `<data root>\model_ranking.json`. Token usage per provider goes into the job log and is summarised when the job ends.

`provider_health.json` remembers failures across jobs: a small JSON file of `{key: {"until": epoch seconds, "reason": text}}`, where a key is a route (`provider:model`) or a provider name. Remembered times: a timeout, three failures or a refused key 30 minutes; a daily quota 3 hours; no free quota and a monthly quota 24 hours. At the start of a job these become cool-downs, so a model that was dead ten minutes ago is not tried first again. Delete the file (or wait) to try everything again.

"No free quota" means the key has no free allowance left for that model - Gemini, for example, answers 429 with "limit: 0" - so that model is skipped for 24 hours. A monthly allowance applies to the whole key, so the whole provider is skipped for 24 hours. The job does not fail: the basic translation is kept, the affected lines are flagged, and running the same input later finishes the AI pass.

Timeouts: the read timeout is 60 s (`REQUEST_TIMEOUT_S`), or three times the median of that model's recent successful replies, capped at 150 s (`LONG_TIMEOUT_S`). A measured model at or above the quality floor that times out gets one more try at 150 s before it is dropped and remembered; unmeasured or below-floor models are dropped at the first timeout. A dead good model costs about 3.5 minutes once per job.

To re-try, start the same input again: there is no separate retry button, and completed stages are cached, so only the unfinished part runs. Lines that were never refined carry "basic machine translation (AI unavailable)"; lines from a fallback model carry "translated by a weaker AI model (stronger models were busy)". Messages you may see: "The provider refused the request (check the API key in Settings)." (`error.provider_auth`) and "The provider could not be reached (network error)." (`error.provider_network`).

**8. Double check, ReviewRequired.txt and the review window**

With `llm_review` on (default), risky lines - failed verification, name problems, a weak model, length ratio, a lost negation or question, audio confidence below 0.5 - are translated a second time by the best model of another provider, and a judge request picks A, B or "both_wrong". B replaces A only when it passes the line and name checks; otherwise both candidates are kept and the line is flagged "the second model disagrees (both versions are kept)" at MEDIUM confidence.

`work\ReviewRequired.txt` is written next to the final subtitle. Its first line is "Lines needing review: N of M (approved lines are not listed)". Each entry looks like:

```
#12  00:01:23,400 --> 00:01:25,900  LOW  [untranslated_or_wrong_script, numbers_differ]  audio=0.42
  SRC: <source line>
  TGT: <translation>
  SUGGESTED: <text>            (only when a reviewer suggestion exists)
  WHY: <reason>
  AI DISAGREEMENT (B): <reason>
  A: <first candidate>
  B: <second candidate>
```

Open the review window with the **Review** button after a job, or with **File > Review an episode folder...** and pick the episode folder. The list shows "Lines needing review" or "All lines"; the columns are number, time, status, source, translation and "Why flagged". Buttons: play the line with 5 s of context, pause, previous/next line, previous/next issue, "Use reviewer suggestion", "Retranslate with AI" (only when cloud AI is enabled in Settings), "Re-transcribe span" (needs a local Whisper model; it replaces only the source line), approve and next, split cue, merge with next, shift timing, delete line, find & replace, and save. Shortcuts: Ctrl+Enter approve, Ctrl+S save, Ctrl+Space play, Ctrl+Down / Ctrl+Up next / previous issue, F2 / Shift+F2 next / previous issue, Ctrl+Shift+S split, Ctrl+Shift+M merge, Ctrl+D delete, Ctrl+F and Ctrl+H find & replace.

Edits are kept in `work\review.json`, so the original AI translation is never lost. Saving rewrites the final `<title>.<lang>.srt`, `MasterTranscript.json` and `ReviewRequired.txt`.

Flag codes ("Why flagged") and their meanings:

| Code | Meaning |
| --- | --- |
| empty_translation / empty_cue | translation is empty / empty subtitle |
| untranslated_or_wrong_script | not translated or wrong script |
| numbers_differ | numbers differ from the source |
| suspicious_length, too_short, too_long | length problems; too short or too long on screen |
| repetition_loop, same_as_previous_line, duplicate_cue | repeated words; same as the previous line |
| not_refined_by_ai, weak_ai_model | basic machine translation; a weaker AI model was used |
| name_mismatch | a character name is missing or differs |
| ai_disagreement, reviewer_suggestion | the second model disagrees; a change is suggested (suggestions come from older runs) |
| low_audio_confidence | speech recognition was unsure |
| invalid_timing, overlap | invalid timing; overlaps the previous line |
| reading_speed, line_too_long, too_many_lines | too fast to read (over 20 characters/s); over 42 characters; more than 2 lines |
| latin_punctuation, arabic_indic_digits, three_dots, tatweel, latin_words, extra_whitespace, space_before_punctuation, combined_question_exclamation | cosmetic or punctuation issues, fixed or reported |

Confidence (HIGH, MEDIUM, LOW) comes from the line's flags plus its audio confidence, the mean word probability of its ASR segment: LOW when a severe flag is present or audio confidence is below 0.5; MEDIUM when any other flag is present or audio confidence is below 0.75; HIGH otherwise. Cosmetic flags do not lower confidence. A line needs review when it is not approved and is not HIGH.

**9. Diarization and speakers**

"Detect who is speaking" (`diarization`, default on) runs after transcription on the CPU: sherpa-onnx pyannote segmentation 3.0 plus 3D-Speaker CAM++ embeddings, pinned by SHA-256 and stored under `models\diarization`. It takes a few minutes per episode (a 60 s smoke test with one thread took 20.8 s while a GPU job ran; a measurement on a free CPU is still pending). Master segments get a speaker id ("S1", "S2", ...) which the translation uses for gender and who is addressed. A failure only costs the speaker labels: the job continues with "Speaker detection failed (...); continuing without speaker labels". When two speakers share one cue, the line is written as two dash-prefixed lines (`- ` by default, from the language style guide) for automatic translations; a line you edited keeps your text.

**10. Caches, resume and maintenance**

Each job lives in `<data root>\jobs\<job key>\`, where the key is a hash of the input file fingerprint or the URL. Inside: `input.json`, `media_info.json`, `job.log`, `download\` (partial downloads used by the resume), the stage outputs and `stages\<stage>.json`. The stages are download, audio, transcribe, diarize, subtitles, translate, refine and export. A stage manifest records the cache key, the completed flag and the output files; the stage is skipped when the key matches and its outputs exist. Cache keys chain, so a change in an earlier stage invalidates everything after it.

Long stages append progress to JSON Lines files, so a killed or cancelled job resumes near the last finished piece: `transcribe.<key>.partial.jsonl`, `translate.<key>.partial.jsonl`, `refine.<key>.partial.jsonl`, `diarize.<key>.partial.jsonl` and `redecode.<key>.partial.jsonl`. Interrupted jobs are marked at startup; starting the same input again resumes from the cache, and cancelling keeps all finished work.

To force a re-run: delete `<job key>\stages\<stage>.json` to redo that stage and every later one (the safe way to redo the AI pass), or delete the whole `<data root>\jobs\<job key>\` folder to redo everything, including download and transcription. Never edit the stage output files themselves: the manifests reference them by name, a missing file makes the stage run again, but a stale file with a matching manifest is trusted.

Keep from the output folder: the final `<title>.<lang>.srt` next to the video (players load it automatically), `work\<title>.<source>.source.srt`, `work\MasterTranscript.json`, `work\ReviewRequired.txt`, `work\ProcessingLog.txt`, `work\audio.wav` and `work\sources\` (the reference subtitles used as evidence). Re-running the pipeline rewrites the final SRT from its own result and does not read `work\review.json`; if you re-ran a job after reviewing, open Review again and press Ctrl+S to re-apply your saved edits.

**11. Troubleshooting**

- "No speech was detected in this file." (`error.no_speech`): no segments were produced. Check that the file has speech, try "Difficult audio" > "Always clean the audio", or set the source language manually. "The file has no usable audio" (`error.no_audio`): no audio stream, or it decoded to zero samples; use another file or container.
- Model download failure: models come from Hugging Face on first use and need internet access and free disk space (a download is refused without about 1 GB of margin). An interrupted download leaves no `.download-complete` marker and is fetched again; nothing is corrupted. The log line "Downloading ... model ... (about N MB)" names the model.
- No FFmpeg: downloads fall back to a single-file format (usually low resolution) and burning fails with "FFmpeg was not found; it is needed to create a video with subtitles" (`error.no_ffmpeg`). A build without libass reports "This FFmpeg build cannot draw subtitles (libass is missing)" (`error.no_libass`).
- `error.provider_auth` ("The provider refused the request (check the API key in Settings)."): the key is wrong, expired or unauthorised; fix it in `api_keys.local.json` / `api_keys.json`. The basic translation is kept for the affected lines. `error.provider_network`: the provider could not be reached; check the connection and start the same input again later.
- "The video is unavailable (private, removed or region-blocked)" (`error.video_unavailable`), age restriction (`error.age_restricted`) or geo-block (`error.geo_blocked`): for age restrictions use "Download access" with the browser you are signed in with; a geo-block needs a different network.
- Bot check (`error.youtube_bot_check`) or "This site needs a signed-in session" (`error.login_required`): use "Download access", the browser you are signed in with, or download the file yourself and open it here. "The site refused the video stream" (`error.po_token`): update yt-dlp or use your browser session.
- Subtitles withheld: YouTube withholds some automatic captions without a PO token. It is logged at info level and is not a failure; the app continues with its own speech recognition. `error.rate_limited` (HTTP 429): the app retries after 5 and 20 seconds; if it still fails, wait and retry.
- Out of memory on a small GPU: on CUDA out-of-memory the batch size is halved down to the sequential decoder and transcription continues; a failing GPU plan falls back to the next plan, usually the CPU. Only one GPU model is loaded at a time, deliberately, because the design GPU has 4 GB. Batched decoding is off by default anyway (`batch_size = 0`).
- "No speech recognition model could be loaded" (`error.asr_unavailable`) or "No translation model could be loaded" (`error.mt_unavailable`): missing or incomplete model files; check the models folder, the free disk space and the log. "The local AI model could not run, so the basic MADLAD engine was used" (`status.local_model_failed`): the llama.cpp runtime or the GGUF file failed; translation still works at basic quality.
- Full technical detail is in `<data root>\logs\app.log`; one episode's log is `work\ProcessingLog.txt`.

**12. Known limitations**

- Translation quality is not verified against a human reference: the project has no reference subtitle set, so no accuracy claim is made for the ASR or the AI changes.
- The target of under 3 GB peak RAM for a 100-minute episode is not met: the measured peak with windowed transcription is about 3.68 GB (3.88 GB without windows), dominated by the model and the CUDA libraries.
- Batched decoding is available but off by default: a 15 % speed-up came with 8-10 segments per 5 minutes instead of 51-88, which destroys sentence-level segments.
- Speaker detection has no speed measurement on a free CPU yet, and snapping to scene changes is unit-tested with a fake FFmpeg only, not on a real video.
- The AI double check and the condense step need free cloud quota; without it the basic translation is kept and the lines are flagged for review.

## Running an episode again from scratch

When an episode finishes, the app deletes its working cache, so nothing is left to reuse. To process it again, delete
the episode's folder inside `output` (or leave it - it is overwritten) and start the same link or file again; every
stage runs again, with no restart of the app. A failed or cancelled episode keeps its cache on purpose so it resumes
where it stopped. To keep the cache of finished episodes (for debugging or `tools/evaluate.py`), tick "Keep the
working files after an episode is finished" in Settings.

