# AI Subtitle Studio

A Windows desktop app that turns a video (a local file **or** a link) into subtitles in another language, using
**free, local, open models** by default:

```
video / URL -> audio -> speech recognition -> translation -> .srt  (+ optional subtitle burn-in)
```

Everything runs on your computer. No account, no subscription and no cloud service is required. Cloud LLM
providers are optional and free of charge (bring your own free API key).

<p align="center">
  <a href="https://github.com/hh2hz/ai-subtitle-studio/releases/latest/download/AI-Subtitle-Studio-Setup.exe">
    <img alt="Download the Windows installer" src="https://img.shields.io/badge/Download-Windows_installer-2ea44f?style=for-the-badge">
  </a>
  <a href="LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-54dfcf?style=for-the-badge">
  </a>
  <a href="https://www.virustotal.com/">
    <img alt="VirusTotal: scan pending" src="https://img.shields.io/badge/VirusTotal-scan_pending-lightgrey?style=for-the-badge">
  </a>
</p>

<p align="center">
  Windows 10/11, 64-bit &nbsp;|&nbsp; <a href="https://github.com/hh2hz/ai-subtitle-studio/releases">Installer and release notes</a> &nbsp;|&nbsp; <a href="#install-windows-installer">Install guide</a>
</p>

* Interface: English and Arabic (right-to-left).
* Keyboard, mouse and screen-reader friendly Qt (PySide6) UI.
* **You are responsible for having the rights to any content you process.**

---

## Contents

- [Install (Windows installer)](#install-windows-installer)
- [Features](#features)
- [Requirements](#requirements)
- [Quick start (from source)](#quick-start-from-source)
- [Free API keys (optional)](#free-api-keys-optional)
- [Sites that ask for a login (cookies)](#sites-that-ask-for-a-login-cookies)
- [Output layout](#output-layout)
- [Reviewing a translation](#reviewing-a-translation)
- [Tests](#tests)
- [Building the Windows installer](#building-the-windows-installer)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [Privacy](#privacy)
- [License](#license)
- [Author](#author)

---

## Install (Windows installer)

1. **[Download `AI-Subtitle-Studio-Setup.exe`](https://github.com/hh2hz/ai-subtitle-studio/releases/latest/download/AI-Subtitle-Studio-Setup.exe)** (about 230 MB; always the newest release).
2. Run it. It installs for the current user only (no administrator rights), adds a Start-menu entry and offers an optional desktop shortcut. FFmpeg and Deno are bundled; Python is not needed.
3. The installer is **not code-signed**, so Windows SmartScreen may show "Windows protected your PC": choose *More info -> Run anyway*.
4. On the first start the app checks its libraries, then asks for the optional cloud AI keys (you can skip it and add them later in *Settings -> API keys...*).
5. Models (about 3-6 GB) download on first use.

**Verify the download (v1.0.0).**

| | |
|---|---|
| File | `AI-Subtitle-Studio-Setup.exe` (228.5 MB) |
| SHA-256 | published with the release (rebuild in progress) |
| VirusTotal | scan of the rebuilt installer pending |

Compare the hash with `certutil -hashfile AI-Subtitle-Studio-Setup.exe SHA256`. A clean scan lowers the risk but is not a guarantee, and an unsigned installer can still trigger SmartScreen.

**Updating.** The program looks for a newer release on GitHub every time it starts (switch it off in *Settings -> Check for a newer version when the program starts*; *Help -> Check for updates...* checks on demand). When one exists you are asked; *Update now* downloads the installer, verifies its SHA-256 against the value GitHub publishes for the release, closes the program, updates it in place and starts it again. Settings, API keys, history and models are kept. A source checkout only opens the release page.

---

## Features

| Area | What it does |
|---|---|
| Input | A local media file, or a link to thousands of sites through `yt-dlp` (YouTube, Vimeo, ...) |
| Download | Chooses the best available quality, prefers direct streams over HLS, probes **without** cookies first (cookies can hide HD), falls back to your browser session only when a site demands a sign-in, retries transient HTTP 403s, warns when the result is lower than requested |
| Speech recognition | `faster-whisper` (`large-v3` on a GPU, `large-v3-turbo` on the CPU), VAD, batched decoding option, ~10-minute windows, resume after a crash, optional audio cleaning, optional re-decode of low-confidence spans |
| Diarization | Sherpa-ONNX speaker segmentation + embeddings (CPU), speaker labels, `- ` dash lines for dialogues, optional speaker context in the translation |
| Translation | Local `TranslateGemma` (default) or MADLAD-400; optional cloud LLMs with your own free keys, glossary of character names, per-language style guides, reading-speed repair, grammar-aware line breaks |
| Quality control | Deterministic checks per line (numbers, script, repetition, length), name consistency against a series glossary, subtitle-timing QA (17 cps, 42 characters), a confidence category per line, and an optional "double check" where a second model re-translates only risky lines |
| Review | A review window per episode: play a line in context, edit, approve, re-translate or re-transcribe a single line, keep or drop the AI suggestion |
| Series | Series/season/episode detection from the link or file name, TVMaze suggestions with posters, a kept-series list, per-series glossary |
| Output | `<title>.<lang>.srt` next to the video, `MasterTranscript.json`, `ReviewRequired.txt`, `ProcessingLog.txt`, optional video with burned-in subtitles |
| Other | Batch queue, ETA, job history, resumable stages with content-addressed caches, provider health memory, secrets never logged |

## Requirements

* Windows 10/11 (64-bit). Linux works for development and tests; the installer is Windows-only.
* Python **3.13 or 3.14** (3.13 is what the CI build and the tests use; 3.14 is used on the author's machine).
* ~3–6 GB of disk for models, plus the video you process.
* Optional NVIDIA GPU (CUDA 12) for fast transcription; the CPU works too.
* FFmpeg **and** Deno on `PATH` when you run from source —
  `winget install Gyan.FFmpeg` and `winget install DenoLand.Deno`. The installer ships both.

## Quick start (from source)

```bat
git clone https://github.com/hh2hz/ai-subtitle-studio.git
cd ai-subtitle-studio

py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt

REM Optional, NVIDIA GPU only (~1.2 GB of CUDA libraries):
pip install -r requirements-gpu.txt

python -m app.main
```

Models download on first use into `%LOCALAPPDATA%\AISubtitleStudio\models`
(Whisper `large-v3-turbo` ~1.6 GB; TranslateGemma 4B ~2.5 GB with the llama.cpp runtime).

Command line: `python -m app.main --data-dir <path>` (or the `AISS_DATA_DIR` environment variable) changes the data
folder, `--smoke-test` starts and exits immediately, `--self-test <report>` checks a packaged installation.

**If the data folder is not writable** (a sandboxed or restricted shell), the app says so and falls back to
`<project>\.appdata` (or a folder under `%TEMP%`) instead of failing, and keeps reading the models from the normal
folder. Starting from a normal terminal or the installed shortcut uses `%LOCALAPPDATA%\AISubtitleStudio`.

### Automatic library check (nothing to type)

Every start begins with a short window, **"Checking libraries for updates..."**, and the user never runs a command:

* **Pinned libraries** (`requirements.txt`) that are missing or have the wrong version are repaired by the app itself
  (`pip`, from source runs only). It never upgrades them past their pins - the pinned versions are the tested ones.
* **yt-dlp and yt-dlp-ejs** are kept at their newest release, because YouTube breaks old versions every few weeks. A
  newer release is downloaded from PyPI, checked against PyPI's SHA-256, unpacked into
  `%LOCALAPPDATA%\AISubtitleStudio\site\` and import-tested in a separate process before it is used. This also works
  in the installed build. The lookup runs at most every 6 hours and takes at most about 8 seconds when offline.
* The window closes by itself when everything is fine. **An error message appears only if something is still wrong
  after the repair** (for example a library that cannot be installed, or FFmpeg missing from the installation), with
  *Quit* and *Continue anyway*. Being offline is not an error: the installed versions are used.
* To switch the check off, set the `dependency_check` setting to `false`.

## Free API keys (optional)

Cloud AI is **off by default**. If you turn it on (Settings -> "Improve with cloud AI models", or the `llm_refine` setting), the app
uses free tiers of several providers and switches to the next one when a key runs out of quota. On the **first
start** the app opens an **API keys** window with one field and a "Get a key" link per provider: paste the keys you
have and press Save. A provider left empty is skipped (it stays in the program and can be added later). Open the same
window any time from **Settings -> API keys...**. One key is enough; more keys mean fewer interruptions.

The window writes **`api_keys.local.json`** (next to the app; git-ignored) or, in the installed app,
**`%LOCALAPPDATA%\AISubtitleStudio\api_keys.json`**. You can still create that file by hand from
[`api_keys.example.json`](api_keys.example.json).

| Provider | Where to get the key | Notes |
|---|---|---|
| `gemini` | <https://aistudio.google.com/apikey> | Google AI Studio, strong free tier |
| `groq` | <https://console.groq.com/keys> | very fast, free tier |
| `nvidia` | <https://build.nvidia.com/settings/api-keys> | NVIDIA NIM, free credits |
| `cloudflare` | <https://dash.cloudflare.com/profile/api-tokens> | also needs `account_id` (Workers AI) |
| `zai` | <https://z.ai/manage-apikey/apikey-list> | Z.ai (GLM) |
| `cohere` | <https://dashboard.cohere.com/api-keys> | trial key |
| `mistral` | <https://console.mistral.ai/api-keys> | free tier |
| `openrouter` | <https://openrouter.ai/keys> | free `:free` models |
| `huggingface` | <https://huggingface.co/settings/tokens> | Inference Providers router |
| `modelscope` | <https://modelscope.cn/my/myaccesstoken> | ModelScope access token |
| `ollama` | <https://ollama.com/settings/keys> | Ollama Cloud |
| `requesty` | <https://app.requesty.ai/api-keys> | Router |
| `aihubmix` | <https://aihubmix.com/token> | AIHubMix |
| `siliconflow` | <https://cloud.siliconflow.cn/account/ak> | SiliconFlow |
| `dashscope` | <https://modelstudio.console.alibabacloud.com/> | Alibaba Model Studio (international endpoint) |
| `vercel` | <https://vercel.com/dashboard> | Vercel AI Gateway |
| `tencent` | <https://console.cloud.tencent.com/> | Tencent MaaS token hub |
| `llm7` | <https://docs.llm7.io/> | free tokens |
| `kilo` | <https://kilo.ai/docs/getting-started/setup-authentication> | Kilo Code gateway |
| `routeway` | <https://routeway.ai/> | sign in and create a key |
| `moark` | <https://moark.ai/docs/organization/access-token> | MoArk access token |
| `tokenharbor` | <https://tokenharbor.ai/> | sign in and create a key |
| `nous` | <https://portal.nousresearch.com/> | Nous Research portal |
| `sealion` | <https://docs.sea-lion.ai/> | SEA-LION (AI Singapore) |
| `siliconflow` | see above | |
| `aionlabs` | <https://api.aionlabs.ai/> | create the key in your account there |
| `llmtech` | <https://llmtech.eu/> | create the key in your account there |
| `llmtr` | <https://llmtr.com/> | create the key in your account there |
| `freeinference` | <https://freeinference.org/> | create the key in your account there |
| `ovh` | *(no key)* | OVH AI Endpoints answers anonymous callers |
| `unturf` | *(no key)* | unturf Hermes answers anonymous callers |

Two **subtitle** sources need a key entered in **Settings**, not in the JSON file:

| Source | Where to get the key |
|---|---|
| OpenSubtitles | <https://www.opensubtitles.com/en/consumers> |
| SubDL | <https://subdl.com/api-doc> |

Check which providers answer from your machine (needs keys and network):

```bat
python -m pytest tests/integration/test_real_llm_api.py -m integration -s
```

Never commit a filled key file: `api_keys.local.json`, `api_keys.json` and `*_api_key` settings are git-ignored,
the keys are never written to the database, and the app never logs a key or a cookie value.

## Sites that ask for a login (cookies)

Some sites refuse a download unless the request carries a signed-in session (YouTube's "confirm you are not a
bot", age-restricted videos). **Settings -> Download access** can use your own browser session ("Use my browser
session" + the browser you are signed in) or an exported `cookies.txt`. `yt-dlp` reads the cookies itself: this app
never copies, stores or logs them, and only your choice is remembered.

Measured on this project: browser cookies made YouTube offer **only 640x360** for one link where the cookie-less
probe offered up to 1080p. The app therefore probes **without** cookies first and uses them only when a site
insists. If a site still refuses, download the file in your browser and open it here - a local file always works
and needs none of this.

## Output layout

```
output/<video title>/
    <video title>.mp4                 the downloaded or copied video
    <video title>.ar.srt              the final subtitle (players pick it up automatically)
    work/
        audio.wav                     extracted audio (16 kHz mono)
        MasterTranscript.json         segments, words, confidence, speakers, QA flags
        ReviewRequired.txt            lines that need a human look, with the reason
        ProcessingLog.txt             per-stage report of the last run
        <title>.tr.source.srt         source-language subtitle
```

## Reviewing a translation

Press **Review** (or *File -> Review an episode folder*). Lines that need attention are listed with their reason:
play the line with 5 s of context, edit it, approve (Ctrl+Enter), jump to the next issue (Ctrl+Down), retranslate
one line with AI (F2), re-transcribe one line from the extracted audio (Shift+F2), save (Ctrl+S). Saving rewrites
the final subtitle next to the video.

## Tests

```bat
python -m pytest                     REM offscreen Qt, no network, no models
python -m pytest -m integration -s   REM needs real models/network (see below)
```

Qt tests run headless (`QT_QPA_PLATFORM=offscreen`). Integration tests need media and a source language:

```bat
set AISS_TEST_MEDIA=C:\path\to\sample.mp4
set AISS_TEST_SOURCE=tr
python -m pytest -m integration -s
```

## Building the Windows installer

Locally:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
```

It installs PyInstaller, builds a one-folder app, downloads **pinned** FFmpeg/FFprobe/Deno (versioned URLs with
SHA-256, package-manager shims refused), runs the packaged self-test and builds an Inno Setup installer into
`dist\installer\`.

On GitHub: the workflow [`.github/workflows/build-windows-installer.yml`](.github/workflows/build-windows-installer.yml)
does the same on a Windows runner - start it from *Actions -> Build Windows installer -> Run workflow*, download the
artifact, or push a tag (`v1.0.0`) to attach the installer to a GitHub release. The installer is not code-signed, so Windows SmartScreen may show "Windows protected your PC": choose *More info -> Run anyway*.

See
[docs/PUBLISHING.md](docs/PUBLISHING.md) for the whole publishing checklist.

## Limitations

* Windows only for the installer; Linux works for development and tests.
* Quality depends on the audio and on the language pair. It was tuned and tested mostly on Turkish -> Arabic; other pairs work but have had less testing.
* Local translation models are small enough for a 4 GB GPU or the CPU, so they are weaker than large cloud models. Cloud AI uses free tiers, which can run out of quota, change or disappear without notice.
* Speech recognition and translation can be wrong. Use the review window before publishing a subtitle.
* The installer is unsigned (SmartScreen warning). Automatic library repair with `pip` works from source runs only; the installed build updates only yt-dlp by itself.
* Some sites refuse downloads without a signed-in session; see the cookies section.

## Documentation

* [docs/USER_GUIDE.md](docs/USER_GUIDE.md) - every setting, every folder, troubleshooting.
* [docs/DECISIONS.md](docs/DECISIONS.md) - why the pipeline works the way it does (measurements included).
* [docs/PUBLISHING.md](docs/PUBLISHING.md) - how to publish this repository and build the installer.

## Privacy

* Audio, video and subtitles stay on your computer. Only when cloud AI is switched on is **text** sent to the
  provider you chose.
* API keys are entered in the **API keys** window (first start, or Settings -> API keys...) and stored in `api_keys.local.json` / `api_keys.json` (never in the database); they are never logged.
* The update check is one plain request to `api.github.com` (no personal data, no identifier) and can be switched off; nothing is installed without your confirmation.
* Cookie files are only read by `yt-dlp`; the app passes a browser name or a file path and nothing else.
* `provider_health.json` in the data folder remembers which provider failed and for how long - no keys, no text.

## License

The source code is released under the license in [LICENSE](LICENSE). Models, FFmpeg, Deno and the API providers have their own licenses and terms; the
installer bundles FFmpeg (GPL build) and Deno, and the app links PySide6 (LGPL-3.0). You are responsible for
complying with them and for having the rights to the content you process.

## Author

Ali Hubail ([@hh2hz](https://github.com/hh2hz)).
