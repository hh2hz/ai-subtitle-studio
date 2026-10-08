# Publishing AI Subtitle Studio on GitHub

Everything below is meant to be run once, from this folder. Commands are PowerShell unless marked otherwise.

## 1. Before the first commit - check for secrets

```powershell
# no key file may exist here (the example file is fine)
Get-ChildItem -Recurse -Force -Include api_keys.local.json, api_keys.json, *.db, *.log |
  Where-Object { $_.FullName -notmatch "\.venv|\\build\\|\\dist\\" }

# no username or personal path anywhere in the sources
Get-ChildItem -Recurse -File -Include *.py,*.md,*.json,*.ps1,*.bat,*.toml,*.txt |
  Where-Object { $_.FullName -notmatch "\.venv|\\build\\|\\dist\\|__pycache__" } |
  Select-String -Pattern "<your-windows-username>|<your-real-name>|<your-github-user>" -List

# no obvious key material: long random-looking strings in the sources
Get-ChildItem -Recurse -File -Include *.py,*.md,*.json,*.txt,*.ps1 |
  Where-Object { $_.FullName -notmatch "\.venv|__pycache__" } |
  Select-String -Pattern "[A-Za-z0-9_-]{40,}"
```

All three commands must print nothing. `api_keys.example.json` contains only the text
`enter your api key here`, so it is safe to publish - and GitHub's own secret scanning (step 4) is the final net.

## 2. Add a license

The repository has no `LICENSE` file yet, because the license is your decision. GitHub shows "no license" for a
repository without one, and other people may not legally reuse the code.

* Open source (most common): create `LICENSE` with the MIT text and put your name and the year in the copyright
  line, or pick another license at <https://choosealicense.com/>.
* Then update the "License" section of `README.md` if you chose something other than MIT.

A short MIT template:

```
MIT License

Copyright (c) <year> <your name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Bundled third-party pieces keep their own licenses: FFmpeg (GPL build from BtbN), Deno (MIT), PySide6 (LGPL-3.0),
the Whisper/TranslateGemma/MADLAD models and every API provider's terms. Mention them in the `README.md`
"License" section (it already does) and never ship a GPL-incompatible combination.

## 3. Run the tests once on a clean checkout

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:QT_QPA_PLATFORM = "offscreen"
.venv\Scripts\python.exe -m pytest -q
```

## 4. Create the GitHub repository and push

```powershell
git init -b main
git add .
git status                      # look at the list once more: no key file, no output/, no evaluation/
git commit -m "AI Subtitle Studio 1.0.0"
git remote add origin https://github.com/<your-user>/ai-subtitle-studio.git
git push -u origin main
```

If you prefer the web UI: create an empty repository on GitHub (no README, no .gitignore), then follow the
"push an existing repository" commands GitHub shows.

Repository settings worth changing right away:

* **Description**: "Local, free subtitle production for Windows: speech recognition, translation and review, with
  optional free cloud LLMs."
* **Topics**: `subtitles`, `speech-recognition`, `whisper`, `translation`, `pyside6`, `ffmpeg`, `yt-dlp`.
* **Settings -> Actions -> General**: leave "Allow all actions" (the workflow uses official GitHub actions only).
* **Settings -> Security -> Secret scanning**: enable push protection, so a key can never be pushed by accident.

## 5. Build the Windows installer on GitHub

The workflow `.github/workflows/build-windows-installer.yml` does everything `packaging/build_windows.ps1` does on
a `windows-latest` runner: it creates `.venv`, installs Inno Setup with Chocolatey, builds with PyInstaller,
downloads the **pinned** FFmpeg/FFprobe/Deno archives (SHA-256 checked, package-manager shims refused), runs the
packaged self-test and builds the installer.

1. **Actions -> Build Windows installer -> Run workflow** (branch `main`).
2. When it finishes, open the run and download the **AI-Subtitle-Studio-Setup** artifact.
3. To publish a version: `git tag v1.0.0` and `git push origin v1.0.0`. The same workflow then creates the release
   and attaches the installer to it.

### What only a real Windows machine can verify

The workflow cannot check everything, so do this once locally before you announce the release:

1. Install the produced `AI-Subtitle-Studio-Setup.exe` on a machine **without** Python and start it.
2. Make sure `ffmpeg -version` and `deno --version` work from the app (the installer puts them in `bin\`).
3. Let the app download the Whisper model on first use, then run one short clip end to end.
4. Code signing: the installer is unsigned, so Windows SmartScreen will warn. Buy a code-signing certificate and
   sign `AISubtitleStudio.exe` and the setup (a signed release is what makes "unknown publisher" disappear).
5. Check that the app writes to `%LOCALAPPDATA%\AISubtitleStudio` (database, logs, models) and not into
   `Program Files`.

## 6. Releasing later

```powershell
# bump the single-sourced version
# app/__init__.py -> __version__ = "1.1.0"
git commit -am "Release 1.1.0"
git tag v1.1.0
git push origin main --tags
```

The tag triggers the installer build and the release. Keep `docs/DECISIONS.md` and `README.md` in step with
behaviour changes, and never commit a filled `api_keys.local.json` (it is git-ignored, and secret scanning is on).
