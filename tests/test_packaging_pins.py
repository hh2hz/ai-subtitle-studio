"""Packaging hardening (task 6.9): everything the build downloads is pinned, and the packaged self-test stays honest.

The PowerShell helpers in packaging/tools.ps1 are exercised with a real PowerShell child process, because that is
where the shim rejection and the SHA-256 checks live. The pins, the Hugging Face revisions, the pyproject version
and the self-test coverage are read from the files themselves.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import app
from app.main import SELF_TEST_MODULES, SELF_TEST_RESOURCES

ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "packaging" / "build_windows.ps1"
TOOLS_SCRIPT = ROOT / "packaging" / "tools.ps1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
requires_powershell = pytest.mark.skipif(POWERSHELL is None, reason="no PowerShell interpreter on PATH")


# --------------------------------------------------------------------------------------- build script pins

def _assignment(name: str, text: str) -> str:
    match = re.search(rf'^\${name} = "(?P<value>[^"]*)"', text, re.MULTILINE)
    assert match, f"${name} is not assigned in {BUILD_SCRIPT.name}"
    return match.group("value")


def test_build_script_pins_versioned_tool_downloads():
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    for url_variable, hash_variable in (("FfmpegUrl", "FfmpegSha256"), ("DenoUrl", "DenoSha256")):
        url = _assignment(url_variable, text)
        digest = _assignment(hash_variable, text)
        assert SHA256_PATTERN.match(digest), f"${hash_variable} must be a real SHA-256, not a placeholder"
        assert "latest" not in url.lower(), f"${url_variable} must name a versioned release, not a rolling 'latest'"
        assert "/releases/download/" in url and url.endswith(".zip")
    for tool in ("ffmpeg", "ffprobe", "deno"):
        assert f'Copy-Tool -ToolName "{tool}"' in text, f"{tool} is not copied through the pinned Copy-Tool"


# ------------------------------------------------------------------------------- downloaded model revisions

def test_every_model_repository_is_pinned_to_a_commit():
    from app.models import model_manager

    specs = list(model_manager.WHISPER_MODELS.items()) + list(model_manager.TRANSLATION_MODELS.items())
    assert len(specs) >= 7
    for name, spec in specs:
        assert spec.revision, f"model {name} ({spec.repo_id}) has no pinned revision"
        assert COMMIT_PATTERN.match(spec.revision), f"model {name} must be pinned to a 40-character commit"


def test_nvidia_wheels_are_version_pinned():
    from app.services import gpu_runtime

    assert gpu_runtime.PACKAGES
    for name, version in gpu_runtime.PACKAGES:
        assert re.fullmatch(r"\d+(\.\d+)+", version), f"{name} must pin an exact version, not a range"


def test_nvidia_wheel_checksum_mismatch_is_refused(tmp_path, monkeypatch):
    """The expected hash comes from the PyPI JSON API; a wheel that does not match it must never be extracted."""
    import io
    import zipfile

    from app.services import gpu_runtime

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("nvidia/cublas/bin/cublas64_12.dll", b"dll")
    wheel = buffer.getvalue()
    infos = {name: {"url": f"mem://{name}", "sha256": "0" * 64, "size": len(wheel),
                    "filename": f"{name}-win_amd64.whl"} for name, _ in gpu_runtime.PACKAGES}
    monkeypatch.setattr(gpu_runtime, "_wheel_info", lambda name, version: infos[name])

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(gpu_runtime.urllib.request, "urlopen", lambda url, timeout=0: Response(wheel))
    target = tmp_path / "cuda"
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        gpu_runtime.download_all(target=target)
    assert not list(target.rglob("*.whl")) and not list(target.rglob("*.dll"))


# ----------------------------------------------------------------------------------- packaged self-test

def test_self_test_checks_every_shipped_resource():
    resources = ROOT / "app" / "resources"
    shipped = {path.relative_to(resources).as_posix() for path in resources.rglob("*") if path.is_file()}
    assert shipped == set(SELF_TEST_RESOURCES)


def test_self_test_checks_the_new_application_modules():
    required = {"app.core.diarize", "app.core.frames", "app.core.platform_text", "app.core.redecode",
                "app.core.risk", "app.core.windows"}
    assert required <= set(SELF_TEST_MODULES)
    for module in sorted(required):
        importlib.import_module(module)


# -------------------------------------------------------------------------------------- version single source

def test_pyproject_single_sources_the_version():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["dynamic"] == ["version"]
    assert "version" not in data["project"]
    assert data["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "app.__version__"
    setuptools_pyprojecttoml = pytest.importorskip("setuptools.config.pyprojecttoml")
    configuration = setuptools_pyprojecttoml.read_configuration(str(ROOT / "pyproject.toml"))
    assert configuration["project"]["version"] == app.__version__


# --------------------------------------------------------------------------------- packaging/tools.ps1

HARNESS = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # same as packaging\build_windows.ps1; there is no console here
. '__TOOLS__'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$tmp = '__TMP__'
$bin = Join-Path $tmp 'bin'
$downloads = Join-Path $tmp 'downloads'
$real = Join-Path $tmp 'tools'
$shims = Join-Path $tmp 'scoop\shims'
$choco = Join-Path $tmp 'chocolatey\bin'
$store = Join-Path $tmp 'WindowsApps'
$winget = Join-Path $tmp 'WinGet\Links'
$npm = Join-Path $tmp 'node_modules\.bin'
$stub = Join-Path $tmp 'stub'
$source = Join-Path $tmp 'source'
foreach ($dir in @($bin, $downloads, $real, $shims, $choco, $store, $winget, $npm, $stub, $source)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
function New-Binary([string]$Path, [int]$Bytes) {
    [System.IO.File]::WriteAllBytes($Path, (New-Object byte[] $Bytes))
}
$result = [ordered]@{}
$result['reasons'] = [ordered]@{}
$candidates = @(
    @{ key = 'real';        path = (Join-Path $real 'ffmpeg.exe') },
    @{ key = 'scoop';       path = (Join-Path $shims 'ffmpeg.exe') },
    @{ key = 'chocolatey';  path = (Join-Path $choco 'ffmpeg.exe') },
    @{ key = 'windowsapps'; path = (Join-Path $store 'ffmpeg.exe') },
    @{ key = 'winget';      path = (Join-Path $winget 'ffmpeg.exe') },
    @{ key = 'npm';         path = (Join-Path $npm 'ffmpeg.exe') })
foreach ($candidate in $candidates) { New-Binary $candidate.path 300000 }
New-Binary (Join-Path $stub 'ffmpeg.exe') 64          # stub-sized, but outside any shim directory
New-Binary (Join-Path $stub 'faketool.exe') 64
New-Binary (Join-Path $shims 'deno.exe') 64           # a Scoop-style deno shim
Set-Content -LiteralPath (Join-Path $tmp 'wrapper.cmd') -Value '@echo off' -Encoding ascii
foreach ($candidate in $candidates) {
    $result['reasons'][$candidate.key] = [string](Get-ShimReason -Path $candidate.path)
}
$result['reasons']['stub'] = [string](Get-ShimReason -Path (Join-Path $stub 'ffmpeg.exe'))
$result['reasons']['wrapper'] = [string](Get-ShimReason -Path (Join-Path $tmp 'wrapper.cmd'))
$result['reasons']['missing'] = [string](Get-ShimReason -Path (Join-Path $tmp 'not-there.exe'))

$stage = Join-Path $tmp 'stage\deno-x'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Set-Content -LiteralPath (Join-Path $stage 'deno.exe') -Value 'from-zip' -Encoding ascii
$goodZip = Join-Path $source 'deno.zip'
[System.IO.Compression.ZipFile]::CreateFromDirectory((Join-Path $tmp 'stage'), $goodZip)
$goodSha = (Get-FileHash -LiteralPath $goodZip -Algorithm SHA256).Hash
$result['good_sha256'] = $goodSha
Set-Content -LiteralPath (Join-Path $source 'garbage.bin') -Value 'not the pinned archive' -Encoding ascii
$goodUri = ([System.Uri]$goodZip).AbsoluteUri
$garbageUri = ([System.Uri](Join-Path $source 'garbage.bin')).AbsoluteUri
$deadUri = ([System.Uri](Join-Path $source 'nothing-here.zip')).AbsoluteUri

# a real installation on PATH is copied, nothing is downloaded
$env:PATH = $real
Copy-Tool -ToolName 'ffmpeg' -Url $deadUri -ZipName 'ffmpeg.zip' -Sha256 $goodSha -Bin $bin -Downloads $downloads
$result['path_copy'] = (Test-Path -LiteralPath (Join-Path $bin 'ffmpeg.exe'))
$result['path_copy_bytes'] = (Get-Item -LiteralPath (Join-Path $bin 'ffmpeg.exe')).Length

# a shim on PATH is refused and the pinned archive is downloaded, verified and unpacked
$env:PATH = $shims
$shimWarnings = @()
Copy-Tool -ToolName 'deno' -Url $goodUri -ZipName 'deno.zip' -Sha256 $goodSha -Bin $bin -Downloads $downloads -WarningVariable shimWarnings
$result['shim_warning'] = @($shimWarnings | ForEach-Object { $_.ToString() })
$result['shim_body'] = (Get-Content -LiteralPath (Join-Path $bin 'deno.exe') -Raw).Trim()
$result['shim_cached_sha256'] = (Get-FileHash -LiteralPath (Join-Path $downloads 'deno.zip') -Algorithm SHA256).Hash

# a cached archive that no longer matches the pin is thrown away and downloaded again
Set-Content -LiteralPath (Join-Path $downloads 'deno.zip') -Value 'corrupt' -Encoding ascii
Remove-Item -LiteralPath (Join-Path $bin 'deno.exe') -Force
$cacheWarnings = @()
Copy-Tool -ToolName 'deno' -Url $goodUri -ZipName 'deno.zip' -Sha256 $goodSha -Bin $bin -Downloads $downloads -WarningVariable cacheWarnings
$result['cache_warning'] = @($cacheWarnings | ForEach-Object { $_.ToString() })
$result['cache_body'] = (Get-Content -LiteralPath (Join-Path $bin 'deno.exe') -Raw).Trim()

# a download whose hash is not the pinned one must fail loudly and leave nothing behind
$env:PATH = $real
$result['mismatch_message'] = ''
try {
    Copy-Tool -ToolName 'ffprobe' -Url $garbageUri -ZipName 'ffprobe.zip' -Sha256 $goodSha -Bin $bin -Downloads $downloads
} catch { $result['mismatch_message'] = $_.Exception.Message }
$result['mismatch_copied'] = (Test-Path -LiteralPath (Join-Path $bin 'ffprobe.exe'))
$result['mismatch_leftovers'] = @(Get-ChildItem -LiteralPath $downloads -Filter 'ffprobe*' | ForEach-Object { $_.Name })

# a stub outside a shim directory is refused too
$env:PATH = $stub
$stubWarnings = @()
$result['stub_message'] = ''
try {
    Copy-Tool -ToolName 'faketool' -Url $deadUri -ZipName 'faketool.zip' -Sha256 $goodSha -Bin $bin -Downloads $downloads -WarningVariable stubWarnings
} catch { $result['stub_message'] = $_.Exception.Message }
$result['stub_warning'] = @($stubWarnings | ForEach-Object { $_.ToString() })
$result['stub_copied'] = (Test-Path -LiteralPath (Join-Path $bin 'faketool.exe'))

# a missing or placeholder pin stops the build instead of skipping the check
$result['placeholder_message'] = ''
try {
    Copy-Tool -ToolName 'deno' -Url $goodUri -ZipName 'deno.zip' -Sha256 '' -Bin $bin -Downloads $downloads
} catch { $result['placeholder_message'] = $_.Exception.Message }

$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath '__RESULT__' -Encoding UTF8
"""


@requires_powershell
def test_copy_tool_refuses_shims_and_verifies_hashes(tmp_path):
    result_path = tmp_path / "tools-result.json"
    script = (HARNESS.replace("__TOOLS__", str(TOOLS_SCRIPT))
                    .replace("__TMP__", str(tmp_path / "fixtures"))
                    .replace("__RESULT__", str(result_path)))
    completed = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, f"PowerShell harness failed:\n{completed.stdout}\n{completed.stderr}"
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))

    reasons = result["reasons"]
    assert reasons["real"] == "", "a real installation must be accepted"
    assert "shim directory" in reasons["scoop"]
    assert "shim directory" in reasons["chocolatey"]
    assert "shim directory" in reasons["windowsapps"]
    assert "shim directory" in reasons["winget"]
    assert "shim directory" in reasons["npm"]
    assert "stub/launcher" in reasons["stub"]
    assert "wrapper script" in reasons["wrapper"]
    assert reasons["missing"], "a missing file must be reported"

    assert result["path_copy"] is True and result["path_copy_bytes"] == 300000
    assert result["shim_warning"] and "shim directory" in result["shim_warning"][0]
    assert result["shim_body"] == "from-zip", "the pinned archive must be unpacked when the shim is refused"
    assert result["shim_cached_sha256"].lower() == result["good_sha256"].lower()

    assert result["cache_warning"] and result["cache_body"] == "from-zip"

    assert "pinned" in result["mismatch_message"] and "refusing" in result["mismatch_message"]
    assert result["mismatch_copied"] is False and result["mismatch_leftovers"] == []

    assert result["stub_warning"] and "stub/launcher" in result["stub_warning"][0]
    assert result["stub_copied"] is False

    assert "SHA-256" in result["placeholder_message"]
