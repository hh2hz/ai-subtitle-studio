# Builds dist\installer\AI-Subtitle-Studio-Setup.exe on Windows.
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
# Steps: PyInstaller one-folder build -> FFmpeg/FFprobe/Deno into bin\ -> self-test -> Inno Setup installer.
$ErrorActionPreference = "Stop"
# Progress bars slow a 200 MB download down and fail outright when the output is redirected (CI logs).
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Missing $Python - create the virtual environment first (see README)." }
# Copy-Tool, the SHA-256 checks and the package-manager shim rejection live in packaging\tools.ps1.
. (Join-Path $PSScriptRoot "tools.ps1")
$Version = (& $Python -c "import app; print(app.__version__)").Trim()
$Dist = Join-Path $Root "dist\AI Subtitle Studio"
$Bin = Join-Path $Dist "bin"
$Downloads = Join-Path $Root "build\downloads"
New-Item -ItemType Directory -Force -Path $Downloads | Out-Null

Write-Host "== 1/5 Python packages (AI Subtitle Studio $Version)"
& $Python -m pip install --disable-pip-version-check -q -r requirements.txt pyinstaller==6.22.3
if ($LASTEXITCODE) { throw "pip failed" }

Write-Host "== 2/5 PyInstaller"
& $Python -m PyInstaller --noconfirm --clean --distpath dist --workpath build\pyinstaller packaging\aistudio.spec
if ($LASTEXITCODE) { throw "PyInstaller failed" }
New-Item -ItemType Directory -Force -Path $Bin | Out-Null

Write-Host "== 3/5 FFmpeg and Deno"
# Pinned tool downloads: a versioned release asset plus the SHA-256 of exactly that asset, verified before use
# (a cached archive is re-verified too, and a missing or placeholder hash stops the build).
# Update the URL, the file name and the hash together, and only to another versioned release asset.
$FfmpegVersion = "ffmpeg-N-127222-g151814650f-win64-gpl"          # BtbN/FFmpeg-Builds, tag autobuild-2026-10-06-13-06
$FfmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-06-13-06/$FfmpegVersion.zip"
$FfmpegSha256 = "8428c7e0d3c5faf21714a8fa7c9663b78a74fc1ee166777771daee03753f4929"
$DenoVersion = "v2.9.7"
$DenoUrl = "https://github.com/denoland/deno/releases/download/$DenoVersion/deno-x86_64-pc-windows-msvc.zip"
$DenoSha256 = "a0c3101b4158d1dfb7d6a78a7bf0f3de80c96bb423c152beec8beb22786f2238"
Copy-Tool -ToolName "ffmpeg" -Url $FfmpegUrl -ZipName "ffmpeg.zip" -Sha256 $FfmpegSha256 -Bin $Bin -Downloads $Downloads
Copy-Tool -ToolName "ffprobe" -Url $FfmpegUrl -ZipName "ffmpeg.zip" -Sha256 $FfmpegSha256 -Bin $Bin -Downloads $Downloads
Copy-Tool -ToolName "deno" -Url $DenoUrl -ZipName "deno.zip" -Sha256 $DenoSha256 -Bin $Bin -Downloads $Downloads
$filters = & (Join-Path $Bin "ffmpeg.exe") -hide_banner -filters 2>$null | Select-String " subtitles "
if (-not $filters) { Write-Warning "The bundled FFmpeg has no libass 'subtitles' filter: 'Save video with subtitles' will not work." }

Write-Host "== 4/5 Self-test of the packaged app"
$Report = Join-Path $Root "build\self-test.txt"
# The self-test takes seconds. A windowed exe that hits an error shows a message box nobody can click on a build
# machine, so it is waited for with a time limit; on a timeout the text of its windows and the partial report are
# printed (that is the error message), then the process is killed.
Remove-Item $Report -ErrorAction SilentlyContinue
$p = Start-Process -FilePath (Join-Path $Dist "AISubtitleStudio.exe") -ArgumentList "--self-test", "`"$Report`"" -PassThru
$null = $p.Handle
if (-not $p.WaitForExit(240000)) {
    Write-Warning "The self-test did not finish within 4 minutes. What the program shows on screen:"
    try {
        Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
        $own = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::ProcessIdProperty, $p.Id)
        $windows = [System.Windows.Automation.AutomationElement]::RootElement.FindAll([System.Windows.Automation.TreeScope]::Children, $own)
        foreach ($w in $windows) {
            Write-Host ("   window: " + $w.Current.Name)
            foreach ($e in $w.FindAll([System.Windows.Automation.TreeScope]::Descendants, [System.Windows.Automation.Condition]::TrueCondition)) {
                if ($e.Current.Name) { Write-Host ("      " + $e.Current.Name) }
            }
        }
    } catch { Write-Warning "Cannot read the windows: $_" }
    if (Test-Path $Report) { Write-Host "Partial self-test report:"; Get-Content $Report }
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    throw "Self-test timed out (see above)"
}
if (Test-Path $Report) { Get-Content $Report }
if ($p.ExitCode -ne 0) { throw "Self-test failed (see above)" }

Write-Host "== 5/5 Installer (Inno Setup)"
$Iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
          "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) {
    Write-Host "   Inno Setup not found: installing it with winget"
    winget install --id JRSoftware.InnoSetup -e --silent --accept-package-agreements --accept-source-agreements
    $Iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
              "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $Iscc) { throw "Inno Setup 6 is required: https://jrsoftware.org/isdl.php" }
}
& $Iscc "/DMyAppVersion=$Version" packaging\installer.iss
if ($LASTEXITCODE) { throw "Inno Setup failed" }
$Setup = Get-ChildItem (Join-Path $Root "dist\installer") -Filter "*.exe" | Sort-Object LastWriteTime | Select-Object -Last 1
Write-Host ""
Write-Host "Installer: $($Setup.FullName) ($([math]::Round($Setup.Length / 1MB)) MB)"
