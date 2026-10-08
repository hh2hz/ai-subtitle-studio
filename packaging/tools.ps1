# Helpers for packaging\build_windows.ps1: pinned tool downloads with SHA-256 verification, and rejection of
# package-manager shims when a tool is picked up from PATH. The file defines functions only (no side effects), so
# it is dot-sourced by the build script and can be loaded on its own by tests\test_packaging_pins.py.

function Assert-PinnedHash {
    <#
    .SYNOPSIS
      Fail loudly when a download has no usable SHA-256 pin.
    .DESCRIPTION
      Nothing is ever bundled unverified: an empty or placeholder hash stops the build instead of silently
      skipping the check.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][AllowEmptyString()][AllowNull()][string]$Sha256
    )
    if ($Sha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw ("No SHA-256 pin for $Label (got '$Sha256'). Downloads are never used unverified: put the real hash of " +
               "the exact release asset in the pin next to this download.")
    }
    return $Sha256.ToLowerInvariant()
}

function Test-PinnedFile {
    <#
    .SYNOPSIS
      $true when the file exists and its SHA-256 equals the pin (comparison is case-insensitive).
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Sha256
    )
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    return ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Sha256)
}

function Get-ShimReason {
    <#
    .SYNOPSIS
      Why this candidate executable must not be bundled, or $null when it looks like a real installation.
    .DESCRIPTION
      Get-Command "ffmpeg.exe" happily returns a package-manager shim, and a shim copied into the installer only
      works on the build machine (it forwards to the real binary, which the user does not have). Three checks:
        1. wrapper scripts (.cmd/.bat/.ps1/.psm1/.vbs/.js) can be resolved through PATHEXT;
        2. shim directories (Scoop, Chocolatey, Microsoft Store aliases, winget links, npm) hold stubs, not
           installations;
        3. a stub/launcher is tiny; the real ffmpeg.exe/ffprobe.exe/deno.exe are tens of megabytes.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [long]$MinRealToolBytes = 262144
    )
    $shimDirectories = @(
        '*\scoop\shims\*',        # Scoop: one <tool>.exe re-exec stub per command
        '*\chocolatey\bin\*',     # Chocolatey shims (C:\ProgramData\chocolatey\bin)
        '*\WindowsApps\*',        # Microsoft Store app execution aliases
        '*\WinGet\Links\*',       # winget puts aliases/symlinks to installed tools here
        '*\node_modules\.bin\*'   # npm/npx shims
    )
    $extension = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
    if ($extension -in @('.cmd', '.bat', '.ps1', '.psm1', '.vbs', '.js')) {
        return "it is a $extension wrapper script, not the tool itself"
    }
    if ($extension -ne '.exe') {
        return "it is not a Windows executable ($extension)"
    }
    # Resolve-Path also expands 8.3 short names, so a shim directory cannot be hidden behind them.
    $full = $Path
    try { $full = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path } catch { }
    foreach ($pattern in $shimDirectories) {
        if ($full -like $pattern) { return "it lives in a package-manager shim directory ($full)" }
    }
    $item = Get-Item -LiteralPath $Path -ErrorAction SilentlyContinue
    if ($null -eq $item) { return "the file does not exist ($Path)" }
    if ($item.Length -lt $MinRealToolBytes) {
        return ("it is only {0} bytes, so it is a stub/launcher and not the real tool" -f $item.Length)
    }
    return $null
}

function Copy-Tool {
    <#
    .SYNOPSIS
      Put one external command line tool into the packaged bin\ directory.
    .DESCRIPTION
      A tool found on PATH is accepted only when it is a real installation; package-manager shims are reported and
      ignored, and the pinned release archive is used instead. A downloaded archive (and a cached one from an
      earlier run) must match $Sha256, and the executable must really be inside it.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$ToolName,    # "ffmpeg", "ffprobe", "deno"
        [Parameter(Mandatory = $true)][string]$Url,         # versioned release asset URL (never "latest")
        [Parameter(Mandatory = $true)][string]$ZipName,     # file name used inside -Downloads
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Sha256,
        [Parameter(Mandatory = $true)][string]$Bin,         # destination directory
        [Parameter(Mandatory = $true)][string]$Downloads    # download cache directory
    )
    $pin = Assert-PinnedHash -Label $ZipName -Sha256 $Sha256
    New-Item -ItemType Directory -Force -Path $Bin | Out-Null

    $found = Get-Command "${ToolName}.exe" -ErrorAction SilentlyContinue
    if ($found -and $found.Source) {
        $reason = Get-ShimReason -Path $found.Source
        if ($reason) {
            Write-Warning "   ignoring $($found.Source) for ${ToolName}: $reason"
        } else {
            Copy-Item -LiteralPath $found.Source -Destination $Bin -Force
            # Shared FFmpeg builds keep their libraries next to the exe.
            Get-ChildItem (Split-Path -Parent $found.Source) -Filter *.dll -ErrorAction SilentlyContinue |
                Copy-Item -Destination $Bin -Force
            Write-Host "   $ToolName copied from $($found.Source)"
            return
        }
    }

    $zip = Join-Path $Downloads $ZipName
    if ((Test-Path -LiteralPath $zip) -and -not (Test-PinnedFile -Path $zip -Sha256 $pin)) {
        Write-Warning "   cached $ZipName does not match the pinned SHA-256; downloading it again"
        Remove-Item -LiteralPath $zip -Force
    }
    if (-not (Test-Path -LiteralPath $zip)) {
        Write-Host "   downloading $Url"
        # No progress bar: it makes a 200 MB download much slower and fails outright when the host has no console
        # (local scope only, so the caller's preference is untouched).
        $ProgressPreference = 'SilentlyContinue'
        $part = "${zip}.part"
        Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        Invoke-WebRequest -Uri $Url -OutFile $part
        if (-not (Test-PinnedFile -Path $part -Sha256 $pin)) {
            $actual = (Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash
            Remove-Item -LiteralPath $part -Force
            throw "$ZipName is $actual but $pin is pinned - refusing to bundle it."
        }
        Move-Item -LiteralPath $part -Destination $zip -Force
    }

    $tmp = Join-Path $Downloads ($ZipName + ".x")
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -LiteralPath $zip -DestinationPath $tmp
    $exe = Get-ChildItem $tmp -Recurse -Filter "${ToolName}.exe" | Select-Object -First 1
    if (-not $exe) { throw "$ToolName.exe is not inside $ZipName - the pinned archive is not what was expected." }
    Copy-Item -LiteralPath $exe.FullName -Destination $Bin -Force
    Get-ChildItem $tmp -Recurse -Filter *.dll | Where-Object { $_.DirectoryName -like "*bin*" } |
        Copy-Item -Destination $Bin -Force
}
