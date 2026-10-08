"""The program icon is used by the exe, the installer, the uninstaller and both shortcuts."""

import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICON = ROOT / "app" / "resources" / "icons" / "app.ico"


def test_the_icon_file_has_the_sizes_windows_needs():
    data = ICON.read_bytes()
    count = struct.unpack("<H", data[4:6])[0]
    sizes = {data[6 + 16 * i] or 256 for i in range(count)}

    assert {16, 32, 48, 256} <= sizes


def test_the_installer_and_the_exe_use_the_icon():
    iss = (ROOT / "packaging" / "installer.iss").read_text(encoding="utf-8")
    spec = (ROOT / "packaging" / "aistudio.spec").read_text(encoding="utf-8")

    assert r"SetupIconFile=..\app\resources\icons\app.ico" in iss
    assert "UninstallDisplayIcon={app}\\{#MyAppExe}" in iss
    assert "icons" in spec and "app.ico" in spec


def test_every_shortcut_names_the_icon():
    iss = (ROOT / "packaging" / "installer.iss").read_text(encoding="utf-8")
    lines = [ln for ln in iss.splitlines() if ln.startswith("Name: \"{auto")]

    assert len(lines) == 2
    assert all('IconFilename: "{app}\\{#MyAppExe}"' in ln for ln in lines)
