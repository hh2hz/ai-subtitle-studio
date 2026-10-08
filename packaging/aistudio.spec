# PyInstaller spec for AI Subtitle Studio (one-folder build). Run through packaging\build_windows.ps1.
# The NVIDIA GPU libraries are NOT bundled (downloaded by the app on first use, D-043); FFmpeg/Deno are copied
# into dist\AI Subtitle Studio\bin by the build script.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
datas = [(str(ROOT / "app" / "resources"), "app/resources")]
datas += collect_data_files("faster_whisper")          # Silero VAD model
datas += collect_data_files("anyascii")                # romanization tables
datas += collect_data_files("yt_dlp_ejs")              # YouTube challenge solver scripts
binaries = collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("sherpa_onnx")        # voice separation (onnxruntime inside sherpa-onnx)
hiddenimports = collect_submodules("sherpa_onnx") + collect_submodules("yt_dlp") + collect_submodules("yt_dlp_ejs") + ["PySide6.QtMultimedia",
                                                                                   "PySide6.QtMultimediaWidgets"]

a = Analysis(
    [str(ROOT / "app" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["nvidia", "tkinter", "pytest", "pytestqt", "IPython", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="AISubtitleStudio",
    icon=str(ROOT / "app" / "resources" / "icons" / "app.ico"),
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="AI Subtitle Studio", upx=False)
