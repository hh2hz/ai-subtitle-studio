"""The installer script must stay within what the 32-bit Inno compiler can allocate (it ran out of memory once)."""

import re
from pathlib import Path

ISS = Path(__file__).resolve().parents[1] / "packaging" / "installer.iss"


def _setting(name):
    match = re.search(rf"^{name}=(.+)$", ISS.read_text(encoding="utf-8"), re.M)
    return match.group(1).strip() if match else None


def test_compression_is_not_the_memory_hungry_ultra_level():
    assert _setting("Compression") == "lzma2/max"


def test_compression_runs_in_a_separate_process_with_few_threads():
    assert _setting("LZMAUseSeparateProcess") == "yes"
    assert int(_setting("LZMANumBlockThreads")) <= 2


def test_the_installer_file_name_has_no_version_so_the_download_link_never_goes_stale():
    assert _setting("OutputBaseFilename") == "AI-Subtitle-Studio-Setup"
