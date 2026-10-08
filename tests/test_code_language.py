"""Arabic text is only allowed in ar.json and test fixtures."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARABIC = re.compile("[\\u0600-\\u06FF\\u0750-\\u077F\\u08A0-\\u08FF\\uFB50-\\uFDFF\\uFE70-\\uFEFF]")
SUFFIXES = {".py", ".md", ".json", ".toml", ".txt", ".cfg", ".ini"}
# Only project sources are scanned; user data such as output folders is not.
SCAN_DIRS = ("app", "tests", "docs")
SKIP_DIRS = {"__pycache__", "fixtures"}


def _files():
    for path in ROOT.iterdir():
        if path.is_file():
            yield path
    for name in SCAN_DIRS:
        yield from (p for p in (ROOT / name).rglob("*") if p.is_file())


def test_no_arabic_outside_allowed_files():
    offenders = []
    for path in _files():
        if path.suffix not in SUFFIXES or path.name == "ar.json":
            continue
        if SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        if ARABIC.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders
