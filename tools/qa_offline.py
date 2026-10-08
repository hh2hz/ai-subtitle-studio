"""Offline check of the timing and line-break changes of phase 5 on a stored job (no network, no model).

    python tools/qa_offline.py <job folder or refined.*.json> [--language ar]

Re-times the translated units of a finished job twice with the final text of each line: A = the old rules (fixed 42
characters, extension only, no grammar-aware breaks) and B = the new ones (style-guide limits, reading-speed repair,
grammar-aware breaks). Both results are counted against the same limits, so the numbers are comparable.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import finalize  # noqa: E402
from app.core.exporter import Cue, fix_overlaps, polish_timing, wrap_text  # noqa: E402
from app.core.subtitle_qa import QaLimits, check_cues  # noqa: E402


def load(path: Path) -> dict:
    if path.is_dir():
        path = max(path.glob("refined.*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(path.read_text(encoding="utf-8"))


def count(cues: list[Cue], limits: QaLimits) -> Counter:
    flags = Counter()
    for cue_flags in check_cues(cues, limits):
        flags.update(cue_flags)
    flags["cues"] = len(cues)
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    parser.add_argument("--language", default=None)
    args = parser.parse_args()
    doc = load(args.job)
    language = args.language or doc.get("target_language") or "ar"
    units = doc["units"]
    limits = QaLimits.for_language(language)
    new_units = copy.deepcopy(units)
    new_cues = finalize.finalize_units(new_units, [], language)
    old_cues = polish_timing(fix_overlaps(
        [Cue(u["start"], u["end"], wrap_text(u["final_text"]), u["id"]) for u in new_units if u["final_text"].strip()]))
    for label, cues in (("A old rules", old_cues), ("B new rules", new_cues)):
        c = count(cues, limits)
        print(f"{label}: cues={c['cues']} reading_speed={c['reading_speed']} line_too_long={c['line_too_long']} "
              f"too_short={c['too_short']} overlap={c['overlap']} too_many_lines={c['too_many_lines']} "
              f"(limits: {limits.max_cps:g} cps, {limits.max_line_chars} chars)")
    print(f"B merged cues: {sum(len(c.absorbed or []) for c in new_cues)}")


if __name__ == "__main__":
    main()
