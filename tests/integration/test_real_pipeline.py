"""Real-model end-to-end run (network for the first model download, real faster-whisper and MADLAD).

Run on the target machine:
    set AISS_TEST_MEDIA=C:\\path\\to\\sample.mp4
    set AISS_TEST_SOURCE=tr          (optional, default auto)
    set AISS_TEST_TARGET=ar          (optional, default ar)
    set AISS_TEST_MODE=balanced      (optional)
    python -m pytest -m integration -s
Models are stored in the normal application models folder so they are downloaded only once.
"""

import json
import os
import threading
from pathlib import Path

import pytest

from app.core.modes import Mode
from app.core.pipeline import JobConfig
from app.services.job_manager import default_pipeline_factory
from app.utils.paths import AppPaths, default_data_root

pytestmark = pytest.mark.integration

MEDIA = os.environ.get("AISS_TEST_MEDIA")
ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


def _arabic_letter_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in ARABIC_RANGES))
    return arabic / len(letters)


@pytest.mark.skipif(not MEDIA, reason="set AISS_TEST_MEDIA to a short media file you have rights to")
def test_real_pipeline(tmp_path):
    paths = AppPaths.from_root(default_data_root()).ensure()
    target = os.environ.get("AISS_TEST_TARGET", "ar")
    config = JobConfig(
        input_path=Path(MEDIA),
        source_language=os.environ.get("AISS_TEST_SOURCE", "auto"),
        target_language=target,
        mode=Mode(os.environ.get("AISS_TEST_MODE", "balanced")),
        output_dir=tmp_path / "out",
    )
    progress = []
    factory = default_pipeline_factory(tmp_path / "jobs", paths.models_dir, paths.root)
    pipeline = factory(config, threading.Event(), lambda *a: progress.append(a), None)
    result = pipeline.run()

    doc = json.loads(Path(result.outputs["master_transcript"]).read_text(encoding="utf-8"))
    print("\nSTATS:", json.dumps(result.stats, indent=2))
    print("WARNINGS:", result.warnings)
    print("OUTPUT:", result.output_dir)
    assert doc["segments"], "no segments transcribed"
    translations = " ".join(u["translation"] for u in doc["translation_units"])
    assert translations.strip()
    if target == "ar":
        ratio = _arabic_letter_ratio(translations)
        print(f"Arabic letter ratio: {ratio:.2f}")
        assert ratio > 0.6
