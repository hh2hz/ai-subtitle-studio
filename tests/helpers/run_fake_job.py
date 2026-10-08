"""Run a pipeline with a slow fake ASR engine in a separate process (killed by a test)."""

import sys
from pathlib import Path

from app.core.modes import Mode
from app.core.pipeline import JobConfig, Pipeline
from tests.fakes import CPU_ASR, CPU_MT, Factory, FakeAsrEngine, FakeTranslator


def main(jobs_dir: str, media: str, out_dir: str, delay: str) -> int:
    config = JobConfig(Path(media), "tr", "ar", Mode.BALANCED, output_dir=Path(out_dir))
    pipe = Pipeline(
        config, Path(jobs_dir), [CPU_ASR], [CPU_MT],
        Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine(delay=float(delay))}),
        Factory({(CPU_MT.model, "cpu"): FakeTranslator()}),
    )
    pipe.run()
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
