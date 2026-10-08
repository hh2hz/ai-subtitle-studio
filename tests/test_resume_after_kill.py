"""M1 acceptance: kill the process mid-run, then resume without redoing completed work."""

import subprocess
import sys
import time
from pathlib import Path

from app.core.exporter import read_srt
from app.core.modes import Mode
from app.core.pipeline import JobConfig, Pipeline
from app.utils.atomic import load_jsonl, read_json
from tests.fakes import CPU_ASR, CPU_MT, SEGMENT_SECONDS, Factory, FakeAsrEngine, FakeTranslator
from tests.media import make_tone_file

ROOT = Path(__file__).resolve().parents[1]


def test_kill_and_resume(tmp_path):
    media = make_tone_file(tmp_path / "ep.m4a", seconds=40.0)    # 20 fake segments
    jobs, out = tmp_path / "jobs", tmp_path / "out"
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.helpers.run_fake_job", str(jobs), str(media), str(out), "0.3"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 90
        partial = None
        while time.monotonic() < deadline:
            found = list(jobs.glob("*/transcribe.*.partial.jsonl"))
            if found and len(load_jsonl(found[0])) >= 4:
                partial = found[0]
                break
            assert proc.poll() is None, proc.stderr.read().decode(errors="replace")
            time.sleep(0.1)
        assert partial is not None, "child never reached the transcription stage"
    finally:
        proc.kill()                       # hard kill, no cleanup in the child
        proc.wait(timeout=30)

    job_dir = partial.parent
    wav_mtime = (job_dir / "audio.wav").stat().st_mtime_ns
    done_before = len(load_jsonl(partial))
    assert 4 <= done_before < 20

    engine = FakeAsrEngine()
    config = JobConfig(media, "tr", "ar", Mode.BALANCED, output_dir=out)
    result = Pipeline(config, jobs, [CPU_ASR], [CPU_MT],
                      Factory({(CPU_ASR.model, "cpu"): engine}),
                      Factory({(CPU_MT.model, "cpu"): FakeTranslator()})).run()

    assert result.stats["stages"]["audio"]["cached"] is True
    assert (job_dir / "audio.wav").stat().st_mtime_ns == wav_mtime          # audio not re-extracted
    assert engine.calls[0]["offset"] > 0                                     # ASR resumed, not restarted
    assert engine.emitted == 20 - done_before                               # only missing segments decoded
    segments = read_json(result.outputs["master_transcript"])["segments"]
    assert [s["start"] for s in segments] == [i * SEGMENT_SECONDS for i in range(20)]
    assert len(read_srt(result.outputs["source_srt"])) == 20
