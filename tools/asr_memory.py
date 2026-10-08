"""Peak memory of transcribing a long episode: one call over the whole audio vs 10-minute windows (D-096).

    python tools/asr_memory.py --minutes 100 --mode windows|whole [--batch 8]

The audio is the evaluation clips repeated up to the requested length (real speech, real VAD workload). The peak
is the process peak working set (Windows) from psutil; the figure includes the model and CUDA libraries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from app.core.audio_processor import SAMPLE_RATE, load_wav  # noqa: E402
from app.core.transcription import AsrOptions, FasterWhisperEngine  # noqa: E402
from app.core.windows import plan_windows  # noqa: E402


def peak_gb() -> float:
    info = psutil.Process().memory_info()
    return round(getattr(info, "peak_wset", info.rss) / 2**30, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=100.0)
    parser.add_argument("--mode", choices=("windows", "whole"), required=True)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()

    from app.models.model_manager import ModelManager
    from app.utils.cuda_setup import register_nvidia_dll_dirs
    from app.utils.paths import AppPaths, default_data_root

    paths = AppPaths.from_root(default_data_root()).ensure()
    register_nvidia_dll_dirs()
    clips = [load_wav(p) for p in sorted((ROOT / "evaluation" / "media").glob("*.wav"))]
    target = int(args.minutes * 60 * SAMPLE_RATE)
    audio = np.concatenate(clips * (target // sum(len(c) for c in clips) + 1))[:target]
    del clips
    after_load = peak_gb()
    engine = FasterWhisperEngine(ModelManager(paths.models_dir).ensure("whisper", args.model), "cuda",
                                 "int8_float16", replace(AsrOptions(beam_size=5), batch_size=args.batch))
    began = time.monotonic()
    count = 0
    if args.mode == "whole":
        for _ in engine.transcribe(audio, "tr", 0.0, None):
            count += 1
    else:
        for start, end in plan_windows(audio):
            for _ in engine.transcribe(audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)], "tr", start, None):
                count += 1
    elapsed = time.monotonic() - began
    print(json.dumps({"mode": args.mode, "batch": args.batch, "minutes": args.minutes, "segments": count,
                      "elapsed_s": round(elapsed), "s_per_audio_minute": round(elapsed / args.minutes, 2),
                      "peak_gb_after_load": after_load, "peak_gb": peak_gb()}))


if __name__ == "__main__":
    main()
