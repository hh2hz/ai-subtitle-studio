"""Compare Whisper on the original audio with Whisper on the cleaned audio (voice separation + level boost).

    .venv\\Scripts\\python.exe tools\\compare_asr.py "D:\\Kurtlar Vadisi\\61.mp4" --language tr --minutes 10

Writes "<name>.asr-compare" next to the video: report.txt (lines, removed hallucinations, time), one SRT per
setup and side_by_side.txt (each original-audio line with what the cleaned-audio run heard at the same time).
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import hallucination  # noqa: E402
from app.core.audio_processor import SAMPLE_RATE, extract_audio, load_wav  # noqa: E402
from app.core.exporter import Cue, write_srt  # noqa: E402
from app.core.modes import Mode  # noqa: E402
from app.core.transcription import AsrOptions  # noqa: E402
from app.models import engines  # noqa: E402
from app.models.model_manager import ModelManager  # noqa: E402
from app.services import hardware_detection  # noqa: E402
from app.utils.cuda_setup import register_nvidia_dll_dirs  # noqa: E402
from app.utils.paths import AppPaths, default_data_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--language", default="tr")
    parser.add_argument("--minutes", type=float, default=10.0, help="0 = whole file")
    args = parser.parse_args()

    out = args.video.with_name(args.video.stem + ".asr-compare")
    out.mkdir(exist_ok=True)
    paths = AppPaths.from_root(default_data_root()).ensure()
    register_nvidia_dll_dirs()
    plan = hardware_detection.recommend_asr(hardware_detection.detect(paths.root), Mode.MAXIMUM_ACCURACY)[0]
    wav = out / "audio.wav"
    if not wav.is_file():
        extract_audio(args.video, wav)
    audio = load_wav(wav)
    if args.minutes > 0:
        audio = audio[: int(args.minutes * 60 * SAMPLE_RATE)]
    seconds = len(audio) / SAMPLE_RATE

    started = time.monotonic()
    cleaned = engines.audio_enhancer(paths.models_dir)(audio)
    clean_time = time.monotonic() - started

    results = {}
    for name, signal, extra in (("original", audio, 0.0), ("cleaned", cleaned, clean_time)):
        engine = engines.load_asr_engine(plan, ModelManager(paths.models_dir), AsrOptions(beam_size=5))
        began = time.monotonic()
        segments = list(engine.transcribe(signal, args.language, 0.0, None))
        took = time.monotonic() - began + extra
        del engine
        gc.collect()
        kept = [s for s in segments if not hallucination.reason(s.to_dict())]
        removed = [s for s in segments if hallucination.reason(s.to_dict())]
        results[name] = (kept, removed, took)
        write_srt([Cue(s.start, s.end, s.text) for s in kept], out / f"{name}.srt")
        print(f"{name}: {len(kept)} lines, {len(removed)} removed, {took:.0f} s")

    lines = [f"Video: {args.video}", f"Audio: {seconds / 60:.1f} min, {args.language}, Whisper {plan.model} on "
             f"{plan.device}", "", f"{'setup':<10}{'lines':>7}{'removed':>9}{'time s':>9}"]
    for name, (kept, removed, took) in results.items():
        lines.append(f"{name:<10}{len(kept):>7}{len(removed):>9}{took:>9.0f}")
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "side_by_side.txt", "w", encoding="utf-8") as f:
        for seg in results["original"][0]:
            heard = " ".join(s.text for s in results["cleaned"][0] if s.end > seg.start and s.start < seg.end)
            f.write(f"[{seg.start:8.2f}]\n  original {seg.text}\n  cleaned  {heard}\n\n")
    print("\n".join(lines) + f"\n\nResults in {out}")


if __name__ == "__main__":
    main()
