"""Measure ASR variants on the evaluation clips (no human reference exists, so no WER).

    python tools/asr_measure.py --label hst --set hallucination_silence_threshold=2.0 [--base baseline]

For every clip in evaluation/media it records seconds per audio minute, segment/word counts, low-confidence
segments (D-089 definition: mean word probability < 0.5, or a run of >= 3 words below 0.3), segments removed by the
hallucination filter and the speech coverage. Results go to evaluation/asr/<label>.json; with --base a side-by-side
diff of every changed segment is written to evaluation/asr/<label>.vs.<base>.txt so a human can judge.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import hallucination  # noqa: E402
from app.core.audio_processor import SAMPLE_RATE, load_wav  # noqa: E402
from app.core.redecode import is_low_confidence  # noqa: E402
from app.core.transcription import AsrOptions  # noqa: E402

OUT = ROOT / "evaluation" / "asr"


def parse_set(items: list[str]) -> dict:
    values = {}
    for item in items:
        key, _, raw = item.partition("=")
        values[key] = None if raw == "None" else json.loads(raw)
    return values


def run_clip(engine, wav: Path, seconds: float, language: str, redo: bool = False, prep=None,
             snr_db: float | None = None) -> dict:
    audio = load_wav(wav)[: int(seconds * SAMPLE_RATE)]
    if snr_db is not None:                      # synthetic stress test: white noise at the given signal-to-noise ratio
        import numpy as np

        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        noise = np.random.default_rng(1).normal(0.0, rms * 10 ** (-snr_db / 20), len(audio))
        audio = np.clip(audio + noise.astype(np.float32), -1.0, 1.0)
    prep_s = 0.0
    if prep:
        began = time.monotonic()
        audio = prep(audio)
        prep_s = round(time.monotonic() - began, 1)
    length = len(audio) / SAMPLE_RATE
    began = time.monotonic()
    segments = [s.to_dict() for s in engine.transcribe(audio, language, 0.0, None)]
    elapsed = time.monotonic() - began
    redone = None
    if redo:
        import tempfile

        from app.core import redecode
        low_before = sum(1 for s in segments if is_low_confidence(s))
        mean_logprob_before = sum(s["avg_logprob"] or 0.0 for s in segments) / max(len(segments), 1)
        records = [{"segment": s} for s in segments]
        began = time.monotonic()
        redone = redecode.redecode(engine, [audio], records, language, length,
                                   Path(tempfile.mkdtemp()) / "re.jsonl", lambda: None)
        redone["elapsed_s"] = round(time.monotonic() - began, 1)
        redone["low_confidence_before"] = low_before
        redone["mean_avg_logprob_before"] = round(mean_logprob_before, 4)
        redone["mean_avg_logprob_after"] = round(sum(r["segment"]["avg_logprob"] or 0.0 for r in records)
                                                 / max(len(records), 1), 4)
        segments = [r["segment"] for r in records]
    kept = [s for s in segments if not hallucination.reason(s)]
    return {
        "audio_s": round(length, 1), "elapsed_s": round(elapsed, 1),
        "s_per_audio_minute": round(elapsed / (length / 60), 2),
        "segments": len(segments), "removed_by_filter": len(segments) - len(kept),
        "words": sum(len(s["words"]) for s in kept),
        "low_confidence": sum(1 for s in kept if is_low_confidence(s)),
        "speech_s": round(sum(s["end"] - s["start"] for s in kept), 1),
        "kept": kept, "redecode": redone, "prep_s": prep_s,
    }


def diff_report(new: dict, old: dict) -> str:
    lines = []
    for clip, a in old["clips"].items():
        b = new["clips"].get(clip)
        if not b:
            continue
        ta = [s["text"] for s in a["kept"]]
        tb = [s["text"] for s in b["kept"]]
        changed = 0
        lines.append(f"=== {clip}: base {len(ta)} segments, new {len(tb)} segments")
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            changed += max(i2 - i1, j2 - j1)
            old_part = a["kept"][i1:i2]
            new_part = b["kept"][j1:j2]
            at = f"{(old_part or new_part)[0]['start']:.1f}"
            lines.append(f"--- @{at}s")
            lines += [f"  BASE [{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in old_part] or ["  BASE (none)"]
            lines += [f"  NEW  [{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in new_part] or ["  NEW  (none)"]
        lines.append(f"changed segments: {changed}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="AsrOptions fields, e.g. vad_threshold=0.3")
    parser.add_argument("--base", help="label of an earlier run to diff against")
    parser.add_argument("--seconds", type=float, default=300.0, help="seconds per clip (0 = whole clip)")
    parser.add_argument("--redecode", action="store_true", help="also run the low-confidence re-decode pass")
    parser.add_argument("--enhance", choices=("old", "new"), help="clean the audio first: old = D-044 (version 1), "
                        "new = per-chunk gate and mix-back (version 2)")
    parser.add_argument("--noise-db", type=float, help="add white noise at this SNR (stress test)")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--engine", default="whisper", help="'whisper' (FasterWhisperEngine)")
    args = parser.parse_args()

    from app.core.transcription import FasterWhisperEngine
    from app.models.model_manager import ModelManager
    from app.utils.cuda_setup import register_nvidia_dll_dirs
    from app.utils.paths import AppPaths, default_data_root

    paths = AppPaths.from_root(default_data_root()).ensure()
    register_nvidia_dll_dirs()
    model_dir = ModelManager(paths.models_dir).ensure("whisper", args.model)
    options = replace(AsrOptions(beam_size=5), **parse_set(args.set))
    engine = FasterWhisperEngine(model_dir, "cuda", "int8_float16", options)
    prep = None
    if args.enhance:
        from app.core import enhance

        sep_dir = enhance.ensure_model(paths.models_dir)
        extra = {} if args.enhance == "new" else {"gate_ratio": 0.0, "mix_back": 0.0}
        prep = lambda audio: enhance.level_speech(enhance.separate_speech(audio, sep_dir, threads=4, **extra))
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"label": args.label, "options": options.__dict__, "clips": {}}
    for wav in sorted((ROOT / "evaluation" / "media").glob("*.wav")):
        result["clips"][wav.stem] = run_clip(engine, wav, args.seconds or 1e9, "tr", args.redecode, prep, args.noise_db)
        c = result["clips"][wav.stem]
        print(f"{wav.stem}: {c['s_per_audio_minute']} s/min, {c['segments']} seg, low={c['low_confidence']}, "
              f"removed={c['removed_by_filter']}, words={c['words']}"
              + (f", redecode={ {k: v for k, v in c['redecode'].items() if k != 'spans'} }" if c["redecode"] else ""),
              flush=True)
    (OUT / f"{args.label}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.base:
        base = json.loads((OUT / f"{args.base}.json").read_text(encoding="utf-8"))
        (OUT / f"{args.label}.vs.{args.base}.txt").write_text(diff_report(result, base), encoding="utf-8")


if __name__ == "__main__":
    main()
