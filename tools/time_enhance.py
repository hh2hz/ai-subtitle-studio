"""Time the voice separation on this PC (D-046): fp16 (used by the app) and fp32 Spleeter models, 1/2/4 threads, 60 s of audio.

    .venv\\Scripts\\python.exe tools\\time_enhance.py "D:\\Kurtlar Vadisi\\61.mp4"
Prints seconds per minute of audio; writes tools/time_enhance_report.txt. The fp32 model (75 MB) is downloaded on first use. Reference (cloud Xeon 2.8 GHz): 4-6 s per minute.
"""

import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import enhance  # noqa: E402
from app.core.audio_processor import SAMPLE_RATE, extract_audio, load_wav  # noqa: E402
from app.utils.paths import AppPaths, default_data_root  # noqa: E402

FP32 = enhance.SeparatorModel(
    url="https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/"
        "sherpa-onnx-spleeter-2stems.tar.bz2",
    sha256="e26401d9c1801f43c0229731d78d32c2e80085e3aceedeb25f29d2de5fa68ca2",
    folder="sherpa-onnx-spleeter-2stems", vocals="vocals.onnx", accompaniment="accompaniment.onnx")


if __name__ == "__main__":
    lines = [f"{platform.platform()} | {platform.processor()} | {os.cpu_count()} logical CPUs"]
    try:
        import sherpa_onnx
        lines.append(f"sherpa-onnx {sherpa_onnx.__version__}")
    except ImportError:
        sys.exit("sherpa-onnx is not installed: pip install -r requirements.txt")
    video = Path(sys.argv[1])
    models_dir = AppPaths.from_root(default_data_root()).ensure().models_dir
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        extract_audio(video, wav)
        audio = load_wav(wav)
    start = min(len(audio) // 3, len(audio) - 60 * SAMPLE_RATE) if len(audio) > 60 * SAMPLE_RATE else 0
    audio = audio[start:start + 60 * SAMPLE_RATE]
    minutes = len(audio) / SAMPLE_RATE / 60
    for model in (enhance.SPLEETER, FP32):
        model_dir = enhance.ensure_model(models_dir, model)
        for threads in (1, 2, 4):
            began = time.monotonic()
            enhance.separate_speech(audio, model_dir, model, threads=threads)
            took = (time.monotonic() - began) / minutes
            lines.append(f"{model.folder:<36} threads={threads}  {took:6.1f} s per minute of audio")
            print(lines[-1], flush=True)
    report = ROOT / "tools" / "time_enhance_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {report}")
