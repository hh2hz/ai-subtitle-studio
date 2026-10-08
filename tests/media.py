"""Self-generated test media (no third-party content)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np


def make_tone_file(path: Path, seconds: float, rate: int = 44100, channels: int = 2,
                   codec: str = "aac") -> Path:
    """Encode a sine tone into an audio-only container chosen by the file extension."""
    import av

    layout = "stereo" if channels == 2 else "mono"
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=rate, layout=layout)
        t = np.arange(int(seconds * rate)) / rate
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        samples = np.tile(tone, (channels, 1)) if channels == 2 else tone[None, :]
        chunk = 1024
        pts = 0
        for i in range(0, samples.shape[1], chunk):
            part = np.ascontiguousarray(samples[:, i:i + chunk])
            frame = av.AudioFrame.from_ndarray(part, format="fltp", layout=layout)
            frame.sample_rate = rate
            frame.pts = pts
            frame.time_base = Fraction(1, rate)
            pts += part.shape[1]
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def make_video_only_file(path: Path, frames: int = 10) -> Path:
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(frames):
            img = np.full((48, 64, 3), i * 20 % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path
