"""On-demand model downloads, disk usage and removal.

Layout: <models_dir>/whisper/<name>/ and <models_dir>/translation/<key>/.
A model directory is complete only when it contains the COMPLETE_MARKER file, so an
interrupted download is never mistaken for an installed model.
"""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

COMPLETE_MARKER = ".download-complete"
_DISK_MARGIN_MB = 1024

ProgressCallback = Callable[[float, str], None]


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    kind: str               # "whisper" or "translation"
    name: str
    repo_id: str
    revision: str | None    # pinned commit, or None for the repository default branch
    approx_size_mb: int     # used for the free-space check and progress estimate
    license: str
    allow_patterns: tuple[str, ...] = ()


# Whisper repositories follow faster-whisper 1.2.1's own name mapping. Sizes are approximate.
# Every repository is pinned to an immutable commit, so a later push to the Hugging Face repository cannot change
# what the app downloads. The commits are the "main" revision resolved through
# https://huggingface.co/api/models/<repo>/revision/main; update them deliberately, never to "main".
_WHISPER_FILES = ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*")
WHISPER_MODELS: dict[str, ModelSpec] = {
    name: ModelSpec("whisper", name, repo, revision, size, "MIT", _WHISPER_FILES)
    for name, repo, revision, size in (
        ("tiny", "Systran/faster-whisper-tiny", "d90ca5fe260221311c53c58e660288d3deb8d356", 80),
        ("base", "Systran/faster-whisper-base", "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66", 150),
        ("small", "Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120", 500),
        ("medium", "Systran/faster-whisper-medium", "08e178d48790749d25932bbc082711ddcfdfbc4f", 1550),
        ("large-v3", "Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478", 3100),
        ("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf", 1650),
    )
}

TRANSLATION_MODELS: dict[str, ModelSpec] = {
    "madlad400-3b-ct2-int8": ModelSpec(
        kind="translation",
        name="madlad400-3b-ct2-int8",
        repo_id="Nextcloud-AI/madlad400-3b-mt-ct2-int8",
        revision="aa32bbdeba7880eff2096ec044cb155a340a9400",
        approx_size_mb=2990,
        license="Apache-2.0",
        allow_patterns=("config.json", "model.bin", "shared_vocabulary.json", "spiece.model"),
    ),
}
DEFAULT_TRANSLATION_MODEL = "madlad400-3b-ct2-int8"

# Local LLM translators (GGUF for the built-in llama.cpp runtime). Not gated; license tag "gemma".
TRANSLATION_MODELS["translategemma-4b-q4km"] = ModelSpec(
    kind="translation",
    name="translategemma-4b-q4km",
    repo_id="mradermacher/translategemma-4b-it-GGUF",
    revision="35a7486e128b19642cdc72d7b91b21ba388aaf42",
    approx_size_mb=2375,
    license="Gemma Terms of Use",
    allow_patterns=("translategemma-4b-it.Q4_K_M.gguf",),
)
LOCAL_LLM_FILES = {"translategemma-4b-q4km": "translategemma-4b-it.Q4_K_M.gguf"}
DEFAULT_LOCAL_MODEL = "translategemma-4b-q4km"

# Only Whisper is supported for ASR
_TABLES = {"whisper": WHISPER_MODELS, "translation": TRANSLATION_MODELS}


def _dir_size(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        try:
            if file.is_file():
                total += file.stat().st_size
        except OSError:
            pass
    return total


class ModelManager:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)

    def spec(self, kind: str, name: str) -> ModelSpec:
        table = _TABLES.get(kind, {})
        if name not in table:
            raise ModelError(f"Unknown {kind} model: {name}")
        return table[name]

    def path(self, kind: str, name: str) -> Path:
        return self.models_dir / kind / name

    def is_installed(self, kind: str, name: str) -> bool:
        return (self.path(kind, name) / COMPLETE_MARKER).is_file()

    def installed(self) -> list[tuple[str, str, int]]:
        """(kind, name, size_bytes) for every complete model."""
        result = []
        for kind in ("whisper", "translation"):
            base = self.models_dir / kind
            if base.is_dir():
                for d in sorted(base.iterdir()):
                    if (d / COMPLETE_MARKER).is_file():
                        result.append((kind, d.name, _dir_size(d)))
        return result

    def remove(self, kind: str, name: str) -> None:
        target = self.path(kind, name)
        if target.exists():
            shutil.rmtree(target)
            log.info("Removed model %s/%s", kind, name)

    def ensure(self, kind: str, name: str, progress: ProgressCallback | None = None) -> Path:
        """Return the local model directory, downloading it first if needed."""
        spec = self.spec(kind, name)
        target = self.path(kind, name)
        if self.is_installed(kind, name):
            return target
        target.mkdir(parents=True, exist_ok=True)
        free_mb = shutil.disk_usage(target).free // (1024 * 1024)
        needed_mb = spec.approx_size_mb - _dir_size(target) // (1024 * 1024) + _DISK_MARGIN_MB
        if free_mb < needed_mb:
            raise ModelError(
                f"Not enough disk space for {name}: about {needed_mb} MB needed, {free_mb} MB free in {target}")
        log.info("Downloading %s model %s from %s (about %d MB)", kind, name, spec.repo_id, spec.approx_size_mb)

        stop = threading.Event()

        def _watch() -> None:
            # Byte-level progress estimated from the growing directory size.
            expected = spec.approx_size_mb * 1024 * 1024
            while not stop.wait(1.0):
                if progress:
                    done = _dir_size(target)
                    progress(min(done / expected, 0.99), f"{done // (1024 * 1024)} / ~{spec.approx_size_mb} MB")

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                local_dir=str(target),
                allow_patterns=list(spec.allow_patterns) or None,
            )
        except Exception as exc:
            raise ModelError(f"Download of {name} failed: {type(exc).__name__}: {exc}") from exc
        finally:
            stop.set()
            watcher.join(timeout=5)
        (target / COMPLETE_MARKER).write_text(spec.revision or "default-branch", encoding="utf-8")
        if progress:
            progress(1.0, "done")
        log.info("Model %s/%s ready (%d MB)", kind, name, _dir_size(target) // (1024 * 1024))
        return target
