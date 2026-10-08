import sys
import types

import pytest

from app.models.model_manager import COMPLETE_MARKER, WHISPER_MODELS, ModelError, ModelManager
from app.services import gpu_probe


def _fake_hub(monkeypatch, behaviour):
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = behaviour
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def test_download_writes_marker_and_lists(tmp_path, monkeypatch):
    calls = []
    revision = WHISPER_MODELS["tiny"].revision

    def fake_download(repo_id, revision, local_dir, allow_patterns):
        calls.append((repo_id, revision))
        (tmp_path / "models" / "whisper" / "tiny" / "model.bin").write_bytes(b"x" * 1000)

    _fake_hub(monkeypatch, fake_download)
    mm = ModelManager(tmp_path / "models")
    assert not mm.is_installed("whisper", "tiny")
    path = mm.ensure("whisper", "tiny")
    assert (path / COMPLETE_MARKER).is_file()
    assert calls == [("Systran/faster-whisper-tiny", revision)]      # the pinned commit, never the default branch
    assert revision is not None
    mm.ensure("whisper", "tiny")                 # already installed: no second download
    assert len(calls) == 1
    assert mm.installed() == [("whisper", "tiny", 1000 + len(revision))]
    mm.remove("whisper", "tiny")
    assert mm.installed() == []


def test_failed_download_is_not_marked_complete(tmp_path, monkeypatch):
    def broken(**kwargs):
        raise ConnectionError("network down")

    _fake_hub(monkeypatch, broken)
    mm = ModelManager(tmp_path / "models")
    with pytest.raises(ModelError, match="network down"):
        mm.ensure("translation", "madlad400-3b-ct2-int8")
    assert not mm.is_installed("translation", "madlad400-3b-ct2-int8")


def test_disk_space_check(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10, 10, 100 * 1024 * 1024))
    with pytest.raises(ModelError, match="Not enough disk space"):
        ModelManager(tmp_path / "models").ensure("whisper", "large-v3")


def test_unknown_model(tmp_path):
    with pytest.raises(ModelError):
        ModelManager(tmp_path).ensure("whisper", "huge-v9")


def test_gpu_probe_failure_is_reported_not_raised(tmp_path):
    ok, detail = gpu_probe.probe("whisper", tmp_path / "no-model", "int8", timeout_s=120)
    assert ok is False and detail


def test_gpu_probe_unknown_kind():
    assert gpu_probe.run_probe("nope", "x", "int8") == 2


def test_gpu_runtime_download_extracts_only_dlls(tmp_path, monkeypatch):
    import hashlib
    import io
    import zipfile

    from app.services import gpu_runtime

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("nvidia/cublas/bin/cublas64_12.dll", b"dll")
        zf.writestr("nvidia/cublas/include/cublas.h", b"h")
        zf.writestr("nvidia/../evil.dll", b"x")
    wheel = buf.getvalue()
    infos = {name: {"url": f"mem://{name}", "sha256": hashlib.sha256(wheel).hexdigest(), "size": len(wheel),
                    "filename": f"{name}-win_amd64.whl"} for name, _ in gpu_runtime.PACKAGES}
    monkeypatch.setattr(gpu_runtime, "_wheel_info", lambda name, version: infos[name])

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(gpu_runtime.urllib.request, "urlopen", lambda url, timeout=0: Resp(wheel))
    seen = []
    target = gpu_runtime.download_all(lambda f, t: seen.append(f), target=tmp_path / "cuda")
    assert (target / "nvidia" / "cublas" / "bin" / "cublas64_12.dll").read_bytes() == b"dll"
    assert not list(target.rglob("*.h")) and not (tmp_path / "evil.dll").exists() and seen[-1] == 1.0
    assert all((target / f"{n}.complete").is_file() for n, _ in gpu_runtime.PACKAGES)
    gpu_runtime.download_all(target=tmp_path / "cuda")                        # cached: nothing downloaded again


def test_default_pipeline_factory_builds_a_pipeline(tmp_path, monkeypatch):
    """The production factory must construct a Pipeline (D-049: a missing engines helper broke every real run
    while all unit tests passed)."""
    from app.core.modes import Mode
    from app.core.pipeline import JobConfig, Pipeline
    from app.services import hardware_detection, job_manager
    from tests.fakes import CPU_ASR

    monkeypatch.setattr(hardware_detection, "detect", lambda data_dir: None)
    monkeypatch.setattr(hardware_detection, "recommend_asr", lambda hw, mode: [CPU_ASR])
    monkeypatch.setattr(job_manager, "translation_plans", lambda hw, extras: [CPU_ASR])
    factory = job_manager.default_pipeline_factory(tmp_path / "jobs", tmp_path / "models", tmp_path)
    config = JobConfig(tmp_path / "a.m4a", "tr", "ar", Mode.BALANCED)
    pipeline = factory(config, None, None, None, {"llm_refine": True, "audio_enhance": "on"})
    assert isinstance(pipeline, Pipeline)
