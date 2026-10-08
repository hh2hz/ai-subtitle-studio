"""Built-in local model runtime (llama.cpp server) and TranslateGemma backend, with a fake server binary."""

import hashlib
import io
import sys
import threading
import types
import zipfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.core.local_llm import LocalModelError, LocalTranslationBackend, clean_line, prompt_style
from app.models import engines
from app.models.model_manager import ModelManager
from app.services import local_runtime
from app.services.hardware_detection import EnginePlan
from app.services.job_manager import translation_plans
from tests.test_hardware import _hw

FAKE_SERVER = Path(__file__).parent / "helpers" / "fake_llama_server.py"


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF")
    return path


@pytest.fixture
def server(model_file, tmp_path):
    srv = local_runtime.LlamaServer(FAKE_SERVER, model_file, gpu=True, log_path=tmp_path / "server.log")
    yield srv
    srv.stop()


def test_clean_line_and_style():
    assert clean_line('[3]  "merhaba"  ') == "merhaba"
    assert prompt_style("translategemma-4b-q4km") == "translategemma"


def test_server_lifecycle_and_translation(server):
    server.start(timeout_s=30)
    backend = LocalTranslationBackend(server, "translategemma-4b-q4km")
    assert backend.translate(["a", "b"], "tr", "ar", context=[("x", "X")]) == ["L(a)", "L(b)"]
    backend.close()
    with pytest.raises(LocalModelError):
        server.complete("x")


def test_gpu_start_failure_falls_back_to_cpu(server, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_FAIL_GPU", "1")
    server.start(timeout_s=30)
    assert server.gpu is False
    assert "failed to initialize CUDA" in server.log_path.read_text()


def test_cuda_dll_dirs_put_on_server_path(model_file, tmp_path, monkeypatch):
    dll_dir = tmp_path / "nvidia" / "cuda_runtime" / "bin"
    monkeypatch.setenv("FAKE_LLAMA_EXPECT_PATH", str(dll_dir))
    srv = local_runtime.LlamaServer(FAKE_SERVER, model_file, gpu=True, log_path=tmp_path / "s.log",
                                    dll_dirs=[dll_dir])
    try:
        srv.start(timeout_s=30)
        assert srv.gpu is True
    finally:
        srv.stop()


def test_server_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_CACHE_MAXSIZE", raising=False)
    env = local_runtime.LlamaServer(FAKE_SERVER, tmp_path / "m.gguf", gpu=True, dll_dirs=[tmp_path])._environment()
    assert env["PATH"].startswith(str(tmp_path)) and env["CUDA_CACHE_MAXSIZE"] == str(4 * 1024 ** 3)


def test_cuda_build_requires_cuda_libraries(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(local_runtime, "nvidia_bin_dirs", lambda: [tmp_path])
    (tmp_path / "cublas64_12.dll").write_bytes(b"x")
    with pytest.raises(LocalModelError, match="cudart64_12.dll"):
        local_runtime.ensure_runtime(tmp_path, local_runtime.WINDOWS_CUDA)


def test_missing_model_fails_cleanly(tmp_path):
    srv = local_runtime.LlamaServer(FAKE_SERVER, tmp_path / "missing.gguf", log_path=tmp_path / "l.log")
    with pytest.raises(LocalModelError, match="exited"):
        srv.start(timeout_s=30)


# -- runtime download -----------------------------------------------------------------------------

def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("llama-server.exe" if sys.platform == "win32" else "llama-server", b"binary")
        zf.writestr("ggml.dll", b"dll")
        zf.writestr("../evil.txt", b"x")              # path traversal attempt is flattened
    return buf.getvalue()


@pytest.fixture
def file_server():
    payload = {"data": _zip_bytes()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload["data"])))
            self.end_headers()
            self.wfile.write(payload["data"])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", payload
    srv.shutdown()


def _build(url, data, sha=None):
    @dataclass(frozen=True)
    class TestBuild(local_runtime.RuntimeBuild):
        @property
        def url(self):
            return url
    return TestBuild("t1", "rt.zip", sha or hashlib.sha256(data).hexdigest(), len(data))


def test_runtime_download_verify_extract(tmp_path, file_server):
    url, payload = file_server
    seen = []
    legacy = tmp_path / "llama.cpp" / "b11368"
    legacy.mkdir(parents=True)
    exe = local_runtime.ensure_runtime(tmp_path, _build(url, payload["data"]), lambda f, t: seen.append(f))
    assert not legacy.exists()                                      # old Vulkan build removed
    assert exe.parent == tmp_path / "llama.cpp" / "t1-cpu"
    assert exe.is_file() and (exe.parent / "ggml.dll").is_file() and (exe.parent / "evil.txt").is_file()
    assert not (tmp_path / "llama.cpp" / "evil.txt").exists() and seen
    assert local_runtime.ensure_runtime(tmp_path, _build(url, payload["data"])) == exe     # cached


def test_runtime_checksum_mismatch(tmp_path, file_server):
    url, payload = file_server
    with pytest.raises(LocalModelError, match="checksum"):
        local_runtime.ensure_runtime(tmp_path, _build(url, payload["data"], sha="0" * 64))


def test_windows_build_refused_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(LocalModelError, match="Windows"):
        local_runtime.ensure_runtime(tmp_path)


# -- loader and plans ------------------------------------------------------------------------------

def test_loader_downloads_model_and_starts_server(tmp_path, monkeypatch):
    def fake_snapshot(repo_id, revision, local_dir, allow_patterns):
        assert repo_id == "mradermacher/translategemma-4b-it-GGUF" and allow_patterns == ["translategemma-4b-it.Q4_K_M.gguf"]
        Path(local_dir, allow_patterns[0]).write_bytes(b"GGUF")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot))
    models = ModelManager(tmp_path / "models")
    backend = engines.load_translation_backend(
        EnginePlan("translategemma-4b-q4km", "local", "cpu", "t"), models, 4,
        runtime_builder=lambda base, build, progress=None: FAKE_SERVER)
    try:
        assert backend.name == "local:translategemma-4b-q4km"
        assert backend.translate(["a"], "tr", "ar") == ["L(a)"]
    finally:
        backend.close()


def test_translation_plans_local_first():
    plans = translation_plans(_hw(vram=4096), {"translation_engine": "local"})
    assert [(p.device, p.compute_type) for p in plans] == [("local", "cuda"), ("local", "cpu"), ("cpu", "int8")]
    plans = translation_plans(_hw(vram=0), {"translation_engine": "local"})
    assert [(p.device, p.compute_type) for p in plans] == [("local", "cpu"), ("cpu", "int8")]
    assert [p.device for p in translation_plans(_hw(vram=4096), {"translation_engine": "madlad"})] == ["cpu"]


def test_pipeline_falls_back_to_madlad_when_local_model_fails(tmp_path):
    from app.utils.atomic import read_json
    from tests.fakes import CPU_MT, Factory, FakeAsrEngine, FakeTranslator
    from tests.media import make_tone_file
    from tests.test_pipeline import _pipeline

    media = make_tone_file(tmp_path / "Ep.m4a", seconds=8.0)
    local = EnginePlan("translategemma-4b-q4km", "local", "cpu", "test")
    mt_f = Factory({("translategemma-4b-q4km", "local"): LocalModelError("llama-server exited with code 1"),
                    (CPU_MT.model, "cpu"): FakeTranslator("madlad")})
    pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), mt_f, mt_plans=(local, CPU_MT))
    result = pipe.run()
    assert any("translategemma-4b-q4km/local" in w for w in result.warnings)
    assert {u["engine"] for u in read_json(result.outputs["master_transcript"])["translation_units"]} == {"madlad"}


def test_msvc_runtime_copied_next_to_server(tmp_path, monkeypatch):
    source = tmp_path / "pyside"
    source.mkdir()
    for name in ("msvcp140.dll", "vcruntime140_1.dll"):
        (source / name).write_bytes(b"dll")
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "vcruntime140_1.dll").write_bytes(b"existing")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(local_runtime, "_msvc_sources", lambda: [source])
    assert local_runtime.provide_msvc_runtime(target) == ["msvcp140.dll"]
    assert (target / "vcruntime140_1.dll").read_bytes() == b"existing"          # never overwritten
