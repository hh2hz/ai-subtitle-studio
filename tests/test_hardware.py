from app.core.modes import Mode
from app.services.hardware_detection import GpuInfo, HardwareInfo, detect, recommend_asr, recommend_translation

GPU_TYPES = ["float32", "int8", "int8_float16", "int8_float32", "float16"]


def _hw(vram=None, ram=16000, cuda=True):
    return HardwareInfo(
        os="test", cpu_name="cpu", logical_cores=8, ram_mb=ram, disk_free_mb=100000,
        gpus=[GpuInfo("GPU", vram)] if vram else [],
        cuda_device_count=1 if cuda and vram else 0,
        cuda_compute_types=GPU_TYPES if cuda and vram else [],
    )


def test_gtx1650_class_gpu():
    hw = _hw(vram=4096)
    balanced = recommend_asr(hw, Mode.BALANCED)
    assert (balanced[0].model, balanced[0].device, balanced[0].compute_type) == ("large-v3-turbo", "cuda", "int8_float16")
    assert balanced[-1].device == "cpu"
    maximum = recommend_asr(hw, Mode.MAXIMUM_ACCURACY)
    assert [p.model for p in maximum[:2]] == ["large-v3", "large-v3-turbo"]
    # 4 GB is too small for the 3B translation model next to the desktop: CPU only.
    assert [(p.device, p.compute_type) for p in recommend_translation(hw, "madlad")] == [("cpu", "int8")]
    assert [p.device for p in recommend_translation(_hw(vram=8192), "madlad")] == ["cuda", "cpu"]


def test_large_gpu_uses_fp16():
    plans = recommend_asr(_hw(vram=8192), Mode.BALANCED)
    assert (plans[0].model, plans[0].compute_type) == ("large-v3", "float16")


def test_cpu_only_by_ram():
    assert [p.model for p in recommend_asr(_hw(ram=16000), Mode.BALANCED)] == ["medium"]
    assert [p.model for p in recommend_asr(_hw(ram=8000), Mode.BALANCED)] == ["small"]
    assert [p.model for p in recommend_asr(_hw(ram=4000), Mode.FAST)] == ["base"]
    assert all(p.device == "cpu" for p in recommend_translation(_hw(), "madlad"))


def test_gpu_without_cuda_runtime_falls_back_to_cpu():
    hw = _hw(vram=8192, cuda=False)
    assert all(p.device == "cpu" for p in recommend_asr(hw, Mode.BALANCED))


def test_detect_never_raises(tmp_path):
    info = detect(tmp_path)
    assert info.logical_cores >= 1
    assert isinstance(info.to_dict(), dict)
