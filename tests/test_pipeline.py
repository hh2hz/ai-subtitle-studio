import json
import threading

import pytest

from app.core.errors import JobCancelled, PipelineError
from app.core.exporter import read_srt
from app.core.master_transcript import validate_document
from app.core.modes import Mode
from app.core.pipeline import JobConfig, Pipeline
from app.utils.atomic import load_jsonl, read_json
from tests.fakes import CPU_ASR, CPU_MT, GPU_ASR, GPU_MT, SEGMENT_SECONDS, Factory, FakeAsrEngine, FakeTranslator
from tests.media import make_tone_file


@pytest.fixture
def media(tmp_path):
    return make_tone_file(tmp_path / "Episode 1.m4a", seconds=20.0)


def _pipeline(tmp_path, media, asr, mt, asr_plans=(CPU_ASR,), mt_plans=(CPU_MT,), refiner_factory=None, **kw):
    config = JobConfig(media, kw.pop("source", "tr"), kw.pop("target", "ar"), Mode.BALANCED,
                       output_dir=tmp_path / "out")
    asr_f = asr if isinstance(asr, Factory) else Factory({(CPU_ASR.model, "cpu"): asr})
    mt_f = mt if isinstance(mt, Factory) else Factory({(CPU_MT.model, "cpu"): mt})
    return Pipeline(config, tmp_path / "jobs", list(asr_plans), list(mt_plans), asr_f, mt_f,
                    refiner_factory=refiner_factory, **kw), asr_f, mt_f


def test_end_to_end_outputs(tmp_path, media):
    progress = []
    stages = []
    pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(),
                           progress=lambda *a: progress.append(a),
                           recorder=lambda *a: stages.append(a[:2]))
    result = pipe.run()
    out = result.outputs
    target = read_srt(out["target_srt"])
    source = read_srt(out["source_srt"])
    assert len(source) == 10
    # One cue per utterance: segments are never merged (user feedback on merged sentences).
    assert len(target) == len(source)
    assert all(c.text.startswith("\u200f[ar] ") for c in target)   # RTL mark for Arabic
    assert all(a.end <= b.start for a, b in zip(target, target[1:]))
    doc = read_json(out["master_transcript"])
    assert validate_document(doc) == []
    assert all(s["translation_unit"] is not None for s in doc["segments"])
    assert result.output_dir == tmp_path / "out" / "Episode 1"
    assert (result.output_dir / "Episode 1.ar.srt").is_file()
    assert (result.output_dir / "work" / "ReviewRequired.txt").is_file()
    assert "transcribe" in result.stats["stages"] and "real_time_factor" in result.stats["stages"]["transcribe"]
    overall = [p[2] for p in progress]
    assert overall == sorted(overall) and overall[-1] == pytest.approx(1.0)
    assert ("export", "completed") in stages
    assert (result.output_dir / "work" / "ProcessingLog.txt").is_file()


def test_rerun_uses_cache(tmp_path, media):
    pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator())
    first = pipe.run()
    pipe2, asr_f, mt_f = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator())
    second = pipe2.run()
    assert asr_f.loads == [] and mt_f.loads == []
    assert all(second.stats["stages"][s]["cached"] for s in ("audio", "transcribe", "translate"))
    assert first.job_key == second.job_key


def test_target_change_only_reruns_translation(tmp_path, media):
    _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator())[0].run()
    pipe, asr_f, mt_f = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), target="en")
    result = pipe.run()
    assert asr_f.loads == [] and len(mt_f.loads) == 1
    assert result.stats["stages"]["transcribe"]["cached"] and not result.stats["stages"]["translate"]["cached"]
    assert read_srt(result.outputs["target_srt"])[0].text.startswith("[en] ")


def test_cancel_then_resume_from_last_segment(tmp_path, media):
    cancel = threading.Event()
    engine = FakeAsrEngine(cancel_after=3, cancel_event=cancel)
    pipe, _, _ = _pipeline(tmp_path, media, engine, FakeTranslator(), cancel=cancel)
    with pytest.raises(JobCancelled):
        pipe.run()
    partial = next((pipe.job_dir).glob("transcribe.*.partial.jsonl"))
    assert len(load_jsonl(partial)) == 3

    engine2 = FakeAsrEngine()
    pipe2, _, _ = _pipeline(tmp_path, media, engine2, FakeTranslator())
    result = pipe2.run()
    first_call = engine2.calls[0]
    assert first_call["offset"] == pytest.approx(2 * SEGMENT_SECONDS + SEGMENT_SECONDS - 0.2)
    assert first_call["prompt"] is not None
    assert result.stats["stages"]["audio"]["cached"]
    starts = [s["start"] for s in read_json(result.outputs["master_transcript"])["segments"]]
    assert starts == [i * SEGMENT_SECONDS for i in range(10)]   # no gaps, no duplicates


def test_gpu_failure_mid_run_falls_back_to_cpu(tmp_path, media):
    gpu = FakeAsrEngine(name="gpu", fail_after=4)
    cpu = FakeAsrEngine(name="cpu")
    asr_f = Factory({(GPU_ASR.model, "cuda"): gpu, (CPU_ASR.model, "cpu"): cpu})
    mt_f = Factory({(GPU_MT.model, "cuda"): FakeTranslator("gpu-mt", fail_after_batches=0),
                    (CPU_MT.model, "cpu"): FakeTranslator("cpu-mt")})
    pipe, _, _ = _pipeline(tmp_path, media, asr_f, mt_f, asr_plans=(GPU_ASR, CPU_ASR), mt_plans=(GPU_MT, CPU_MT))
    result = pipe.run()
    doc = read_json(result.outputs["master_transcript"])
    engines = [s["sources"][0]["engine"] for s in doc["segments"]]
    assert engines == ["gpu"] * 4 + ["cpu"] * 6
    assert {u["engine"] for u in doc["translation_units"]} == {"cpu-mt"}
    assert any("continuing with the next plan" in w for w in result.warnings)


def test_unloadable_first_plan_is_skipped(tmp_path, media):
    asr_f = Factory({(GPU_ASR.model, "cuda"): RuntimeError("cudnn64_9.dll not found"),
                     (CPU_ASR.model, "cpu"): FakeAsrEngine()})
    pipe, _, _ = _pipeline(tmp_path, media, asr_f, FakeTranslator(), asr_plans=(GPU_ASR, CPU_ASR))
    result = pipe.run()
    assert [p.device for p in asr_f.loads] == ["cuda", "cpu"]
    assert any("unavailable" in w for w in result.warnings)


def test_all_plans_fail(tmp_path, media):
    asr_f = Factory({(CPU_ASR.model, "cpu"): RuntimeError("broken")})
    pipe, _, _ = _pipeline(tmp_path, media, asr_f, FakeTranslator())
    with pytest.raises(PipelineError) as info:
        pipe.run()
    assert info.value.ui_key == "error.asr_unavailable"


def test_cpu_failure_is_not_retried(tmp_path, media):
    pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(fail_after=2), FakeTranslator())
    with pytest.raises(PipelineError) as info:
        pipe.run()
    assert info.value.ui_key == "error.asr_failed"


def test_auto_language_detection_is_cached(tmp_path, media):
    engine = FakeAsrEngine(language=("tr", 0.3))
    pipe, _, _ = _pipeline(tmp_path, media, engine, FakeTranslator(), source="auto")
    result = pipe.run()
    assert read_json(result.outputs["master_transcript"])["language"]["code"] == "tr"
    assert any("uncertain" in w for w in result.warnings)
    engine2 = FakeAsrEngine()
    _pipeline(tmp_path, media, engine2, FakeTranslator(), source="auto")[0].run()
    assert not any(c.get("detect") for c in engine2.calls)


def test_no_speech(tmp_path):
    short = make_tone_file(tmp_path / "short.m4a", seconds=0.3)
    pipe, _, _ = _pipeline(tmp_path, short, FakeAsrEngine(), FakeTranslator())
    with pytest.raises(PipelineError) as info:
        pipe.run()
    assert info.value.ui_key == "error.no_speech"


def test_missing_input(tmp_path):
    pipe, _, _ = _pipeline(tmp_path, tmp_path / "missing.mp4", FakeAsrEngine(), FakeTranslator())
    with pytest.raises(PipelineError) as info:
        pipe.run()
    assert info.value.ui_key == "error.input_not_found"


def test_translation_receives_previous_lines_as_context(tmp_path, media):
    mt = FakeTranslator()
    pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), mt)
    pipe.run()
    assert mt.contexts[0] == [] and len(mt.contexts[1]) == 3
    assert mt.contexts[1][-1][1].startswith("[ar] ")


# -- AI refinement stage ----------------------------------------------------------------------------

def _refiner_factory(*servers, **kw):
    from app.core.llm_providers import ChatClient, ModelRanking, ProviderPool, ProviderSpec
    from app.core.llm_translation import LlmRefiner

    def factory(src, tgt, media):
        clients = []
        for i, server in enumerate(servers):
            client = ChatClient(ProviderSpec(f"p{i}", server.url, prefer=("good",)), "k")
            client.discover_models()
            clients.append(client)
        options = {k: v for k, v in kw.items() if k != "floor"}
        return LlmRefiner(ProviderPool(clients, ModelRanking(floor=kw.get("floor", 0))), src, tgt, media,
                          sleep=lambda s: None, **options)
    factory.options = kw
    return factory


def test_refine_stage_replaces_draft(tmp_path, media):
    from tests.fake_llm_api import FakeLlmApi

    with FakeLlmApi() as a:
        pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(),
                               refiner_factory=_refiner_factory(a, review=False))
        result = pipe.run()
    target = read_srt(result.outputs["target_srt"])
    assert target[0].text == "‏AR(Cumle 0 burada.)‏"
    doc = read_json(result.outputs["master_transcript"])
    unit = doc["translation_units"][2]
    assert unit["draft"].startswith("[ar] ") and unit["llm_translation"] == "AR(Cumle 2 burada.)"
    assert "reviewer_suggestion" not in unit["flags"]
    assert result.stats["refine"]["refined"] == 10
    from tests.fake_llm_api import NEVZAT_AR
    assert result.stats["refine"]["glossary"] == {"Nevzat": NEVZAT_AR}


def test_refine_without_providers_keeps_draft(tmp_path, media):
    def no_providers(src, tgt, media):
        return None
    pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), refiner_factory=no_providers)
    result = pipe.run()
    assert read_srt(result.outputs["target_srt"])[0].text.startswith("\u200f[ar] ")
    assert any("No usable AI translation provider" in w for w in result.warnings)


def test_refine_resumes_after_providers_were_exhausted(tmp_path, media):
    from tests.fake_llm_api import FakeLlmApi

    with FakeLlmApi(behaviour="rate_limit", retry_after="999") as limited:
        pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(),
                               refiner_factory=_refiner_factory(limited, review=False))
        first = pipe.run()
    assert first.stats["refine"]["refined"] == 0
    assert any("rate limited or unavailable" in w or "AI translation failed" in w for w in first.warnings)
    with FakeLlmApi() as good:
        pipe, asr_f, mt_f = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(),
                                      refiner_factory=_refiner_factory(good, review=False))
        second = pipe.run()
    assert asr_f.loads == [] and mt_f.loads == []          # transcription and draft translation cached
    assert second.stats["refine"]["refined"] == 10


def test_without_draft_madlad_only_fills_lines_the_ai_missed(tmp_path, media):
    from tests.fake_llm_api import FakeLlmApi

    with FakeLlmApi(behaviour="drop_first") as server:       # the AI never returns the first line of a block
        mt = FakeTranslator("madlad")
        pipe, _, mt_f = _pipeline(tmp_path, media, FakeAsrEngine(), mt,
                                  refiner_factory=_refiner_factory(server, mode="translate", review=False))
        result = pipe.run()
    assert result.stats["stages"]["translate"]["skipped"].startswith("not needed")
    assert mt.inputs == ["Cumle 0 burada."]                   # MADLAD translated only the missed line
    doc = read_json(result.outputs["master_transcript"])
    first, second = doc["translation_units"][:2]
    assert first["engine"] == "madlad" and "not_refined_by_ai" in first["flags"]
    assert second["translation"] == "AR(Yarim cumle 1)"
    llm_requests = [json.loads(r["messages"][1]["content"]) for r in server.requests]
    assert all("translation" not in line for req in llm_requests for line in req.get("lines", []))


def test_correct_mode_keeps_local_lines_and_applies_corrections(tmp_path, media):
    from tests.fake_llm_api import FakeLlmApi

    with FakeLlmApi(reviewer_fix=2) as server:
        mt = FakeTranslator("local")
        pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), mt,
                               refiner_factory=_refiner_factory(server, mode="correct", review=False))
        result = pipe.run()
    assert len(mt.inputs) == 10                                  # the local model translated every line
    doc = read_json(result.outputs["master_transcript"])
    units = doc["translation_units"]
    assert units[2]["translation"] == "FIXED" and units[2]["engine"] == "p0:good-model" and units[2]["ai_corrected"]
    assert units[0]["translation"].startswith("[ar] ") and units[0]["engine"] == "local"
    assert not units[0]["ai_corrected"] and "not_refined_by_ai" not in units[0]["flags"]
    stats = result.stats["refine"]
    assert stats["mode"] == "correct" and stats["corrected"] == 1 and stats["refined"] == 10
    # One correct request plus the episode brief request (D-055).
    assert stats["tokens"]["p0"] == {"requests": 2, "prompt_tokens": 200, "completion_tokens": 40}


def test_lines_from_weak_models_are_flagged(tmp_path, media):
    from tests.fake_llm_api import FakeLlmApi

    with FakeLlmApi() as server:
        pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(),
                               refiner_factory=_refiner_factory(server, mode="translate", review=False, floor=99))
        result = pipe.run()
    units = read_json(result.outputs["master_transcript"])["translation_units"]
    assert all("weak_ai_model" in u["flags"] for u in units) and units[0]["confidence"] != "HIGH"


def test_episode_brief_is_cached_only_when_it_has_content(tmp_path, media):
    from tests.fake_llm_api import FakeLlmApi

    with FakeLlmApi() as server:
        pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(),
                               refiner_factory=_refiner_factory(server, mode="translate", review=False))
        pipe.run()
    briefs = list((tmp_path / "jobs").rglob("brief.*.json"))
    assert len(briefs) == 1 and read_json(briefs[0])["characters"][0]["name"] == "Nevzat"

    with FakeLlmApi(behaviour="garbage") as server:
        pipe, _, _ = _pipeline(tmp_path / "b", media, FakeAsrEngine(), FakeTranslator(),
                               refiner_factory=_refiner_factory(server, mode="translate", review=False))
        result = pipe.run()
    assert not list((tmp_path / "b" / "jobs").rglob("brief.*.json"))
    assert any("episode brief unavailable" in w for w in result.warnings)
