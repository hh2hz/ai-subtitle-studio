"""tools/evaluate.py: metrics, alignment, judging with stub models, evaluation set (D-049)."""

import json
import wave
from types import SimpleNamespace

import pytest

from tests.media import make_tone_file
from tools import evaluate as ev


def U(i, start, end, source, target, confidence="HIGH", engine="gemini:m", flags=()):
    return ev.Unit(i, start, end, source, target, confidence, list(flags), engine)


def test_normalize_turkish_and_punctuation():
    assert ev.normalize("İSTANBUL'a, Işık!", "tr") == "istanbul a ışık"
    assert ev.normalize("Hello, World!") == "hello world"


def test_wer_cer_chrf():
    assert ev.wer("bir iki uc", "bir iki dort", "tr") == (1, 3)
    assert ev.cer("abc", "abd") == (1, 3)
    assert ev.chrf(["merhaba dunya"], ["merhaba dunya"]) == 100.0
    assert ev.chrf(["xyz"], ["abc"]) == 0.0
    assert 0 < ev.chrf(["merhaba dunyalar"], ["merhaba dunya"]) < 100


def test_align_split_merge_missing():
    a = [U(0, 0, 2, "a", ""), U(1, 2, 4, "b", ""), U(2, 10, 12, "c", "")]
    b = [U(0, 0, 1, "x", ""), U(1, 1, 4, "y", ""), U(2, 20, 21, "z", "")]
    pairs = ev.align(a, b)
    matched = [(x.id, y.id) for x, y in pairs if x and y]
    assert matched == [(0, 0), (1, 1)]
    assert any(x and x.id == 2 and y is None for x, y in pairs)        # omission
    assert any(x is None and y.id == 2 for x, y in pairs)              # insertion


def test_wilson_interval():
    assert ev.wilson(0, 0) == (0.5, 0.0, 1.0)
    rate, low, high = ev.wilson(60, 40)
    assert rate == 0.6 and low < 0.6 < high and round(high - low, 2) == 0.19


def _transcript(path, units, stats=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "MasterTranscript.json").write_text(json.dumps({
        "language": {"code": "tr"}, "job": {"target_language": "ar"}, "stats": stats or {},
        "translation_units": [{"id": u.id, "start": u.start, "end": u.end, "text": u.source, "translation": u.target,
                               "final_text": u.target, "confidence": u.confidence, "flags": list(u.flags),
                               "qa_flags": [], "engine": u.engine} for u in units]}), encoding="utf-8")


def test_load_result_and_basic_metrics(tmp_path):
    units = [U(0, 0, 2, "a", "x", "LOW", flags=["name_mismatch"]), U(1, 2, 4, "b", "y"), U(2, 4, 6, "c", "z", "MEDIUM")]
    _transcript(tmp_path / "ep" / "work", units, {"media_duration_s": 120, "total_elapsed_s": 60,
                                                  "refine": {"tokens": {"gemini": {"prompt_tokens": 10,
                                                                                   "completion_tokens": 5}}}})
    result = ev.load_result(tmp_path / "ep", "ep")
    m = ev.basic_metrics(result)
    assert m["lines"] == 3 and m["flagged_share"] == round(2 / 3, 4) and m["low_share"] == round(1 / 3, 4)
    assert m["name_mismatch_share"] == round(1 / 3, 4) and m["seconds_per_media_minute"] == 30.0
    assert m["tokens"] == {"gemini": 15}


class StubClient:
    """Judge that prefers any candidate containing 'good' and marks 'bad' subtitles as major errors."""

    def __init__(self):
        self.calls = 0

    def chat(self, model, messages, temperature=0.0, json_mode=False):
        self.calls += 1
        items = json.loads(messages[1]["content"])["items"]
        if "subtitle" in items[0]:
            return json.dumps({"results": [{"id": i["id"], "errors": [{"type": "meaning", "severity": "major"}]
                                            if "bad" in i["subtitle"] else []} for i in items]})
        out = []
        for i in items:
            winner = "A" if "good" in i["A"] else "B" if "good" in i["B"] else "tie"
            out.append({"id": i["id"], "winner": winner, "reason": "r"})
        return json.dumps({"results": out})


def _panel(tmp_path, n=3):
    client = StubClient()
    routes = [SimpleNamespace(key=f"p{k}:m", provider=f"p{k}", model="m", client=client) for k in range(n)]
    return ev.Panel(routes, ev.JudgeCache(tmp_path / "cache.jsonl")), client


def test_pairwise_counts_from_the_new_runs_view_and_uses_the_cache(tmp_path):
    old = ev.Result("ep", "tr", "ar", [U(i, i * 2, i * 2 + 2, f"s{i}", f"bad {i}") for i in range(30)], {})
    new_units = [U(i, i * 2, i * 2 + 2, f"s{i}", f"good {i}" if i < 24 else f"bad {i}") for i in range(30)]
    new_units[29] = U(29, 58, 60, "s29", "bad 29")
    new = ev.Result("ep", "tr", "ar", new_units, {})
    panel, client = _panel(tmp_path)
    report = ev.pairwise(old, new, panel)
    t = report["translation"]
    assert t["compared"] == 24 and t["new_better"] == 24 and t["old_better"] == 0 and t["win_rate"] == 1.0
    assert report["transcript"]["compared"] == 0
    calls = client.calls
    ev.pairwise(old, new, ev.Panel(panel.routes, ev.JudgeCache(tmp_path / "cache.jsonl")))
    assert client.calls == calls                                         # all answers from the cache


def test_pairwise_identical_runs_are_not_judged(tmp_path):
    units = [U(i, i, i + 1, f"s{i}", f"t{i}") for i in range(5)]
    panel, client = _panel(tmp_path)
    report = ev.pairwise(ev.Result("ep", "tr", "ar", units, {}), ev.Result("ep", "tr", "ar", list(units), {}), panel)
    assert report["translation"]["compared"] == 0 and client.calls == 0


def test_absolute_error_rate_and_flag_precision(tmp_path):
    units = [U(0, 0, 1, "a", "bad", "LOW"), U(1, 1, 2, "b", "fine", "LOW"), U(2, 2, 3, "c", "bad"),
             U(3, 3, 4, "d", "fine")]
    panel, _ = _panel(tmp_path, 2)
    a = ev.absolute(ev.Result("ep", "tr", "ar", units, {}), panel)
    assert a["judged_lines"] == 4 and a["major_error_rate"] == 0.5
    assert a["flag_precision"] == 0.5 and a["flag_recall"] == 0.5 and a["error_types"] == {"meaning": 2}


def test_reference_metrics(tmp_path):
    reference = [ev.RefLine(0, 2, "bir iki", "x y"), ev.RefLine(2, 4, "uc dort", "z w"), ev.RefLine(9, 10, "bes", "q")]
    result = ev.Result("ep", "tr", "ar", [U(0, 0, 2, "bir iki", "x y"), U(1, 2, 4, "uc", "z w")], {})
    m = ev.reference_metrics(result, reference)
    assert m["lines"] == 3 and m["omissions"] == 1 and m["wer"] == round(2 / 5, 4)


def _job(jobs, name, media, segments, duration):
    job = jobs / name
    job.mkdir(parents=True)
    (job / "master.k.json").write_text(json.dumps({"segments": segments, "media": {"duration": duration},
                                                   "language": {"code": "tr"}, "job": {"target_language": "ar"}}),
                                       encoding="utf-8")
    (job / "refined.k.json").write_text("{}", encoding="utf-8")
    (job / "job.log").write_text(f"Job started: JobConfig(input_path=PosixPath({str(media)!r}), ...)",
                                 encoding="utf-8")


def test_make_set_picks_three_different_jobs(tmp_path):
    medias = [tmp_path / f"Konseyi'ne {k}.mp4" for k in range(3)]
    for m in medias:
        m.write_bytes(b"x")
    jobs = tmp_path / "jobs"
    seg = lambda s, conf, words: {"start": s, "end": s + 2, "text": " ".join(["w"] * words), "audio_confidence": conf}
    _job(jobs, "aaaa1111", medias[0], [seg(i * 3, 0.9, 3) for i in range(100)], 300)      # many segments
    _job(jobs, "bbbb2222", medias[1], [seg(i * 10, 0.3, 3) for i in range(10)], 300)      # low confidence
    _job(jobs, "cccc3333", medias[2], [seg(i * 10, 0.9, 12) for i in range(10)], 300)     # fast speech
    items = ev.make_set(jobs)
    assert [(i["kind"], i["job"]) for i in items] == [("dialogue", "aaaa1111"), ("noise", "bbbb2222"),
                                                      ("fast_speech", "cccc3333")]
    assert items[0]["from"] == 0 and items[0]["to"] == 300


def test_cut_audio(tmp_path):
    source = make_tone_file(tmp_path / "s.m4a", seconds=6.0)
    out = ev.cut_audio(source, 2.0, 4.5, tmp_path / "cut.wav")
    with wave.open(str(out)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        assert abs(w.getnframes() / 16000 - 2.5) < 0.05


def test_copy_transcription_only(tmp_path):
    src = tmp_path / "a" / "job1"
    (src / "stages").mkdir(parents=True)
    for name in ("audio.wav", "master.k.json", "transcribe.k.partial.jsonl", "refined.k.json", "input.json"):
        (src / name).write_text("x")
    for stage in ("audio", "transcribe", "refine"):
        (src / "stages" / f"{stage}.json").write_text("{}")
    ev._copy_transcription(tmp_path / "a", tmp_path / "b")
    copied = {p.relative_to(tmp_path / "b" / "job1").as_posix() for p in (tmp_path / "b" / "job1").rglob("*")
              if p.is_file()}
    assert copied == {"audio.wav", "master.k.json", "transcribe.k.partial.jsonl", "input.json",
                      "stages/audio.json", "stages/transcribe.json"}


def test_reports_are_written(tmp_path):
    item = {"pairwise": {"translation": {"compared": 2, "new_better": 2, "old_better": 0, "ties": 0, "win_rate": 1.0,
                                         "ci95": [0.3, 1.0], "judge_agreement": 1.0, "disputed": []}}}
    report = {"items": {"ep": item}, "total": ev._total({"ep": item})}
    md = ev.write_report("x", report, tmp_path)
    assert md.is_file() and "win rate 100.0%" in md.read_text(encoding="utf-8")
    assert json.loads((tmp_path / "history.jsonl").read_text().splitlines()[0])["label"] == "x"


@pytest.mark.parametrize("command", [["score", "missing"], ["compare", "a", "b"]])
def test_missing_runs_fail_cleanly(command, tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    with pytest.raises(SystemExit):
        ev.main(command)


def test_run_set_end_to_end_with_fake_engines(tmp_path, monkeypatch):
    """`run` cuts each item, runs the pipeline with the app settings and stores results that `score` can read."""
    from app.core.pipeline import Pipeline
    from app.database.settings import DEFAULTS
    from app.services import job_manager
    from tests.fakes import CPU_ASR, CPU_MT, Factory, FakeAsrEngine, FakeTranslator

    media = make_tone_file(tmp_path / "ep.m4a", seconds=12.0)
    (tmp_path / "set.json").write_text(json.dumps([{"name": "dialogue_x", "media": str(media), "from": 2, "to": 10,
                                                    "source_language": "tr", "target_language": "ar",
                                                    "series": {"series_name": "Kurtlar Vadisi"}}]))
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(ev, "_app_settings", lambda: dict(DEFAULTS))
    seen = {}

    def fake_factory(jobs_dir, models_dir, data_dir):
        def make(config, cancel, progress, recorder, extras):
            seen["series"] = config.series_override.series_name
            seen["extras"] = extras
            return Pipeline(config, jobs_dir, [CPU_ASR], [CPU_MT], Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine()}),
                            Factory({(CPU_MT.model, "cpu"): FakeTranslator()}))
        return make

    monkeypatch.setattr(job_manager, "default_pipeline_factory", fake_factory)
    settings = tmp_path / "s.json"
    settings.write_text(json.dumps({"mode": "fast", "extras": {"llm_review": True}}))
    ev.run_set("base", settings, None)
    results = ev.run_results("base")
    assert list(results) == ["dialogue_x"] and results["dialogue_x"].units
    assert seen["series"] == "Kurtlar Vadisi" and seen["extras"]["llm_review"] is True
    stored = json.loads((tmp_path / "runs" / "base" / "settings.json").read_text())
    assert stored["mode"] == "fast" and not any(k.endswith("_api_key") for k in stored["extras"])
    with pytest.raises(SystemExit):
        ev.run_set("base", None, None)                                   # labels are never overwritten
    ev.run_set("again", None, "base")                                    # transcription reused
    assert ev.run_results("again")["dialogue_x"].units


def test_pairwise_samples_large_items(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "SAMPLE_PAIRWISE", 10)
    old = ev.Result("ep", "tr", "ar", [U(i, i, i + 1, f"s{i}", f"bad {i}") for i in range(40)], {})
    new = ev.Result("ep", "tr", "ar", [U(i, i, i + 1, f"s{i}", f"good {i}") for i in range(40)], {})
    panel, _ = _panel(tmp_path)
    report = ev.pairwise(old, new, panel)
    assert report["differing_lines"]["translation"] == 40 and report["translation"]["compared"] == 10


def test_failing_judge_is_replaced_by_a_spare(tmp_path, monkeypatch):
    monkeypatch.setattr(ev.time, "sleep", lambda s: None)

    class Broken:
        def chat(self, *a, **k):
            raise RuntimeError("timed out")

    good = StubClient()
    bad = SimpleNamespace(key="bad:m", provider="bad", model="m", client=Broken())
    spare = SimpleNamespace(key="spare:m", provider="spare", model="m", client=good)
    other = SimpleNamespace(key="ok:m", provider="ok", model="m", client=good)
    panel = ev.Panel([bad, other], ev.JudgeCache(tmp_path / "c.jsonl"), [spare])
    units_old = [U(i, i, i + 1, f"s{i}", f"bad {i}") for i in range(60)]
    units_new = [U(i, i, i + 1, f"s{i}", f"good {i}") for i in range(60)]
    ev.pairwise(ev.Result("ep", "tr", "ar", units_old, {}), ev.Result("ep", "tr", "ar", units_new, {}), panel)
    assert [r.key for r in panel.routes] == ["spare:m", "ok:m"]


def test_failed_block_is_answered_by_the_spare_at_once(tmp_path, monkeypatch):
    def no_sleep(seconds):
        raise AssertionError("must not wait when a spare exists")

    monkeypatch.setattr(ev.time, "sleep", no_sleep)

    class Broken:
        def chat(self, *a, **k):
            raise RuntimeError("timed out")

    good = StubClient()
    bad = SimpleNamespace(key="bad:m", provider="bad", model="m", client=Broken())
    same = SimpleNamespace(key="ok:m2", provider="ok", model="m2", client=good)
    other = SimpleNamespace(key="new:m", provider="new", model="m", client=good)
    first = SimpleNamespace(key="ok:m", provider="ok", model="m", client=good)
    panel = ev.Panel([bad, first], ev.JudgeCache(tmp_path / "c.jsonl"), [same, other])
    reply = panel.ask(bad, "system", {"items": [{"id": 1, "A": "good", "B": "x"}]})
    assert reply != {}
    # The spare from a provider not already judging is preferred.
    assert [r.key for r in panel.routes] == ["new:m", "ok:m"]
    # A later call with the replaced judge goes to its replacement.
    assert panel.ask(bad, "system", {"items": [{"id": 2, "A": "good", "B": "x"}]}) != {}
    assert [r.key for r in panel.routes] == ["new:m", "ok:m"]


class _ScriptedJudge:
    """Absolute judge that marks the given line ids as major errors and leaves out the lines listed in skip."""

    def __init__(self, major, skip=()):
        self.major, self.skip = set(major), set(skip)

    def chat(self, model, messages, temperature=0.0, json_mode=False):
        items = json.loads(messages[1]["content"])["items"]
        return json.dumps({"results": [{"id": i["id"], "errors": [{"type": "meaning", "severity": "major"}]
                                        if i["id"] in self.major else []}
                                       for i in items if i["id"] not in self.skip]})


def test_absolute_uses_the_judges_majority(tmp_path):
    judges = [SimpleNamespace(key=f"p{k}:m", provider=f"p{k}", model="m", client=c) for k, c in
              enumerate([_ScriptedJudge({0, 1}), _ScriptedJudge({0}), _ScriptedJudge({0, 2}, skip={3})])]
    result = ev.Result("ep", "tr", "ar", [U(i, i, i + 1, f"s{i}", f"t{i}") for i in range(4)], {})
    report = ev.absolute(result, ev.Panel(judges, ev.JudgeCache(tmp_path / "c.jsonl")))
    # Line 0: 3 of 3 say major; lines 1 and 2: 1 of 3 each; line 3: answered by 2 judges, no errors.
    assert report["judged_lines"] == 4 and report["major_error_rate"] == 0.25


def test_pairwise_missing_answer_is_not_a_tie(tmp_path):
    class Silent:
        def chat(self, *a, **k):
            return json.dumps({"results": []})

    routes = [SimpleNamespace(key=f"p{k}:m", provider=f"p{k}", model="m", client=Silent()) for k in range(2)]
    old = ev.Result("ep", "tr", "ar", [U(i, i, i + 1, f"s{i}", f"bad {i}") for i in range(3)], {})
    new = ev.Result("ep", "tr", "ar", [U(i, i, i + 1, f"s{i}", f"good {i}") for i in range(3)], {})
    report = ev.pairwise(old, new, ev.Panel(routes, ev.JudgeCache(tmp_path / "c.jsonl")))
    assert report["translation"]["ties"] == 0 and report["translation"]["unjudged"] == 3


def test_panel_reset_restores_the_starting_judges(tmp_path):
    a, b, spare = (SimpleNamespace(key=f"{k}:m", provider=k, model="m", client=None) for k in ("a", "b", "s"))
    panel = ev.Panel([a, b], ev.JudgeCache(tmp_path / "c.jsonl"), [spare])
    assert panel._replace(a) is spare and [r.key for r in panel.routes] == ["s:m", "b:m"]
    panel.reset()
    assert [r.key for r in panel.routes] == ["a:m", "b:m"] and panel.spares == [spare]
