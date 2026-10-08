"""Task 2.3: second-model double check of risky lines (D-104)."""

import json

import pytest

from app.core import risk
from app.core.llm_providers import ChatClient, ModelRanking, ProviderPool, ProviderSpec
from app.core.llm_translation import BlockResult, LlmRefiner
from tests.fake_llm_api import FakeLlmApi
from tests.fakes import FakeAsrEngine, FakeTranslator
from tests.test_pipeline import _pipeline, media  # noqa: F401

# -- risky-line rules --------------------------------------------------------------------------------------------


def reasons(source, target, tgt="en", src="tr", **kw):
    return risk.risk_reasons(source, target, tgt, src, **kw)


def test_a_good_line_is_not_risky():
    assert reasons("Nereye gidiyorsun?", "Where are you going?") == []


def test_each_rule_marks_a_line_risky():
    assert "check_line" in reasons("Kapıyı aç 125", "Open the door")                     # number lost
    assert "name_mismatch" in reasons("Merhaba", "Hello there friend", name_issue=True)
    assert "weak_model" in reasons("Merhaba", "Hello there friend", weak=True)
    assert "length_ratio" in reasons("Bu çok uzun bir cümle olarak söylenmiş", "Yes")
    assert "low_audio" in reasons("Merhaba arkadaşlar", "Hello friends", audio_confidence=0.3)
    assert "low_audio" not in reasons("Merhaba arkadaşlar", "Hello friends", audio_confidence=0.9)


def test_lost_negation_and_questions_are_found_per_language():
    assert "negation_lost" in reasons("Gelmedi.", "He came.")
    assert "negation_lost" in reasons("Bu doğru değil", "This is true")
    assert "negation_lost" not in reasons("Gelmedi.", "He did not come.")
    assert "negation_lost" not in reasons("Tamam, hemen geliyorum.", "Okay, I am coming right away.")   # not a verb
    assert "negation_lost" in reasons("I do not know", "Je sais", tgt="fr", src="en")
    assert "negation_lost" not in reasons("I do not know", "Je ne sais pas", tgt="fr", src="en")
    assert "negation_lost" in reasons("Ich weiß es nicht", "I know it", tgt="en", src="de")
    assert "question_lost" in reasons("Nereye gidiyorsun?", "You are going somewhere.")
    assert "question_lost" not in reasons("Nereye gidiyorsun?", "\u0623\u064a\u0646 \u062a\u0630\u0647\u0628\u061f", tgt="ar")
    assert "negation_lost" not in reasons("Gelmedi.", "text", tgt="xx")        # no word list: no claim


def test_select_keeps_the_riskiest_lines_and_caps_the_count():
    many = {i: ["a"] * (1 + i % 3) for i in range(30)}
    chosen = risk.select(many)
    assert len(chosen) == risk.MAX_RISKY_PER_BLOCK and list(chosen) == sorted(chosen)
    assert sum(len(r) == 3 for r in chosen.values()) == 10
    assert risk.select({1: [], 2: ["x"]}) == {2: ["x"]}


# -- the double check ----------------------------------------------------------------------------------------------

LINES = [{"id": 0, "source": "Gelmedi.", "draft": ""}, {"id": 1, "source": "Merhaba arkadaşlar", "draft": ""}]


def _pool(*servers, floor=0, scores=None):
    clients = []
    for i, server in enumerate(servers):
        client = ChatClient(ProviderSpec(f"p{i}", server.url, prefer=("good",)), "k")
        client.discover_models()
        clients.append(client)
    return ProviderPool(clients, ModelRanking(scores=scores or {}, floor=floor))


def _refiner(pool, tgt="en", **kw):
    return LlmRefiner(pool, "tr", tgt, {"title": "t"}, sleep=lambda s: None, review=True, **kw)


def _result():
    return BlockResult(translations={0: "He came.", 1: "Hello friends"}, translator="p0:good-model")


def _sent(server):
    return [json.loads(r["messages"][1]["content"]) for r in server.requests]


def _check(servers, choice="B", result=None, **kw):
    refiner = _refiner(_pool(*servers, **kw))
    result = result or _result()
    refiner._double_check(LINES, [{"source": "a", "translation": "b"}], [{"source": "c"}], {}, result)
    return result


def test_judge_picks_b_and_b_replaces_a_when_it_passes_the_checks():
    with FakeLlmApi() as a, FakeLlmApi(translations={"Gelmedi.": "He did not come."}) as b, \
            FakeLlmApi(judge_choice="B") as judge:
        result = _check([a, b, judge])
    assert result.translations[0] == "He did not come." and result.second == {0: "p1:good-model"}
    assert result.risky == {0: ["negation_lost"]} and result.disagreement == {} and result.judge == "p2:good-model"
    assert result.translations[1] == "Hello friends"
    # Only the risky line was sent to the second model and to the judge, each time as ONE request.
    assert [l["id"] for l in _sent(b)[0]["lines"]] == [0] and len(b.requests) == 1 and len(judge.requests) == 1
    judged = _sent(judge)[0]["lines"][0]
    assert judged["A"] == "He came." and judged["B"] == "He did not come." and "max_chars" not in judged
    assert result.double_check_requests == 2 and not a.requests


def test_judge_picks_a_keeps_the_line_without_a_flag():
    with FakeLlmApi() as a, FakeLlmApi(translations={"Gelmedi.": "He did not come."}) as b, \
            FakeLlmApi(judge_choice="A") as judge:
        result = _check([a, b, judge])
    assert result.translations[0] == "He came." and result.disagreement == {} and result.second == {}


def test_both_wrong_keeps_a_and_records_both_candidates():
    with FakeLlmApi() as a, FakeLlmApi(translations={"Gelmedi.": "He did not come."}) as b, \
            FakeLlmApi(judge_choice="both_wrong") as judge:
        result = _check([a, b, judge])
    assert result.translations[0] == "He came."
    assert result.disagreement == {0: {"choice": "both_wrong", "reason": "test reason", "a": "He came.",
                                       "b": "He did not come."}}


def test_b_that_fails_the_rule_check_is_rejected_and_flagged():
    with FakeLlmApi() as a, FakeLlmApi(translations={"Gelmedi.": "He came 12 times."}) as b, \
            FakeLlmApi(judge_choice="B") as judge:
        result = _check([a, b, judge])
    assert result.translations[0] == "He came." and result.disagreement[0]["choice"] == "B_rejected"
    assert result.disagreement[0]["b"] == "He came 12 times." and result.second == {}


def test_second_model_is_another_provider_and_judge_falls_back_to_the_second_route():
    with FakeLlmApi() as a, FakeLlmApi(translations={"Gelmedi.": "He did not come."}, judge_choice="B") as b:
        result = _check([a, b])
    assert not a.requests and len(b.requests) == 2           # translation, then the verdict by the same route
    assert result.translations[0] == "He did not come."
    with FakeLlmApi() as only:
        result = _check([only])
    assert result.risky and result.second == {} and "no second provider" in result.notes[0] and len(only.requests) == 0


def test_models_below_the_quality_floor_are_never_used_for_the_double_check():
    with FakeLlmApi() as a, FakeLlmApi() as weak:
        result = _check([a, weak], floor=80, scores={"p0:good-model": 90, "p1:good-model": 50})
    assert not weak.requests and "no second provider" in result.notes[0]


def test_nothing_is_sent_when_no_line_is_risky_or_the_check_is_off():
    with FakeLlmApi() as a, FakeLlmApi() as b:
        refiner = _refiner(_pool(a, b))
        calm = BlockResult(translations={0: "He did not come.", 1: "Hello friends"}, translator="p0:good-model")
        refiner._double_check(LINES, [], [], {}, calm)
        assert not a.requests and not b.requests and calm.risky == {}
    with FakeLlmApi() as a, FakeLlmApi() as b:
        off = LlmRefiner(_pool(a, b), "tr", "en", {}, sleep=lambda s: None, review=False)
        result = off.translate_block(LINES, [], [], {})
        assert len(a.requests) == 1 and not b.requests and result.risky == {}


def test_a_failing_judge_keeps_a_without_a_flag():
    with FakeLlmApi() as a, FakeLlmApi(translations={"Gelmedi.": "He did not come."}) as b, \
            FakeLlmApi(behaviour="garbage") as judge:
        result = _check([a, b, judge])
    assert result.translations[0] == "He came." and result.disagreement == {}


def test_correct_mode_double_checks_too():
    with FakeLlmApi() as a, FakeLlmApi(translations={"Gelmedi.": "He did not come."}, judge_choice="B") as b:
        refiner = _refiner(_pool(a, b), mode="correct")
        result = refiner.translate_block([{"id": 0, "source": "Gelmedi.", "draft": "He came."}], [], [], {})
    assert result.translations[0] == "He did not come." and 0 in result.changed and result.second


# -- pipeline ------------------------------------------------------------------------------------------------------


def _factory(*servers, **kw):
    from tests.test_pipeline import _refiner_factory
    return _refiner_factory(*servers, **kw)


def test_pipeline_flags_disagreements_lists_both_candidates_and_has_no_reviewer_flag(tmp_path, media):  # noqa: F811
    # The fake source text "Cumle 0 burada." is Latin; with target "fr" the lines are risky when A copies the source.
    copy_source = {"Cumle 0 burada.": "Cumle 0 burada.", "Yarim cumle 1": "Yarim cumle 1"}
    with FakeLlmApi(translations=copy_source) as a, FakeLlmApi(translations={"Cumle 0 burada.": "Phrase 0 ici."}) as b, \
            FakeLlmApi(judge_choice="both_wrong") as judge:
        pipe, _, _ = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), target="fr",
                               refiner_factory=_factory(a, b, judge, mode="translate", review=True))
        result = pipe.run()
    from app.utils.atomic import read_json
    doc = read_json(result.outputs["master_transcript"])
    flagged = [u for u in doc["translation_units"] if "ai_disagreement" in u["flags"]]
    assert flagged and all("reviewer_suggestion" not in u["flags"] for u in doc["translation_units"])
    assert flagged[0]["disagreement"]["choice"] == "both_wrong" and flagged[0]["disagreement"]["a"]
    text = (result.output_dir / "work" / "ReviewRequired.txt").read_text(encoding="utf-8")
    assert "AI DISAGREEMENT (both_wrong)" in text and "  A: " in text and "  B: " in text
    dc = result.stats["refine"]["double_check"]
    assert dc["risky_lines"] >= 1 and dc["requests"] >= 2 and dc["disagreements"] == len(flagged)
    assert flagged[0]["confidence"] in ("MEDIUM", "LOW")


def test_ai_disagreement_is_a_medium_flag():
    from app.core import confidence
    assert "ai_disagreement" not in confidence.SEVERE_DEFAULT
    assert confidence.category(["ai_disagreement"], 0.95) == confidence.MEDIUM
    assert confidence.category(["name_mismatch"], 0.95) == confidence.LOW


def test_double_check_is_on_by_default():
    from app.database.settings import DEFAULTS
    assert DEFAULTS["llm_review"] is True
