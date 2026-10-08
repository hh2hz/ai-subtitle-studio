"""Episode brief (task 2.1, D-055)."""

import json

from app.core import brief as br
from app.core.llm_providers import ChatClient, ModelRanking, ProviderPool, ProviderSpec
from tests.fake_llm_api import FakeLlmApi

UNITS = [{"id": i, "start": i * 2.0, "text": t} for i, t in
         enumerate(["Nevzat abi geldi mi?", "Gelmedi.", "Polat nerede?"])]


def _pool(*servers):
    clients = []
    for i, server in enumerate(servers):
        client = ChatClient(ProviderSpec(f"p{i}", server.url, prefer=("good",)), "k")
        client.discover_models()
        clients.append(client)
    return ProviderPool(clients, ModelRanking(floor=0))


def test_validate_drops_invalid_items_and_fixes_enums():
    data = {"summary": 5, "characters": [{"name": "Polat", "gender": "x"}, {"gender": "male"}, "bad"],
            "relations": [{"from": "A", "to": "B", "register": "weird"}, {"from": "A"}],
            "terms": [{"source": "abi", "meaning": "elder brother"}, {}]}
    b = br.validate_brief(data)
    assert b["summary"] == "" and [c["name"] for c in b["characters"]] == ["Polat"]
    assert b["characters"][0]["gender"] == "unknown" and b["relations"][0]["register"] == "informal"
    assert len(b["relations"]) == 1 and len(b["terms"]) == 1
    assert br.is_empty(br.validate_brief({})) and br.is_empty(None)


def test_trim_keeps_only_people_and_terms_in_the_lines():
    b = br.validate_brief({"summary": "s", "characters": [
        {"name": "Polat Alemdar", "gender": "male"}, {"name": "Elif", "gender": "female", "aliases": ["Elifim"]},
        {"name": "Memati", "gender": "male"}],
        "relations": [{"from": "Memati", "to": "Polat Alemdar", "address": "abi"}],
        "terms": [{"source": "abi", "meaning": "brother"}, {"source": "efendim", "meaning": "sir"}]})
    t = br.trim_brief(b, ["Polat nerede abi?", "Elifim geldi."])
    assert [c["name"] for c in t["characters"]] == ["Polat Alemdar", "Elif"]
    assert len(t["relations"]) == 1 and [x["source"] for x in t["terms"]] == ["abi"] and t["summary"] == "s"
    assert br.trim_brief(None, ["x"]) is None


def test_merge_keeps_first_entry_per_name():
    a = br.validate_brief({"summary": "one", "characters": [{"name": "Polat", "gender": "male"}]})
    b = br.validate_brief({"summary": "two", "characters": [{"name": "polat", "gender": "female"},
                                                            {"name": "Elif", "gender": "female"}]})
    m = br.merge_briefs([a, b])
    assert m["summary"] == "one two" and [(c["name"], c["gender"]) for c in m["characters"]] == \
        [("Polat", "male"), ("Elif", "female")]


def test_failed_provider_is_retried_on_another_provider():
    with FakeLlmApi(behaviour="rate_limit") as limited, FakeLlmApi() as good:
        result = br.build_brief(_pool(limited, good), "tr", "ar", UNITS, {"title": "t"}, sleep=lambda s: None)
    assert result and [c["name"] for c in result["characters"]] == ["Nevzat"]
    assert len(good.requests) == 1 and "Nevzat abi" in json.loads(good.requests[0]["messages"][1]["content"])[
        "transcript"]


def test_no_brief_when_every_provider_fails():
    with FakeLlmApi(behaviour="garbage") as bad:
        assert br.build_brief(_pool(bad), "tr", "ar", UNITS, {}, sleep=lambda s: None) is None


def test_long_transcript_is_split_into_parts(monkeypatch):
    monkeypatch.setattr(br, "PART_CHARS", 30)
    assert len(br._parts(UNITS)) == 3
    monkeypatch.setattr(br, "MAX_PARTS", 2)
    assert len(br._parts(UNITS)) == 2


def test_brief_reaches_block_payloads():
    from app.core.llm_translation import LlmRefiner

    with FakeLlmApi() as server:
        refiner = LlmRefiner(_pool(server), "tr", "ar", {"title": "t"}, sleep=lambda s: None, review=False)
        refiner.brief = br.validate_brief({"summary": "s", "characters": [
            {"name": "Nevzat", "gender": "male"}, {"name": "Elif", "gender": "female"}]})
        refiner.translate_block([{"id": 0, "source": "Nevzat geldi.", "draft": ""}], [], [], {})
    payload = json.loads(server.requests[-1]["messages"][1]["content"])
    assert [c["name"] for c in payload["brief"]["characters"]] == ["Nevzat"]
