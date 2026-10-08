"""Tests for Task 2.7: reading-speed budget and single-request condense step."""

import json

from app.core import style
from app.core.llm_providers import ChatClient, ModelRanking, ProviderPool, ProviderSpec
from app.core.llm_translation import LlmRefiner, _source_line
from tests.fake_llm_api import FakeLlmApi


def _client(server, name="p1", models=("good-model",)):
    spec = ProviderSpec(name, server.url, prefer=("good",), fallback_models=models)
    client = ChatClient(spec, "secret")
    client.models = list(models)
    return client


def _refiner(*clients, **kw):
    pool = ProviderPool(list(clients), ModelRanking(floor=0))
    return LlmRefiner(pool, "tr", "ar", {"title": "Show"}, sleep=lambda s: None, **kw)


def test_style_max_chars_budget():
    # 0.5s * 17 cps = 8.5 -> min is 12
    assert style.max_chars(0.5, "ar") == 12
    assert style.max_chars(0.0, "ar") == 12
    assert style.max_chars(-1.0, "ar") == 12

    # 2.0s * 17 cps = 34
    assert style.max_chars(2.0, "ar") == 34

    # 10.0s * 17 cps = 170 -> capped at 2 * max_line (2 * 42 = 84 for ar)
    assert style.max_chars(10.0, "ar") == 84

    # Japanese: max_line is 16 -> capped at 2 * 16 = 32
    assert style.max_chars(5.0, "ja") == 32

    # Unknown language fallback: cps 17, max_line 42
    assert style.max_chars(2.0, "xx") == 34
    assert style.max_chars(10.0, "xx") == 84


def test_source_line_includes_max_chars():
    line_with = {"id": 1, "source": "Merhaba", "max_chars": 25}
    assert _source_line(line_with) == {"id": 1, "source": "Merhaba", "max_chars": 25}

    line_without = {"id": 2, "source": "Evet"}
    assert _source_line(line_without) == {"id": 2, "source": "Evet"}


def test_condense_triggers_when_translation_exceeds_budget():
    # max_chars = 12 -> 1.2 * 12 = 14.4 chars.
    # Initial translation: 26 characters (well over 14.4).
    # Model condenses to 9 characters (valid, shorter, passes check_line).
    with FakeLlmApi() as server:
        client = _client(server)
        refiner = _refiner(client, review=False)

        lines = [{"id": 0, "source": "Evet tamam.", "draft": "", "max_chars": 12}]

        def reply(body):
            sys_msg = body["messages"][0]["content"]
            if "Shorten the subtitle lines" in sys_msg:
                # "\u0646\u0639\u0645 \u062a\u0645\u0627\u0645\u0627" = 9 chars
                return json.dumps({"lines": [{"id": 0, "text": "\u0646\u0639\u0645 \u062a\u0645\u0627\u0645\u0627"}]})
            # "\u0646\u0639\u0645 \u0628\u0627\u0644\u062a\u0627\u0643\u064a\u062f \u0647\u0630\u0627 \u0635\u062d\u064a\u062d \u062c\u062f\u0627" = 26 chars
            return json.dumps({"lines": [{"id": 0, "text": "\u0646\u0639\u0645 \u0628\u0627\u0644\u062a\u0627\u0643\u064a\u062f \u0647\u0630\u0627 \u0635\u062d\u064a\u062d \u062c\u062f\u0627"}]})

        server.reply = reply
        result = refiner.translate_block(lines, [], [], {})

        assert result.condense_requests == 1
        assert result.translations[0] == "\u0646\u0639\u0645 \u062a\u0645\u0627\u0645\u0627"
        assert len(result.translations[0]) < 26


def test_condense_not_triggered_when_within_budget():
    with FakeLlmApi() as server:
        client = _client(server)
        refiner = _refiner(client, review=False)

        # max_chars = 40. Translation length is 8 (< 1.2 * 40 = 48).
        lines = [{"id": 0, "source": "Evet.", "draft": "", "max_chars": 40}]

        def reply(body):
            # "\u0646\u0639\u0645 \u062d\u0633\u0646\u0627" = 8 chars
            return json.dumps({"lines": [{"id": 0, "text": "\u0646\u0639\u0645 \u062d\u0633\u0646\u0627"}]})

        server.reply = reply
        result = refiner.translate_block(lines, [], [], {})

        assert result.condense_requests == 0
        assert result.translations[0] == "\u0646\u0639\u0645 \u062d\u0633\u0646\u0627"


def test_condense_at_most_one_request_per_block():
    # Multiple lines exceed budget in the same block:
    # Must send exactly ONE condense request with all over-budget lines batched.
    with FakeLlmApi() as server:
        client = _client(server)
        refiner = _refiner(client, review=False)

        lines = [
            {"id": 0, "source": "Birinci cumle.", "draft": "", "max_chars": 12},
            {"id": 1, "source": "Ikinci cumle.", "draft": "", "max_chars": 12},
            {"id": 2, "source": "Ucuncu kisa.", "draft": "", "max_chars": 20},
        ]

        calls = []

        def reply(body):
            calls.append(body)
            sys_msg = body["messages"][0]["content"]
            if "Shorten the subtitle lines" in sys_msg:
                # Condense reply for both 0 and 1
                return json.dumps({"lines": [
                    {"id": 0, "text": "\u0627\u0644\u0627\u0648\u0644 \u0642\u0635\u064a\u0631"},
                    {"id": 1, "text": "\u0627\u0644\u062b\u0627\u0646\u064a \u0642\u0635\u064a\u0631"},
                ]})
            return json.dumps({"lines": [
                {"id": 0, "text": "\u062a\u0631\u062c\u0645\u0629 \u0637\u0648\u064a\u0644\u0629 \u062c\u062f\u0627 \u062c\u062f\u0627 \u062c\u062f\u0627 \u0644\u0644\u0627\u0648\u0644"},
                {"id": 1, "text": "\u062a\u0631\u062c\u0645\u0629 \u0637\u0648\u064a\u0644\u0629 \u062c\u062f\u0627 \u062c\u062f\u0627 \u062c\u062f\u0627 \u0644\u0644\u062b\u0627\u0646\u064a"},
                {"id": 2, "text": "\u0642\u0635\u064a\u0631"},
            ]})

        server.reply = reply
        result = refiner.translate_block(lines, [], [], {})

        assert result.condense_requests == 1
        # Exactly 2 LLM requests in total: 1 translation + 1 condense
        assert len(calls) == 2
        assert result.translations[0] == "\u0627\u0644\u0627\u0648\u0644 \u0642\u0635\u064a\u0631"
        assert result.translations[1] == "\u0627\u0644\u062b\u0627\u0646\u064a \u0642\u0635\u064a\u0631"
        assert result.translations[2] == "\u0642\u0635\u064a\u0631"


def test_condense_rejected_if_longer_or_fails_check():
    with FakeLlmApi() as server:
        client = _client(server)
        refiner = _refiner(client, review=False)

        # Line 0: candidate condensed text is longer -> rejected
        # Line 1: candidate fails check_line (dropped number) -> rejected
        lines = [
            {"id": 0, "source": "Birinci.", "draft": "", "max_chars": 12},
            {"id": 1, "source": "Ikinci 100 kisi.", "draft": "", "max_chars": 12},
        ]

        def reply(body):
            sys_msg = body["messages"][0]["content"]
            if "Shorten the subtitle lines" in sys_msg:
                return json.dumps({"lines": [
                    # Even longer than original (46 chars vs 21 chars)
                    {"id": 0, "text": "\u0647\u0630\u0627 \u0627\u0644\u0646\u0635 \u0627\u0635\u0628\u062d \u0627\u0637\u0648\u0644 \u0628\u0643\u062b\u064a\u0631 \u0645\u0646 \u0627\u0644\u0646\u0635 \u0627\u0644\u0627\u0635\u0644\u064a \u0627\u0644\u0633\u0627\u0628\u0642"},
                    # Dropped number 100 when source has 100, causing numbers_differ check_line failure
                    {"id": 1, "text": "\u0627\u0634\u062e\u0627\u0635"},
                ]})
            return json.dumps({"lines": [
                {"id": 0, "text": "\u0646\u0635 \u0637\u0648\u064a\u0644 \u064a\u062d\u062a\u0627\u062c \u0627\u062e\u062a\u0635\u0627\u0631"},
                {"id": 1, "text": "\u0645\u0627\u0626\u0629 \u0634\u062e\u0635 \u0647\u0646\u0627\u0643 100"},
            ]})

        server.reply = reply
        result = refiner.translate_block(lines, [], [], {})

        assert result.condense_requests == 1
        # Original translations retained because new ones were rejected
        assert result.translations[0] == "\u0646\u0635 \u0637\u0648\u064a\u0644 \u064a\u062d\u062a\u0627\u062c \u0627\u062e\u062a\u0635\u0627\u0631"
        assert result.translations[1] == "\u0645\u0627\u0626\u0629 \u0634\u062e\u0635 \u0647\u0646\u0627\u0643 100"


def test_condense_rejected_if_it_drops_a_glossary_name():
    glossary = {"Memati": "\u0645\u064a\u0645\u0627\u062a\u064a"}
    lines = [{"id": 0, "source": "Dun Memati buraya geldi.", "draft": "", "max_chars": 12}]
    long_text = "\u0645\u064a\u0645\u0627\u062a\u064a \u062c\u0627\u0621 \u0625\u0644\u0649 \u0647\u0646\u0627 \u0623\u0645\u0633 \u0645\u0633\u0627\u0621"
    no_name = "\u062c\u0627\u0621 \u0625\u0644\u0649 \u0647\u0646\u0627 \u0623\u0645\u0633"
    with FakeLlmApi() as server:
        refiner = _refiner(_client(server), review=False)

        def reply(body):
            if "Shorten the subtitle lines" in body["messages"][0]["content"]:
                return json.dumps({"lines": [{"id": 0, "text": no_name}]})
            return json.dumps({"lines": [{"id": 0, "text": long_text}]})

        server.reply = reply
        result = refiner.translate_block(lines, [], [], glossary)
    assert result.condense_requests == 1
    assert result.translations[0] == long_text


def test_condense_does_not_use_a_cooled_down_locked_model():
    with FakeLlmApi() as server:
        client = _client(server, models=("model-a", "model-b"))
        refiner = _refiner(client, review=False)
        locked = next(r for r in refiner.pool.routes() if r.model == "model-a")
        refiner._locked_route = locked
        refiner.pool.cool_down(locked.key, 3600)
        used = []

        def fake_call(route, system, payload):
            used.append(route.model)
            return {"model": route.key, "data": {"lines": []}}

        refiner._call = fake_call
        long_text = "x" * 60
        from app.core.llm_translation import BlockResult
        result = BlockResult(translations={0: long_text}, translator="t")
        lines = [{"id": 0, "source": "Kisa", "draft": "", "max_chars": 12}]
        refiner._condense(lines, result)
        assert used == ["model-b"]
        assert result.condense_requests == 1
