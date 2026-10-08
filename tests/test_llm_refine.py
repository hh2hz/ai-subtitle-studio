"""Cloud LLM refinement: providers, model choice, orchestration and the pipeline stage (fake HTTP servers)."""

import json
from dataclasses import replace

import pytest

from app.core import llm_providers as lp
from app.core.llm_providers import ChatClient, ModelRanking, ProviderCallError, ProviderPool, choose_models
from app.core.llm_translation import LlmRefiner, extract_json
from app.services.api_keys import load_keys
from tests.fake_llm_api import NEVZAT_AR, FakeLlmApi, fake_text

LINES = [{"id": 0, "source": "Tir hazir mi?", "draft": "draft0"},
         {"id": 1, "source": "Nevzat, gel.", "draft": "draft1 " + NEVZAT_AR}]


def _client(server, name="p1", models=None):
    spec = lp.ProviderSpec(name, server.url, prefer=("good",), fallback_models=("good-model",))
    client = ChatClient(spec, "secret")
    client.discover_models() if models is None else setattr(client, "models", models)
    return client


def _refiner(*clients, ranking=None, **kw):
    pool = ProviderPool(list(clients), ranking or ModelRanking(floor=0))
    return LlmRefiner(pool, "tr", "ar", {"title": "Show"}, sleep=lambda s: None, **kw)


# -- model selection ------------------------------------------------------------------------------

def test_choose_models_prefers_pattern_then_newest():
    spec = lp.PROVIDERS["gemini"]
    ids = ["models/gemini-2.5-flash", "models/gemini-3.8-flash", "models/gemini-3.8-flash-lite",
           "models/gemini-3.8-pro", "models/text-embedding-004", "models/gemini-3.8-flash-image"]
    # Every text model is kept (Pro and Lite too), best estimate first; embeddings and image models are not.
    assert choose_models(spec, ids) == ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-3.8-pro",
                                        "gemini-3.8-flash-lite"]


def test_choose_models_require_free_and_fallbacks():
    assert choose_models(lp.PROVIDERS["kilo"], ["deepseek/deepseek-v3", "deepseek/deepseek-v3:free"]) == \
        ["deepseek/deepseek-v3:free"]
    assert choose_models(lp.PROVIDERS["cloudflare"], None) == list(lp.PROVIDERS["cloudflare"].fallback_models)
    assert choose_models(lp.PROVIDERS["groq"], ["whisper-large-v3", "llama-3.3-70b-versatile"]) == \
        ["llama-3.3-70b-versatile"]


def test_build_clients_respects_order_and_account_id():
    keys = {"cloudflare": {"api_key": "k", "account_id": "ACC"}, "gemini": {"api_key": "g"}, "aionlabs": {"api_key": "x"},
            "unknown": {"api_key": "y"}}
    clients = lp.build_clients(keys)
    assert [c.name for c in clients] == ["gemini", "cloudflare", "aionlabs"]
    assert "/accounts/ACC/ai/v1" in clients[1].base_url


def test_extract_json_variants():
    assert extract_json('<think>x</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json")


# -- client ---------------------------------------------------------------------------------------

def test_client_discovers_and_chats():
    with FakeLlmApi(models=("bad-x", "good-model")) as server:
        client = _client(server)
        assert client.models == ["good-model", "bad-x"]
        reply = client.chat("good-model", [{"role": "system", "content": "x"},
                                           {"role": "user", "content": json.dumps({"lines": LINES})}])
        assert "AR(Tir hazir mi?)" in reply
        assert server.requests[0]["auth"] == "Bearer secret"


def test_client_errors():
    with FakeLlmApi(behaviour="unauthorized") as server:
        with pytest.raises(ProviderCallError) as info:
            _client(server, models=["m"]).chat("m", [])
        assert info.value.fatal
    with FakeLlmApi(behaviour="rate_limit", retry_after="7") as server:
        with pytest.raises(ProviderCallError) as info:
            _client(server, models=["m"]).chat("m", [])
        assert info.value.rate_limited and info.value.retry_after == 7.0


# -- refiner --------------------------------------------------------------------------------------

def test_translate_block_payload_and_names():
    with FakeLlmApi() as translator:
        refiner = _refiner(_client(translator, "a"), review=False)
        result = refiner.translate_block(LINES, [{"source": "x", "translation": "y"}], [{"source": "z"}],
                                         {"Memati": "MEMATI-AR"})
        assert result.translations == {0: "AR(Tir hazir mi?)", 1: fake_text("Nevzat, gel.")}
        assert result.translator == "a:good-model" and result.names == {"Nevzat": NEVZAT_AR}
        payload = json.loads(translator.requests[0]["messages"][1]["content"])
        assert payload["lines"][0] == {"id": 0, "source": "Tir hazir mi?"}         # no draft in translate mode
        assert payload["glossary"] == [{"source": "Memati", "target": "MEMATI-AR"}]
        assert payload["previous_lines"] and payload["next_lines"] and payload["media"]["title"] == "Show"
        assert result.changed is None


def test_review_can_be_disabled():
    with FakeLlmApi() as server:
        refiner = _refiner(_client(server), review=False)
        refiner.translate_block(LINES, [], [], {})
        assert len(server.requests) == 1


def test_correct_mode_returns_only_corrections():
    with FakeLlmApi(reviewer_fix=0) as server:
        client = _client(server)
        refiner = _refiner(client, mode="correct", review=False)
        result = refiner.translate_block(LINES, [{"source": "x", "translation": "y"}], [{"source": "z"}],
                                         {"Memati": "MEMATI-AR"})
        assert result.translations == {0: "FIXED", 1: "draft1 " + NEVZAT_AR} and result.changed == {0}
        assert result.translator == "p1:good-model" and result.reviewer is None
        assert result.names == {"Nevzat": NEVZAT_AR}
        assert len(server.requests) == 1
        payload = json.loads(server.requests[0]["messages"][1]["content"])
        assert payload["lines"][0] == {"id": 0, "source": "Tir hazir mi?", "translation": "draft0"}
        assert payload["glossary"] == [{"source": "Memati", "target": "MEMATI-AR"}] and payload["previous_lines"]
        assert client.usage == {"requests": 1, "prompt_tokens": 100, "completion_tokens": 20}


def test_correct_mode_accepts_all_and_skips_failed_provider():
    with FakeLlmApi(behaviour="rate_limit", retry_after="999") as bad, FakeLlmApi() as good:
        refiner = _refiner(_client(bad, "a"), _client(good, "b"), mode="correct")
        result = refiner.translate_block(LINES, [], [], {})
        assert result.translations == {0: "draft0", 1: "draft1 " + NEVZAT_AR} and result.changed == set()
        assert result.translator == "b:good-model"


def test_correct_mode_without_providers_handles_nothing():
    with FakeLlmApi(behaviour="garbage") as server:
        result = _refiner(_client(server), mode="correct").translate_block(LINES, [], [], {})
        assert result.translations == {} and result.translator is None and result.notes


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        LlmRefiner(ProviderPool([]), "tr", "ar", {}, mode="draft")


def test_missing_lines_go_to_the_next_provider():
    with FakeLlmApi(behaviour="drop_first") as a, FakeLlmApi() as b:
        result = _refiner(_client(a, "a"), _client(b, "b"), review=False).translate_block(LINES, [], [], {})
        assert result.translations[0] == "AR(Tir hazir mi?)" and result.translations[1] == fake_text("Nevzat, gel.")
        second = json.loads(b.requests[0]["messages"][1]["content"])
        assert [l["id"] for l in second["lines"]] == [0]


def test_rate_limited_and_unauthorized_providers_are_skipped():
    with FakeLlmApi(behaviour="rate_limit", retry_after="0") as a, FakeLlmApi(behaviour="unauthorized") as b, \
            FakeLlmApi() as c:
        pool_clients = [_client(a, "a", ["m"]), _client(b, "b", ["m"]), _client(c, "c")]
        refiner = _refiner(*pool_clients, review=False, ranking=ModelRanking(scores={"a:m": 90, "b:m": 80}, floor=0))
        result = refiner.translate_block(LINES, [], [], {})
        assert len(result.translations) == 2 and result.translator == "c:good-model"
        assert len(a.requests) == 2                      # one retry after Retry-After: 0, then cool-down
        assert "b" in refiner.pool.disabled and "a:m" in refiner.pool.cooldown_until
        assert [r.key for r in refiner.pool.available()] == ["c:good-model"]


def test_all_providers_fail_returns_empty():
    with FakeLlmApi(behaviour="garbage") as a:
        result = _refiner(_client(a, "a", ["m"])).translate_block(LINES, [], [], {})
        assert result.translations == {} and result.notes


# -- keys -----------------------------------------------------------------------------------------

def test_load_keys_merges_and_ignores_comments(tmp_path):
    data_file, project_file = tmp_path / "a.json", tmp_path / "b.json"
    data_file.write_text(json.dumps({"groq": {"api_key": "new"}}), encoding="utf-8")
    project_file.write_text(json.dumps({"_comment": "x", "groq": {"api_key": "old"}, "gemini": {"api_key": "g"},
                                        "nvidia": {"api_key": ""}}), encoding="utf-8")
    keys = load_keys([data_file, project_file])
    assert keys == {"groq": {"api_key": "new"}, "gemini": {"api_key": "g"}}


def test_no_quota_errors_disable_provider():
    err = ProviderCallError('zai HTTP 429: {"error":{"code":"1113","message":"Insufficient balance or no resource package. Please recharge."}}', 429)
    assert err.fatal and not err.rate_limited
    assert ProviderCallError("x HTTP 402: Insufficient balance", 402).fatal
    assert ProviderCallError("x HTTP 404: Function not found", 404).model_unavailable


def test_unavailable_model_is_removed_and_next_model_used():
    with FakeLlmApi() as server:
        client = _client(server, "a", ["missing-model", "good-model"])
        original = client.chat

        def chat(model, messages, **kw):
            if model == "missing-model":
                raise ProviderCallError("a HTTP 404: not found", 404)
            return original(model, messages, **kw)

        client.chat = chat
        refiner = _refiner(client, review=False, ranking=ModelRanking(scores={"a:missing-model": 90}, floor=0))
        assert len(refiner.translate_block(LINES, [], [], {}).translations) == 2
        assert client.models == ["good-model"]


# -- model routing --------------------------------------------------------------------------------

def test_rate_limited_model_falls_back_to_next_model_of_same_provider():
    with FakeLlmApi() as server:
        client = _client(server, "g", ["best", "second"])
        original = client.chat

        def chat(model, messages, **kw):
            if model == "best":
                raise ProviderCallError("g HTTP 429: GenerateRequestsPerDayPerProjectPerModel-FreeTier", 429)
            return original(model, messages, **kw)

        client.chat = chat
        refiner = _refiner(client, review=False, ranking=ModelRanking(scores={"g:best": 95, "g:second": 90}, floor=0))
        result = refiner.translate_block(LINES, [], [], {})
        assert result.translator == "g:second" and len(result.translations) == 2
        assert refiner.pool.cooldown_until["g:best"] - lp.time.monotonic() > 3600       # daily quota: whole job
        assert "g" not in refiner.pool.disabled


def test_ranking_orders_models_across_providers_and_excludes(tmp_path):
    shipped, user = tmp_path / "a.json", tmp_path / "b.json"
    shipped.write_text(json.dumps({"floor": 70, "models": {"b:x": {"score": 60}, "a:y": {"score": 99}}}), "utf-8")
    user.write_text(json.dumps({"models": {"b:x": {"score": 95}, "a:y": {"exclude": True}}}), "utf-8")
    ranking = ModelRanking.load(shipped, user, tmp_path / "missing.json")
    assert ranking.floor == 70 and ranking.scores == {"b:x": 95.0} and ranking.excluded == {"a:y"}
    with FakeLlmApi() as server:
        pool = ProviderPool([_client(server, "a", ["y", "z"]), _client(server, "b", ["x"])], ranking)
        routes = pool.routes()
        assert [r.key for r in routes] == ["b:x", "a:z"] and routes[0].measured and not routes[1].measured
        assert [r.key for r in pool.available(min_score=70)] == ["b:x"]


def test_correct_mode_uses_only_models_above_floor_and_translate_flags_weak():
    with FakeLlmApi() as server:
        ranking = ModelRanking(scores={"p1:good-model": 40}, floor=65)
        correct = _refiner(_client(server), mode="correct", ranking=ranking)
        assert not correct.has_routes()
        assert correct.translate_block(LINES, [], [], {}).translations == {} and not server.requests
        translate = _refiner(_client(server), review=False, ranking=ranking)
        result = translate.translate_block(LINES, [], [], {})
        assert len(result.translations) == 2 and result.weak == {0, 1}


def test_reasoning_effort_sent_and_dropped_when_rejected():
    assert lp.reasoning_effort("gemini", "gemini-2.5-flash") == "none"
    assert lp.reasoning_effort("gemini", "gemini-2.5-pro") == "low"
    assert lp.reasoning_effort("gemini", "gemini-3.8-flash") == "low"
    assert lp.reasoning_effort("gemini", "gemma-4-31b-it") is None
    assert lp.reasoning_effort("groq", "openai/gpt-oss-120b") == "low"
    assert lp.reasoning_effort("cohere", "command-a-03-2025") is None
    with FakeLlmApi(reject_reasoning=True) as server:
        spec = lp.ProviderSpec("gemini", server.url, prefer=())
        client = ChatClient(spec, "k")
        client.models = ["gemini-3.8-flash"]
        client.chat("gemini-3.8-flash", [{"role": "system", "content": "x"},
                                         {"role": "user", "content": json.dumps({"lines": LINES})}])
        client.chat("gemini-3.8-flash", [{"role": "system", "content": "x"},
                                         {"role": "user", "content": json.dumps({"lines": LINES})}])
        sent = [r.get("reasoning_effort") for r in server.requests]
        assert sent == ["low", None, None]             # rejected once, then never sent again
        assert client.usage_by_model["gemini-3.8-flash"]["requests"] == 2


def test_heuristic_scores_rank_families():
    g = lp.PROVIDERS["gemini"]
    assert lp.heuristic_score(g, "gemini-3.8-flash") > lp.heuristic_score(g, "gemini-3.1-pro") > \
        lp.heuristic_score(g, "gemma-4-31b-it") > lp.heuristic_score(g, "gemini-3.5-flash-lite")
    groq = lp.PROVIDERS["groq"]
    assert lp.heuristic_score(groq, "llama-3.1-8b-instant") < 50
    assert not lp.usable_model(g, "gemini-2.5-flash-preview-tts") and lp.usable_model(g, "gemini-3.1-pro")


def test_cloudflare_models_are_discovered_from_model_search():
    with FakeLlmApi(models=("@cf/openai/gpt-oss-120b", "@cf/meta/llama-guard-3-8b", "@cf/qwen/qwen3-30b")) as server:
        spec = replace(lp.PROVIDERS["cloudflare"], base_url=server.url + "/accounts/{account_id}/ai/v1",
                       models_url=server.url + "/accounts/{account_id}/ai/models/search?task=Text%20Generation")
        client = ChatClient(spec, "k", account_id="ACC")
        assert set(client.discover_models()) == {"@cf/openai/gpt-oss-120b", "@cf/qwen/qwen3-30b"}


def test_free_models_by_suffix_or_zero_price():
    zero, paid = {"prompt": "0", "completion": "0"}, {"prompt": "0.000001", "completion": "0.000002"}
    with FakeLlmApi(models=("a/x:free", "b/y", "c/z"), pricing={"b/y": zero, "c/z": paid}) as server:
        client = ChatClient(replace(lp.PROVIDERS["kilo"], base_url=server.url), "k")
        assert set(client.discover_models()) == {"a/x:free", "b/y"}
    assert lp.exclusion_reason(lp.PROVIDERS["kilo"], "c/z") == "not free (no :free)"
    assert lp.exclusion_reason(lp.PROVIDERS["gemini"], "gemini-3-pro-image").startswith("not a text chat model")


def test_model_without_free_quota_is_dropped_not_cooled_down():
    err = ProviderCallError("gemini HTTP 429: Quota exceeded for metric ... limit: 0, model: gemini-3.1-pro", 429)
    assert err.not_free and not err.rate_limited and not err.fatal
    with FakeLlmApi() as server:
        client = _client(server, "g", ["pro", "flash"])
        original = client.chat

        def chat(model, messages, **kw):
            if model == "pro":
                raise err
            return original(model, messages, **kw)

        client.chat = chat
        refiner = _refiner(client, review=False, ranking=ModelRanking(scores={"g:pro": 99}, floor=0))
        assert refiner.translate_block(LINES, [], [], {}).translator == "g:flash"
        assert client.models == ["flash"] and "g:pro" not in refiner.pool.cooldown_until


def test_list_models_tool_updates_exclusions(tmp_path):
    from tools.list_models import update_ranking

    path = tmp_path / "model_ranking.json"
    path.write_text(json.dumps({"models": {"g:old": {"exclude": True, "reason": "x"}, "g:s": {"score": 80}}}), "utf-8")
    added, removed = update_ranking([
        {"key": "g:old", "status": "usable", "detail": ""},
        {"key": "g:s", "status": "not free", "detail": "limit: 0"},
        {"key": "g:busy", "status": "rate limited", "detail": ""},
        {"key": "g:net", "status": "error", "detail": "timeout"}], path)
    models = json.loads(path.read_text(encoding="utf-8"))["models"]
    assert (added, removed) == (1, 1) and "g:old" not in models and "g:busy" not in models and "g:net" not in models
    assert models["g:s"]["exclude"] and models["g:s"]["score"] == 80


def test_gemini_per_minute_quota_is_not_fatal():
    err = ProviderCallError('gemini HTTP 429: [{"error": {"code": 429, "message": "You exceeded your current quota, '
                            'please check your plan and billing details. ... limit: 10, model: gemini-3.8-flash"}}]', 429)
    assert err.rate_limited and not err.fatal and not err.not_free


def test_extract_json_takes_first_object_when_followed_by_more():
    assert extract_json('{"corrections": []}\n{"note": "done"}') == {"corrections": []}


def test_max_tokens_lowered_when_model_caps_it():
    with FakeLlmApi(max_tokens_cap=4096) as server:
        client = _client(server, models=["good-model"])
        message = [{"role": "system", "content": "x"}, {"role": "user", "content": json.dumps({"lines": LINES})}]
        client.chat("good-model", message, max_tokens=6000)
        client.chat("good-model", message, max_tokens=6000)
        assert [r["max_tokens"] for r in server.requests] == [6000, 2048, 2048]


def test_clean_keeps_inner_quotes():
    from app.core.llm_translation import _clean
    assert _clean('"wrapped line"') == "wrapped line"
    assert _clean('labels saying "police"') == 'labels saying "police"'
    assert _clean('a\\nb  c') == "a b c"


# -- names (D-039) ---------------------------------------------------------------------------------

POLAT_AR = "\u0628\u0648\u0644\u0627\u062a"
MURAD_AR = "\u0645\u0631\u0627\u062f"            # the dubbed name used instead of Polat


def test_transliteration_check():
    from app.core import names as nm
    assert nm.transliteration_ok("Polat", POLAT_AR) and not nm.transliteration_ok("Polat", MURAD_AR)
    assert nm.transliteration_ok("Nevzat", NEVZAT_AR) and nm.transliteration_ok("Mehmet Bey", "\u0645\u062d\u0645\u062f")
    assert nm.line_issues("Polat'in kizi", "\u0627\u0628\u0646\u0629 " + MURAD_AR, {"Polat"}, {}) == ["Polat"]
    assert nm.line_issues("Polat'in kizi", "\u0627\u0628\u0646\u0629 " + POLAT_AR, {"Polat"}, {}) == []
    assert nm.line_issues("Karahanli geldi", "x", {"Kara", "Karahanli"}, {}) == ["Karahanli"]   # not "Kara"
    assert nm.line_issues("kara bir gun", "x", {"Kara"}, {}) == []                              # lowercase word


class _DubbedApi(FakeLlmApi):
    """Translates Polat with the dubbed name; fixes it when asked by the name-fix prompt."""

    def reply(self, body):
        system = body["messages"][0]["content"]
        payload = json.loads(body["messages"][1]["content"])
        if "fix person and place names" in system:
            return json.dumps({"lines": [{"id": l["id"], "text": POLAT_AR + " geldi"} for l in payload["lines"]]})
        return json.dumps({"lines": [{"id": l["id"], "text": MURAD_AR + " geldi"} for l in payload["lines"]],
                           "names": [{"source": "Polat", "target": MURAD_AR}]})


def test_dubbed_name_rejected_and_line_fixed():
    lines = [{"id": 0, "source": "Polat geldi.", "draft": ""}]
    with _DubbedApi() as server:
        refiner = _refiner(_client(server), review=False)
        result = refiner.translate_block(lines, [], [], {})
        assert result.names == {} and result.rejected_names == {"Polat": MURAD_AR}
        assert result.translations[0] == POLAT_AR + " geldi" and result.name_issues == {}
        assert len(server.requests) == 2


def test_unfixable_name_is_reported():
    lines = [{"id": 0, "source": "Polat geldi.", "draft": ""}]

    class Stubborn(_DubbedApi):
        def reply(self, body):
            payload = json.loads(body["messages"][1]["content"])
            return json.dumps({"lines": [{"id": l["id"], "text": MURAD_AR} for l in payload["lines"]]})

    with Stubborn() as server:
        result = _refiner(_client(server), review=False).translate_block(lines, [], [], {"Polat": POLAT_AR})
        assert result.name_issues == {0: ["Polat"]}


def test_series_glossary_persists_and_drops_dubbed_names(tmp_path):
    from app.core.names import SeriesGlossary

    path = tmp_path / "glossaries" / "kurtlar-vadisi.ar.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"names": {"Polat": MURAD_AR, "Memati": "x"}, "user": ["Memati"]}), "utf-8")
    glossary = SeriesGlossary.for_series(tmp_path, "Kurtlar Vadisi")
    assert glossary.names == {"Memati": "x"}                     # learned dubbed name dropped, user name kept
    assert glossary.learn({"Polat": POLAT_AR, "Memati": "y", "Nevzat": MURAD_AR}) == 1
    glossary.save()
    again = SeriesGlossary.for_series(tmp_path, "Kurtlar Vadisi")
    assert again.names == {"Memati": "x", "Polat": POLAT_AR} and again.user == {"Memati"}


def test_names_checked_in_any_script():
    from app.core import names as nm
    cases = [("Polat", "Полат", True), ("Polat", "Мурад", False),   # Russian
             ("Polat", "Πολάτ", True),                                                     # Greek
             ("Polat", "פולאט", True),                                                     # Hebrew
             ("Polat", "\u067e\u0648\u0644\u0627\u062a", True),                                                     # Persian
             ("Polat", "폴라트", True),                                                                 # Korean
             ("Polat", "Polat", True), ("Polat", "Murat", False),                                                   # English
             ("Наташа", "Natasha", True), ("Наташа", "Maria", False),  # Russian source
             ("Mr Smith", "\u0633\u0645\u064a\u062b", True)]                                                        # title ignored
    for source, target, expected in cases:
        assert nm.transliteration_ok(source, target) is expected, (source, target)
    assert nm.line_issues("Наташа пришла.", "Maria came.",
                          {"Наташа"}, {})
    assert not nm.supported_target("ja") and nm.supported_target("en") and nm.supported_target("ru")


def test_common_word_is_not_rewritten_as_a_name():
    """'Aslan' is a known name (Aslan Akbey) but here means 'lion': flag, do not rewrite (D-047)."""
    lines = [{"id": 0, "source": "Aslan gibi dovustu.", "draft": ""}]
    with FakeLlmApi() as server:
        refiner = _refiner(_client(server), review=False)
        refiner.known_names.add("Aslan Akbey")
        result = refiner.translate_block(lines, [], [], {})
        assert len(server.requests) == 1                      # no name-fix request
        assert result.translations[0] == "AR(Aslan gibi dovustu.)"


def test_title_alone_is_not_learned_as_a_name():
    from app.core.names import SeriesGlossary
    glossary = SeriesGlossary(None)
    assert glossary.learn({"Abi": "\u0622\u0628\u064a"}) == 0


def test_name_certainty():
    from app.core import names as nm
    assert not nm.certain_in("Aslan Akbey", "Aslan gibi dovustu.")      # sentence start: may be a word
    assert nm.certain_in("Polat", "Sen Polat ile konustun mu?")          # capitalised mid-sentence
    assert nm.core_tokens("Abi") == []


# -- style lock (D-051) --------------------------------------------------------------------------


def _lock_refiner(monkeypatch, **kw):
    """Refiner over two routes; _call is replaced: it records the route and fails for keys in `failing`."""
    with FakeLlmApi() as server:
        refiner = _refiner(_client(server, models=["good-model", "good-two"]), review=False, **kw)
    calls, failing = [], set()

    def fake_call(route, system, payload):
        calls.append(route.key)
        if route.key in failing:
            raise ProviderCallError("temporary", status=503)
        if "corrections" in system or kw.get("mode") == "correct":
            return {"data": {"corrections": []}, "model": route.key}
        return {"data": {"lines": [{"id": line["id"], "text": "t"} for line in payload["lines"]]},
                "model": route.key}

    monkeypatch.setattr(refiner, "_call", fake_call)
    return refiner, calls, failing


def test_style_lock_keeps_the_first_model_even_when_a_better_one_returns(monkeypatch):
    refiner, calls, failing = _lock_refiner(monkeypatch)
    best, second = refiner.pool.available()
    failing.add(best.key)
    refiner.translate_block(LINES, [], [], {})          # best fails -> second succeeds and is locked
    assert refiner._locked_route.key == second.key
    failing.clear()                                      # best is healthy again
    calls.clear()
    refiner.translate_block(LINES, [], [], {})
    assert calls == [second.key]                         # the lock wins over the better-ranked model


def test_style_lock_temporary_failure_uses_another_model_for_that_block_only(monkeypatch):
    refiner, calls, failing = _lock_refiner(monkeypatch)
    first, other = refiner.pool.available()
    refiner.translate_block(LINES, [], [], {})
    assert refiner._locked_route.key == first.key
    failing.add(first.key)
    calls.clear()
    result = refiner.translate_block(LINES, [], [], {})
    assert calls == [first.key, other.key] and result.translator == other.key
    assert refiner._locked_route.key == first.key        # lock kept
    failing.clear()
    calls.clear()
    refiner.translate_block(LINES, [], [], {})
    assert calls == [first.key]                          # back to the lock on the next block


def test_style_lock_moves_when_the_locked_model_leaves_the_pool(monkeypatch):
    refiner, calls, failing = _lock_refiner(monkeypatch)
    first, other = refiner.pool.available()
    refiner.translate_block(LINES, [], [], {})
    refiner.pool.drop(first)                             # e.g. removed after 3 failures
    calls.clear()
    refiner.translate_block(LINES, [], [], {})
    assert calls == [other.key] and refiner._locked_route.key == other.key


def test_style_lock_in_correct_mode(monkeypatch):
    refiner, calls, failing = _lock_refiner(monkeypatch, mode="correct")
    first, _ = refiner.pool.available()
    lines = [dict(line, draft="d") for line in LINES]
    refiner.translate_block(lines, [], [], {})
    refiner.translate_block(lines, [], [], {})
    assert refiner._locked_route.key == first.key and calls == [first.key, first.key]


def test_translation_requests_use_temperature_zero():
    with FakeLlmApi() as server:
        client = _client(server)
        seen = {}
        original = client.chat

        def spy(model, messages, **kw):
            seen.update(kw)
            return original(model, messages, **kw)

        client.chat = spy
        _refiner(client, review=False).translate_block(LINES, [], [], {})
    assert seen.get("temperature") == 0.0





def test_block_progress_moves_inside_a_block():
    with FakeLlmApi() as server:
        refiner = _refiner(_client(server))
        steps = []
        refiner.on_step = steps.append

        refiner.translate_block(LINES, [], [], {})

    assert steps[0] == 0.0 and steps[-1] == 1.0 and steps == sorted(steps) and len(steps) >= 4


def test_a_failing_progress_callback_never_breaks_the_translation():
    with FakeLlmApi() as server:
        refiner = _refiner(_client(server))

        def broken(fraction):
            raise RuntimeError("ui gone")

        refiner.on_step = broken

        assert refiner.translate_block(LINES, [], [], {}).translations
