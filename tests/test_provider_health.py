"""Faster refinement when models are dead or out of quota (D-086): no network, fake clients."""

import json
import time

import pytest

from app.core import llm_providers as lp
from app.core.llm_providers import HealthStore, ProviderCallError, ProviderPool


def test_timeout_and_monthly_quota_are_recognised():
    timeout = ProviderCallError("nvidia request failed: The read operation timed out")
    assert timeout.timed_out and not timeout.rate_limited
    assert not ProviderCallError("gemini HTTP 503: high demand", 503).timed_out
    cohere = ProviderCallError("cohere HTTP 429: You are using a Trial key, which is limited to 1000 API calls / month.", 429)
    assert cohere.monthly_quota and cohere.daily_quota
    assert ProviderCallError("openrouter HTTP 429: Rate limit exceeded: free-models-per-day.", 429).daily_quota
    assert not ProviderCallError("groq HTTP 429: tokens per minute", 429).monthly_quota


def test_default_read_timeout_is_short():
    assert lp.REQUEST_TIMEOUT_S <= 60


def test_health_store_roundtrip_and_expiry(tmp_path):
    now = [1000.0]
    path = tmp_path / "sub" / "health.json"
    store = HealthStore(path, clock=lambda: now[0])
    store.record("nvidia:x", 600, "timeout")
    assert 590 < store.active()["nvidia:x"] <= 600
    assert HealthStore(path, clock=lambda: now[0]).active().keys() == {"nvidia:x"}   # survives a restart
    now[0] += 601
    assert HealthStore(path, clock=lambda: now[0]).active() == {}                    # expired


def test_broken_health_file_means_no_memory(tmp_path):
    path = tmp_path / "health.json"
    path.write_text("{not json", encoding="utf-8")
    assert HealthStore(path).active() == {}
    path.write_text(json.dumps({"a": "text", "b": {"until": time.time() + 99}}), encoding="utf-8")
    assert set(HealthStore(path).active()) == {"b"}


class _Client:
    def __init__(self, name, models):
        self.name = name
        self.models = list(models)
        self.spec = lp.PROVIDERS.get(name) or next(iter(lp.PROVIDERS.values()))


def _pool(tmp_path):
    clients = [_Client("gemini", ["g1"]), _Client("nvidia", ["n1", "n2"])]
    return ProviderPool(clients, health=HealthStore(tmp_path / "h.json"))


def test_timeout_removes_the_model_at_once_and_the_next_job_skips_it(tmp_path):
    pool = _pool(tmp_path)
    route = next(r for r in pool.routes() if r.key == "nvidia:n1")
    pool.timeout(route, "read timeout")
    assert "nvidia:n1" not in {r.key for r in pool.routes()}

    later = _pool(tmp_path)                                  # a new job: the models are listed again
    assert "nvidia:n1" in {r.key for r in later.routes()}
    later.apply_health()
    assert "nvidia:n1" not in {r.key for r in later.available()}
    assert "nvidia:n2" in {r.key for r in later.available()}


def test_provider_wide_memory_skips_every_model_of_the_provider(tmp_path):
    pool = _pool(tmp_path)
    pool.remember("nvidia", 3600, "monthly quota")
    later = _pool(tmp_path)
    later.apply_health()
    assert {r.provider for r in later.available()} == {"gemini"}


def test_three_strikes_are_remembered(tmp_path):
    pool = _pool(tmp_path)
    route = next(r for r in pool.routes() if r.key == "gemini:g1")
    for _ in range(3):
        pool.strike(route, "HTTP 503")
    later = _pool(tmp_path)
    assert "gemini:g1" in later.apply_health()


# -- through the refiner (same fixtures as tests/test_llm_refine.py) ---------------------------------

from app.core.llm_providers import ModelRanking  # noqa: E402
from tests.fake_llm_api import FakeLlmApi  # noqa: E402
from tests.test_llm_refine import LINES, _client, _refiner  # noqa: E402


def test_a_timed_out_model_is_dropped_after_one_failure_and_remembered(tmp_path):
    with FakeLlmApi() as server:
        client = _client(server, "g", ["slow", "good"])
        original = client.chat
        calls = []

        def chat(model, messages, **kw):
            calls.append(model)
            if model == "slow":
                raise ProviderCallError("g request failed: The read operation timed out")
            return original(model, messages, **kw)

        client.chat = chat
        refiner = _refiner(client, review=False, ranking=ModelRanking(scores={"g:slow": 95, "g:good": 90}, floor=0))
        refiner.pool.health = HealthStore(tmp_path / "h.json")
        assert len(refiner.translate_block(LINES, [], [], {}).translations) == 2
        assert calls.count("slow") == 2 and client.models == ["good"]   # base try + one long try, then dropped
        assert "g:slow" in HealthStore(tmp_path / "h.json").active()


def test_monthly_quota_skips_the_whole_provider(tmp_path):
    with FakeLlmApi() as server:
        cohere = _client(server, "c", ["big", "small"])
        other = _client(server, "o", ["fine"])
        original = other.chat
        cohere.chat = lambda *a, **kw: (_ for _ in ()).throw(ProviderCallError(
            "c HTTP 429: You are using a Trial key, which is limited to 1000 API calls / month.", 429))
        scores = {"c:big": 99, "c:small": 98, "o:fine": 90}
        refiner = _refiner(cohere, other, review=False, ranking=ModelRanking(scores=scores, floor=0))
        refiner.pool.health = HealthStore(tmp_path / "h.json")
        result = refiner.translate_block(LINES, [], [], {})
        assert result.translator == "o:fine"
        assert {r.provider for r in refiner.pool.available()} == {"o"}      # c:small is skipped without a request
        assert "c" in HealthStore(tmp_path / "h.json").active()
        del original


# -- adaptive timeout (D-087) ------------------------------------------------------------------------

def test_timeout_grows_with_the_models_own_reply_time():
    with FakeLlmApi() as server:
        client = _client(server, "g", ["m"])
        assert client.timeout_for("m") == client.timeout_s
        client.latency["m"] = [30.0, 40.0, 50.0]
        assert client.timeout_for("m") == 120.0                       # 3 x median 40, within the cap
        client.latency["m"] = [10.0, 12.0]
        assert client.timeout_for("m") == client.timeout_s            # fast model keeps the base value
        client.latency["m"] = [100.0, 100.0]
        assert client.timeout_for("m") == lp.LONG_TIMEOUT_S           # capped


def test_a_measured_good_model_gets_one_longer_try_before_it_is_dropped(tmp_path):
    with FakeLlmApi() as server:
        client = _client(server, "g", ["slow", "other"])
        original = client.chat
        seen = []

        def chat(model, messages, **kw):
            seen.append((model, kw.get("timeout_s")))
            if model == "slow" and kw.get("timeout_s", 0) < lp.LONG_TIMEOUT_S:
                raise ProviderCallError("g request failed: The read operation timed out")
            return original(model, messages, **{k: v for k, v in kw.items() if k != "timeout_s"})

        client.chat = chat
        ranking = ModelRanking(scores={"g:slow": 95, "g:other": 90}, floor=78)
        refiner = _refiner(client, review=False, ranking=ranking)
        result = refiner.translate_block(LINES, [], [], {})
        assert result.translator == "g:slow"                        # the long retry succeeded
        assert [t for m, t in seen if m == "slow"][-1] == lp.LONG_TIMEOUT_S
        assert client.models == ["slow", "other"]                   # not dropped


def test_an_unmeasured_model_is_dropped_without_a_long_retry(tmp_path):
    with FakeLlmApi() as server:
        client = _client(server, "g", ["unknown", "good"])
        original = client.chat
        calls = []

        def chat(model, messages, **kw):
            calls.append(model)
            if model == "unknown":
                raise ProviderCallError("g request failed: The read operation timed out")
            return original(model, messages, **{k: v for k, v in kw.items() if k != "timeout_s"})

        client.chat = chat
        refiner = _refiner(client, review=False, ranking=ModelRanking(scores={"g:good": 1}, floor=0))
        refiner.translate_block(LINES, [], [], {})
        assert calls.count("unknown") == 1 and "unknown" not in client.models
