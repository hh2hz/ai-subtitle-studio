"""Cancelling a job must not wait for a slow cloud AI provider (D-115)."""

import threading
import time

import pytest

from app.core import brief as br
from app.core.errors import JobCancelled
from app.core.llm_providers import ChatClient, ModelRanking, ProviderPool, ProviderSpec
from app.core.llm_translation import LlmRefiner

SLOW_S = 6.0
UNITS = [{"id": i, "start": i * 2.0, "text": t} for i, t in enumerate(["Nevzat abi geldi mi?", "Gelmedi."])]


def _client(monkeypatch, cancel):
    """A client whose server never answers within the test (the real socket is replaced by a long sleep)."""
    client = ChatClient(ProviderSpec("slow", "http://127.0.0.1:9", prefer=("good",)), "k")
    client.models = ["good"]
    client.cancel = cancel

    def blocked(self, path, body=None, url=None, timeout_s=None):
        time.sleep(SLOW_S)
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(ChatClient, "_request_blocking", blocked)
    return client


def _cancel_soon(cancel, delay=0.3):
    threading.Timer(delay, cancel.set).start()


def test_a_blocked_request_returns_within_a_moment_of_the_cancel(monkeypatch):
    cancel = threading.Event()
    client = _client(monkeypatch, cancel)
    _cancel_soon(cancel)
    started = time.monotonic()

    with pytest.raises(JobCancelled):
        client.chat("good", [{"role": "user", "content": "hi"}])

    assert time.monotonic() - started < 2.0


def test_an_already_cancelled_job_sends_nothing(monkeypatch):
    cancel = threading.Event()
    cancel.set()
    client = _client(monkeypatch, cancel)
    called = []
    monkeypatch.setattr(ChatClient, "_request_blocking", lambda *a, **k: called.append(1))

    with pytest.raises(JobCancelled):
        client.chat("good", [{"role": "user", "content": "hi"}])

    assert not called


def test_without_a_cancel_event_the_request_runs_in_the_calling_thread(monkeypatch):
    client = ChatClient(ProviderSpec("plain", "http://127.0.0.1:9", prefer=("m",)), "k")
    seen = []
    monkeypatch.setattr(ChatClient, "_request_blocking",
                        lambda self, *a, **k: seen.append(threading.current_thread()) or
                        {"choices": [{"message": {"content": "ok"}}]})

    client.chat("m", [{"role": "user", "content": "hi"}])

    assert seen == [threading.current_thread()]


def test_errors_of_the_request_still_reach_the_caller(monkeypatch):
    from app.core.llm_providers import ProviderCallError

    client = ChatClient(ProviderSpec("plain", "http://127.0.0.1:9", prefer=("m",)), "k")
    client.cancel = threading.Event()

    def fail(self, *a, **k):
        raise ProviderCallError("plain HTTP 500: boom", 500)

    monkeypatch.setattr(ChatClient, "_request_blocking", fail)

    with pytest.raises(ProviderCallError):
        client.chat("m", [{"role": "user", "content": "hi"}])


def test_the_refiner_stops_a_translation_call_and_does_not_punish_the_model(monkeypatch):
    cancel = threading.Event()
    client = _client(monkeypatch, None)
    pool = ProviderPool([client], ModelRanking(floor=0))
    refiner = LlmRefiner(pool, "tr", "ar", {})
    refiner.set_cancel(cancel)
    route = pool.available()[0]
    _cancel_soon(cancel)
    started = time.monotonic()

    with pytest.raises(JobCancelled):
        refiner._call(route, "system", {"lines": []})

    assert time.monotonic() - started < 2.0
    assert pool.failures == {} and "good" in client.models        # cancelling is not a model failure


def test_the_refiner_wait_between_retries_ends_at_once(monkeypatch):
    cancel = threading.Event()
    refiner = LlmRefiner(ProviderPool([]), "tr", "ar", {})
    refiner.set_cancel(cancel)
    _cancel_soon(cancel)
    started = time.monotonic()

    with pytest.raises(JobCancelled):
        refiner._sleep(30)

    assert time.monotonic() - started < 2.0


def test_the_episode_brief_is_not_swallowed_when_cancelled(monkeypatch):
    cancel = threading.Event()
    client = _client(monkeypatch, cancel)
    pool = ProviderPool([client], ModelRanking(floor=0))
    _cancel_soon(cancel)
    started = time.monotonic()

    with pytest.raises(JobCancelled):
        br.build_brief(pool, "tr", "ar", UNITS, {})

    assert time.monotonic() - started < 2.0
