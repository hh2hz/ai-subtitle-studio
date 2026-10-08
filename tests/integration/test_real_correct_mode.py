"""Live "correct" mode: the built-in local model translates, then the cloud AI returns only corrections.
Prints the local draft, the final line and the token usage reported by each provider.

    .venv\\Scripts\\python.exe -m pytest tests/integration/test_real_correct_mode.py -m integration -s
"""

import time

import pytest

from app.core.llm_providers import ProviderCallError, ProviderPool, build_clients
from app.core.llm_translation import LlmRefiner
from app.models import engines
from app.models.model_manager import DEFAULT_LOCAL_MODEL, ModelManager
from app.services.api_keys import load_keys
from app.services.hardware_detection import EnginePlan
from app.utils.paths import AppPaths, default_data_root
from tests.integration.test_real_local_llm import LINES

pytestmark = pytest.mark.integration


def test_local_translation_corrected_by_cloud_ai():
    clients = []
    for client in build_clients(load_keys()):
        try:
            if client.discover_models():
                clients.append(client)
        except ProviderCallError as exc:
            print(f"{client.name} unusable: {exc}")
    if not clients:
        pytest.skip("no usable API provider")

    paths = AppPaths.from_root(default_data_root()).ensure()
    backend = engines.load_translation_backend(
        EnginePlan(DEFAULT_LOCAL_MODEL, "local", "cuda", "test"), ModelManager(paths.models_dir), 4)
    try:
        started = time.monotonic()
        drafts = backend.translate(LINES, "tr", "ar")
        print(f"\nlocal model: {len(LINES)} lines in {time.monotonic() - started:.1f} s "
              f"on the {'GPU' if backend.client.gpu else 'CPU'}")
    finally:
        backend.close()

    refiner = LlmRefiner(ProviderPool(clients), "tr", "ar", {"title": "Kurtlar Vadisi"}, mode="correct")
    started = time.monotonic()
    result = refiner.translate_block([{"id": i, "source": s, "draft": d} for i, (s, d) in enumerate(zip(LINES, drafts))],
                                     [], [], {})
    print(f"cloud check by {result.translator} in {time.monotonic() - started:.1f} s; "
          f"{len(result.changed or ())} of {len(LINES)} lines corrected")
    for i, src in enumerate(LINES):
        mark = "CORRECTED" if i in (result.changed or ()) else "kept"
        print(f"{src}\n  local: {drafts[i]}\n  final: {result.translations.get(i, '')}  [{mark}]")
    for client in clients:
        if client.usage["requests"]:
            print(f"tokens {client.name}: {client.usage['prompt_tokens']} input + "
                  f"{client.usage['completion_tokens']} output in {client.usage['requests']} requests")
    assert result.translator, result.notes
