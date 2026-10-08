"""Live check of every configured free LLM provider (network; uses your api_keys.local.json).

    .venv\\Scripts\\python.exe -m pytest tests/integration/test_real_llm_api.py -m integration -s
Prints, per provider: chosen models, the translation of six self-written Turkish lines, and timing.
"""

import time

import pytest

from app.core.llm_providers import ProviderCallError, ProviderPool, build_clients
from app.core.llm_translation import LlmRefiner
from app.core.verification import check_line
from app.services.api_keys import load_keys

pytestmark = pytest.mark.integration

# Self-written lines (no third-party content): idiom, imperative, name, number, ellipsis, slang address.
LINES = [
    "Kamyon hazir mi abi?",
    "Bana bulamadim deme, Nevzat.",
    "Edebini tak, burasi anaokulu degil.",
    "Saat 5'te limanda bulusalim.",
    "Helikopter mi? Bu is cok riskli.",
    "Tamam abi, hallederim.",
]


def test_each_provider_translates():
    clients = build_clients(load_keys())
    if not clients:
        pytest.skip("no API keys configured")
    working = []
    for client in clients:
        print(f"\n=== {client.name}")
        try:
            print("models:", client.discover_models())
        except ProviderCallError as exc:
            print("UNUSABLE:", exc)
            continue
        refiner = LlmRefiner(ProviderPool([client]), "tr", "ar", {"title": "test"}, review=False)
        started = time.monotonic()
        result = refiner.translate_block([{"id": i, "source": s, "draft": ""} for i, s in enumerate(LINES)],
                                         [], [], {})
        elapsed = time.monotonic() - started
        if not result.translations:
            print("FAILED:", result.notes)
            continue
        working.append(client.name)
        print(f"translator: {result.translator}  ({elapsed:.1f} s)")
        for i, src in enumerate(LINES):
            tgt = result.translations.get(i, "")
            print(f"  {src}\n    -> {tgt}  {check_line(src, tgt, 'ar') or ''}")
    print("\nWORKING PROVIDERS:", working)
    assert working, "no provider produced a translation"
