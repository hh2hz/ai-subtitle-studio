"""The example key file copied without editing must not create providers with fake keys."""

import json

from app.core.llm_providers import build_clients
from app.services.api_keys import load_keys


def test_example_file_yields_no_clients(tmp_path):
    example = json.loads((__import__("pathlib").Path(__file__).resolve().parents[1] / "api_keys.example.json")
                         .read_text(encoding="utf-8"))
    path = tmp_path / "keys.json"
    path.write_text(json.dumps(example), encoding="utf-8")
    keys = load_keys([path])
    assert set(keys) <= {"ovh", "unturf"}                    # only the keyless providers remain
    assert all(c.spec.keyless for c in build_clients(keys))


def test_filled_entry_is_kept_and_placeholder_account_id_dropped(tmp_path):
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"cloudflare": {"api_key": "k", "account_id": "enter your account id here"},
                                "groq": {"api_key": "enter your api key here"}}), encoding="utf-8")
    keys = load_keys([path])
    assert list(keys) == ["cloudflare"] and "account_id" not in keys["cloudflare"]
