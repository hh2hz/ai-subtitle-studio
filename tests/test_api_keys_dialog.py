"""The API keys window, the key file writer and the first-start prompt (D-117)."""

import json
import re
from pathlib import Path

import pytest

from app.core.llm_providers import DEFAULT_ORDER, PROVIDERS, build_clients
from app.services import api_keys
from app.ui import api_keys_dialog as akd
from app.ui.api_keys_dialog import ApiKeysDialog, maybe_prompt_first_run, provider_names

ROOT = Path(__file__).resolve().parents[1]
KEY_SECRET = "sk-test-SECRET-1234567890"


@pytest.fixture
def key_file(tmp_path, monkeypatch):
    """Point every key-file lookup at a temporary folder."""
    target = tmp_path / "api_keys.json"
    monkeypatch.setattr(api_keys, "PROJECT_FILE", target)
    monkeypatch.setattr(api_keys, "key_files", lambda: [target])
    return target


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# -- save_keys -------------------------------------------------------------------------------------------------

def test_save_round_trips_through_load_keys(key_file):
    api_keys.save_keys({"gemini": {"api_key": KEY_SECRET}, "cloudflare": {"api_key": "cf", "account_id": "acc"}},
                       key_file)

    loaded = api_keys.load_keys([key_file])

    assert loaded["gemini"]["api_key"] == KEY_SECRET
    assert loaded["cloudflare"] == {"api_key": "cf", "account_id": "acc"}


def test_blank_and_placeholder_entries_are_not_written(key_file):
    api_keys.save_keys({"gemini": {"api_key": "  "}, "groq": {"api_key": "enter your api key here"},
                        "nvidia": {"api_key": "real"}}, key_file)

    assert set(_read(key_file)) == {"nvidia"}


def test_a_provider_cleared_in_the_window_is_removed_from_the_file(key_file):
    api_keys.save_keys({"gemini": {"api_key": "a"}, "groq": {"api_key": "b"}}, key_file)

    api_keys.save_keys({"gemini": {"api_key": ""}, "groq": {"api_key": "b"}}, key_file)

    assert set(_read(key_file)) == {"groq"}


def test_comments_and_unknown_entries_survive_a_save(key_file):
    key_file.write_text(json.dumps({"_comment": "keep me", "future_provider": {"api_key": "x"},
                                    "gemini": {"api_key": "old", "_where": "url"}}), encoding="utf-8")

    api_keys.save_keys({"gemini": {"api_key": "new"}}, key_file)

    data = _read(key_file)
    assert data["_comment"] == "keep me"
    assert data["future_provider"] == {"api_key": "x"}
    assert data["gemini"] == {"api_key": "new"}


def test_save_leaves_no_temporary_file_and_never_logs_a_key(key_file, caplog):
    caplog.set_level("DEBUG")

    api_keys.save_keys({"gemini": {"api_key": KEY_SECRET}}, key_file)

    assert [p.name for p in key_file.parent.iterdir()] == [key_file.name]
    assert KEY_SECRET not in caplog.text


def test_a_failed_write_raises_oserror_and_keeps_the_old_file(key_file, monkeypatch):
    api_keys.save_keys({"gemini": {"api_key": "old"}}, key_file)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(api_keys.os, "replace", broken)
    with pytest.raises(OSError):
        api_keys.save_keys({"gemini": {"api_key": "new"}}, key_file)

    assert _read(key_file)["gemini"]["api_key"] == "old"
    assert [p.name for p in key_file.parent.iterdir()] == [key_file.name]


def test_a_skipped_provider_is_not_used_but_stays_known(key_file):
    api_keys.save_keys({"gemini": {"api_key": "a"}, "groq": {"api_key": ""}}, key_file)

    clients = build_clients(api_keys.load_keys([key_file]))

    assert [c.spec.name for c in clients] == ["gemini"]
    assert "groq" in PROVIDERS and "groq" in DEFAULT_ORDER


# -- the window ------------------------------------------------------------------------------------------------

def test_the_window_has_a_row_for_every_provider(qapp, translator, key_file):
    dialog = ApiKeysDialog(translator, path=key_file)

    keyed = {n for n in PROVIDERS if not PROVIDERS[n].keyless}
    assert set(dialog._fields) == keyed
    assert set(dialog._keyless) == set(PROVIDERS) - keyed
    assert provider_names() and set(provider_names()) == set(PROVIDERS)


def test_the_window_shows_saved_keys_hidden_until_asked(qapp, translator, key_file):
    api_keys.save_keys({"gemini": {"api_key": KEY_SECRET}}, key_file)
    dialog = ApiKeysDialog(translator, path=key_file)

    edit = dialog._fields["gemini"]
    assert edit.text() == KEY_SECRET
    assert edit.echoMode().name == "Password"
    dialog.show_check.setChecked(True)
    assert edit.echoMode().name == "Normal"


def test_save_writes_only_the_filled_providers(qapp, translator, key_file):
    dialog = ApiKeysDialog(translator, path=key_file)
    dialog._fields["gemini"].setText("  g-key ")

    dialog.save_button.click()

    assert dialog.saved
    data = _read(key_file)
    assert data["gemini"] == {"api_key": "g-key"}
    assert "groq" not in data


def test_a_cloudflare_key_without_an_account_id_is_refused(qapp, translator, key_file, monkeypatch):
    shown = []
    monkeypatch.setattr(akd.QMessageBox, "warning", lambda *a, **k: shown.append(a))
    dialog = ApiKeysDialog(translator, path=key_file)
    dialog._fields["cloudflare"].setText("cf")

    dialog.save_button.click()

    assert shown and not dialog.saved and not key_file.exists()


def test_skip_writes_nothing(qapp, translator, key_file):
    dialog = ApiKeysDialog(translator, path=key_file, first_run=True)
    dialog._fields["gemini"].setText("typed but skipped")

    dialog.skip_button.click()

    assert not dialog.saved and not key_file.exists()


def test_keyless_providers_follow_their_checkbox(qapp, translator, key_file):
    dialog = ApiKeysDialog(translator, path=key_file)
    assert dialog._keyless["ovh"].isChecked()                    # on by default
    dialog._keyless["ovh"].setChecked(False)
    dialog._fields["gemini"].setText("g")

    dialog.save_button.click()

    data = _read(key_file)
    assert "ovh" not in data and data["gemini"]["api_key"] == "g"


def test_a_write_error_is_shown_and_the_window_stays_open(qapp, translator, key_file, monkeypatch):
    shown = []
    monkeypatch.setattr(akd.QMessageBox, "critical", lambda *a, **k: shown.append(a))
    monkeypatch.setattr(akd.api_keys, "save_keys", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    dialog = ApiKeysDialog(translator, path=key_file)
    dialog._fields["gemini"].setText("g")

    dialog.save_button.click()

    assert shown and not dialog.saved


# -- first start ----------------------------------------------------------------------------------------------

def test_first_start_without_keys_asks_once(settings, translator, key_file):
    calls = []

    assert maybe_prompt_first_run(translator, settings, ask=lambda: calls.append(1)) is True
    assert maybe_prompt_first_run(translator, settings, ask=lambda: calls.append(1)) is False

    assert calls == [1] and settings.get("api_keys_prompted") is True


def test_existing_keys_mean_no_window_but_the_question_is_closed(settings, translator, key_file):
    api_keys.save_keys({"gemini": {"api_key": "g"}}, key_file)
    calls = []

    assert maybe_prompt_first_run(translator, settings, ask=lambda: calls.append(1)) is False

    assert not calls and settings.get("api_keys_prompted") is True


def test_skipping_the_first_window_is_remembered(settings, translator, key_file):
    maybe_prompt_first_run(translator, settings, ask=lambda: None)         # the user pressed Skip

    assert settings.get("api_keys_prompted") is True
    assert maybe_prompt_first_run(translator, settings, ask=lambda: pytest.fail("asked again")) is False


# -- wiring ---------------------------------------------------------------------------------------------------

def test_settings_has_the_api_keys_button(qapp, translator, settings, monkeypatch):
    from app.ui.settings_window import ProviderSettingsDialog

    opened = []
    monkeypatch.setattr("app.ui.settings_window.ApiKeysDialog",
                        lambda *a, **k: type("D", (), {"saved": False, "exec": lambda self: opened.append(1)})())
    window = ProviderSettingsDialog(translator, settings)

    window.api_keys_button.click()

    assert opened == [1]


def test_the_texts_exist_in_both_languages():
    for language in ("en", "ar"):
        strings = json.loads((ROOT / "app/resources/translations" / f"{language}.json")
                             .read_text(encoding="utf-8-sig"))["strings"]
        for key in ("title", "intro_first", "intro", "placeholder", "account_id", "get_key", "keyless", "show",
                    "note", "save", "skip", "cancel", "open", "open_tip", "need_account", "save_failed"):
            assert strings.get(f"keys.{key}"), f"{language}: keys.{key}"


def test_signup_links_belong_to_known_providers_and_are_https():
    assert set(api_keys.SIGNUP_URLS) <= set(PROVIDERS)
    assert all(url.startswith("https://") for url in api_keys.SIGNUP_URLS.values())
    keyed = {n for n, spec in PROVIDERS.items() if not spec.keyless}
    assert keyed - set(api_keys.SIGNUP_URLS) == set()            # every provider that needs a key has a link


def test_the_example_file_lists_the_same_providers_as_the_program():
    example = ROOT / "api_keys.example.json"
    if not example.is_file():
        pytest.skip("source tree without the example file")
    names = {n for n in json.loads(example.read_text(encoding="utf-8")) if not n.startswith("_")}
    assert names <= set(PROVIDERS)


def test_the_dialog_never_writes_keys_to_the_log():
    source = (ROOT / "app/ui/api_keys_dialog.py").read_text(encoding="utf-8")
    assert not re.search(r"log\.\w+\([^)]*(api_key|entries|text\(\))", source)
