"""Loading LLM API keys from local JSON files (never from the database, never logged)."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from app.utils.paths import default_data_root

log = logging.getLogger(__name__)
# From source: api_keys.local.json in the project root. Installed app: the file in the data folder.
PROJECT_FILE = (default_data_root() / "api_keys.json" if getattr(sys, "frozen", False)
                else Path(__file__).resolve().parents[2] / "api_keys.local.json")


def is_placeholder(value: object) -> bool:
    """True for the text of api_keys.example.json ("enter your api key here"); such an entry is not a key."""
    return isinstance(value, str) and value.strip().lower().startswith("enter your")


def key_files() -> list[Path]:
    """Data-folder file first (wins), then the file next to the application."""
    return [default_data_root() / "api_keys.json", PROJECT_FILE]


def load_keys(paths: list[Path] | None = None) -> dict:
    keys: dict = {}
    for path in reversed(paths or key_files()):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            log.warning("Cannot read API keys file %s: %s", path, exc)
            continue
        for name, entry in data.items():
            if not name.startswith("_") and isinstance(entry, dict) and entry.get("api_key"):
                if is_placeholder(entry["api_key"]):
                    continue                      # the example file copied without filling this provider in
                if is_placeholder(entry.get("account_id")):
                    entry = {k: v for k, v in entry.items() if k != "account_id"}
                keys[name] = entry
    log.info("API keys available for: %s", ", ".join(sorted(keys)) or "none")
    return keys


# Where a user gets a free key. Providers that answer anonymous callers need none (KEYLESS below).
SIGNUP_URLS: dict[str, str] = {
    "gemini": "https://aistudio.google.com/apikey",
    "groq": "https://console.groq.com/keys",
    "nvidia": "https://build.nvidia.com/settings/api-keys",
    "cloudflare": "https://dash.cloudflare.com/profile/api-tokens",
    "zai": "https://z.ai/manage-apikey/apikey-list",
    "cohere": "https://dashboard.cohere.com/api-keys",
    "llm7": "https://docs.llm7.io/",
    "kilo": "https://kilo.ai/docs/getting-started/setup-authentication",
    "aionlabs": "https://api.aionlabs.ai/",
    "mistral": "https://console.mistral.ai/api-keys",
    "dashscope": "https://modelstudio.console.alibabacloud.com/",
    "ollama": "https://ollama.com/settings/keys",
    "openrouter": "https://openrouter.ai/keys",
    "llmtech": "https://llmtech.eu/",
    "requesty": "https://app.requesty.ai/api-keys",
    "aihubmix": "https://aihubmix.com/token",
    "huggingface": "https://huggingface.co/settings/tokens",
    "llmtr": "https://llmtr.com/",
    "modelscope": "https://modelscope.cn/my/myaccesstoken",
    "freeinference": "https://freeinference.org/",
    "moark": "https://moark.ai/docs/organization/access-token",
    "tokenharbor": "https://tokenharbor.ai/",
    "nous": "https://portal.nousresearch.com/",
    "routeway": "https://routeway.ai/",
    "vercel": "https://vercel.com/dashboard",
    "sealion": "https://docs.sea-lion.ai/",
    "siliconflow": "https://cloud.siliconflow.cn/account/ak",
    "tencent": "https://console.cloud.tencent.com/",
}
KEYLESS = "keyless"          # the value stored for providers that need no key (ovh, unturf)


def target_file() -> Path:
    """The key file the settings dialog writes: the first existing file in priority order, otherwise the data-folder
    file of an installed build or the project file when running from sources."""
    for path in key_files():
        if path.is_file():
            return path
    return PROJECT_FILE


def read_file(path: Path) -> dict:
    """The raw content of one key file ({} when it is missing or unreadable)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_keys(entries: dict[str, dict], path: Path | None = None) -> Path:
    """Write the user's keys. `entries` maps a provider name to {"api_key": ..., "account_id": ...}; providers with
    an empty key are skipped (removed from the file) but stay known to the program, so they can be added later.
    Everything else in an existing file (comments, unknown names) is kept. The write is atomic. Keys are never
    logged. Raises OSError when the file cannot be written."""
    path = path or target_file()
    data = read_file(path)
    for name in list(data):
        if not name.startswith("_") and name in entries:
            del data[name]
    for name, entry in entries.items():
        key = str(entry.get("api_key", "")).strip()
        if not key or is_placeholder(key):
            continue
        clean = {"api_key": key}
        account = str(entry.get("account_id", "") or "").strip()
        if account and not is_placeholder(account):
            clean["account_id"] = account
        data[name] = clean
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temp_name, path)
    except OSError:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    log.info("API keys saved for %d provider(s)", sum(1 for n in data if not n.startswith("_")))
    return path
