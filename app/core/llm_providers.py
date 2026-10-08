"""OpenAI-compatible chat clients for free LLM API providers.

Model names on these services change often, so models are discovered at runtime (GET /models). Every
text model of every provider is used (DECISIONS D-034): all of them form one list ranked by quality (measured
scores from tools/benchmark_models.py, otherwise a name heuristic), and a model that hits its rate limit or
daily quota is skipped in favour of the next one. Fixed fallback names are used only when discovery is not
available. Only free offerings are targeted (see DECISIONS D-024). Implemented from public documentation; the
services are not reachable from the development environment, so live behaviour is untested here.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from app import __version__
from app.core.errors import JobCancelled

log = logging.getLogger(__name__)
USER_AGENT = f"AISubtitleStudio/{__version__}"


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str                         # may contain {account_id}
    prefer: tuple[str, ...]               # substrings of the provider's stronger model families, best first
    exclude: tuple[str, ...] = ()         # provider-specific exclusions on top of NON_TEXT
    fallback_models: tuple[str, ...] = ()
    discover: bool = True
    require: str | None = None            # substring marking free model ids (e.g. ":free"); others must be priced 0
    free_only: bool = False               # use only models the provider itself prices at zero (no price -> not used)
    keyless: bool = False                 # provider answers without an API key (no Authorization header)
    base_score: float = 50.0              # heuristic quality of the provider's models before any benchmark
    models_url: str | None = None         # model list outside the OpenAI-compatible base (Cloudflare)


# Models that cannot do plain text chat (media generation, speech, embeddings, realtime, agents, safety filters).
NON_TEXT = ("image", "tts", "embed", "live", "audio", "native", "veo", "imagen", "lyria", "transcribe", "whisper",
            "guard", "rerank", "moderation", "computer-use", "robotics", "aqa", "deep-research", "antigravity",
            "speech", "ocr", "vision", "-vl", "learnlm", "retriever", "reward", "parse", "safety", "compound",
            "playai", "prompt", "coder", "math", "base", "banana", "video", "clip", "detector", "kosmos", "deplot",
            "fuyu", "neva", "vila")
MAX_MODELS_PER_PROVIDER: int | None = None      # no cap: every usable model is a candidate (D-035)
# Sentinel keys for providers that answer anonymous callers with no Authorization header.
KEYLESS_KEYS = frozenset({"keyless", "none", "-", "undefined", ""})

PROVIDERS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        "gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
        prefer=("flash", "pro", "gemma"), fallback_models=("gemini-2.5-flash",), base_score=72),
    "groq": ProviderSpec(
        "groq", "https://api.groq.com/openai/v1",
        prefer=("kimi-k2", "gpt-oss-120b", "llama-3.3-70b"),
        fallback_models=("openai/gpt-oss-120b", "llama-3.3-70b-versatile"), base_score=58),
    "nvidia": ProviderSpec(
        "nvidia", "https://integrate.api.nvidia.com/v1",
        prefer=("deepseek-v4", "deepseek-v3", "kimi-k2", "gpt-oss-120b", "llama-3.3-70b", "llama-4"),
        fallback_models=("meta/llama-3.3-70b-instruct",), base_score=55),
    "cloudflare": ProviderSpec(
        "cloudflare", "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        prefer=("gpt-oss-120b", "llama-4", "llama-3.3-70b", "mistral-small", "gemma"),
        models_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search"
                   "?task=Text%20Generation&per_page=200",
        fallback_models=("@cf/openai/gpt-oss-120b", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"), base_score=52),
    "zai": ProviderSpec(
        "zai", "https://api.z.ai/api/paas/v4",
        prefer=("flash",), require="flash", fallback_models=("glm-4.7-flash", "glm-4.5-flash"), base_score=52),
    "cohere": ProviderSpec(
        "cohere", "https://api.cohere.ai/compatibility/v1",
        prefer=("command-a-plus", "command-a-translate", "command-a"),
        exclude=("reasoning", "translate-lite"), fallback_models=("command-a-03-2025",), base_score=62),
    "llm7": ProviderSpec(
        "llm7", "https://api.llm7.io/v1",
        prefer=("gpt-oss", "deepseek", "gemini", "mistral", "llama"), base_score=45),
    "kilo": ProviderSpec(
        "kilo", "https://api.kilo.ai/api/gateway",
        prefer=("deepseek", "kimi", "glm", "gpt-oss", "llama", "gemini"), require=":free", base_score=52),
    "aionlabs": ProviderSpec(
        "aionlabs", "https://api.aionlabs.ai/v1",
        prefer=("aion-3.5", "aion-3.0", "aion-2.0"),
        fallback_models=("aion-labs/aion-3.5", "aion-labs/aion-3.0", "aion-labs/aion-2.0"), base_score=50),
    # ---- added 2026-10-06: every provider wired into the local gateway (config.json) ----
    "mistral": ProviderSpec(
        "mistral", "https://api.mistral.ai/v1",
        prefer=("large", "medium", "magistral", "codestral", "small"),
        fallback_models=("mistral-large-latest", "mistral-medium-latest", "magistral-medium-latest",
                         "mistral-small-latest", "codestral-latest"), base_score=60),
    "dashscope": ProviderSpec(
        "dashscope", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        prefer=("max", "pro", "flash", "glm", "deepseek"),
        fallback_models=("qwen3.8-max", "qwen3.8-27b", "qwen3.8-flash", "deepseek-v4-pro", "glm-5.3",
                         "deepseek-v4-flash", "glm-5.2"), discover=False, base_score=62),
    "ollama": ProviderSpec(
        "ollama", "https://ollama.com/v1",
        prefer=("ultra", "super", "gpt-oss:120b", "mistral-large", "glm", "gemma4"),
        fallback_models=("nemotron-3-ultra", "gpt-oss:120b", "nemotron-3-super", "mistral-large-3:675b",
                         "gemma4:31b", "gpt-oss:20b", "nemotron-3-nano:30b"), discover=False, base_score=58),
    "ovh": ProviderSpec(
        "ovh", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        prefer=("397b", "llama-3_3-70b", "qwen3-coder-30b", "gpt-oss-120b", "qwen3.8-27b", "mistral-small-3.2"),
        fallback_models=("Qwen3.5-397B-A17B", "Meta-Llama-3_3-70B-Instruct", "gpt-oss-120b",
                         "Qwen3-Coder-30B-A3B-Instruct", "Qwen3.8-27B", "Mistral-Small-3.2-24B-Instruct-2506",
                         "Qwen3.6-27B", "gpt-oss-20b"), keyless=True, base_score=58),
    "openrouter": ProviderSpec(
        "openrouter", "https://openrouter.ai/api/v1",
        prefer=("ultra", "super", "nemotron", "glm", "kimi", "deepseek", "gpt-oss", "qwen", "gemma"),
        require=":free", base_score=55),
    "llmtech": ProviderSpec(
        "llmtech", "https://api.llmtech.eu/v1",
        prefer=("qwen3.8",), fallback_models=("nvidia/Qwen3.8-27B-NVFP4",), discover=False, base_score=55),
    "requesty": ProviderSpec(
        "requesty", "https://router.requesty.ai/v1",
        prefer=("ultra", "super", "nemotron", "gpt-oss", "glm", "gemma", "qwen", "muse"),
        free_only=True, base_score=55),
    "aihubmix": ProviderSpec(
        "aihubmix", "https://aihubmix.com/v1",
        prefer=("glm", "kimi", "mimo", "intern", "agents", "dots"),
        require="-free", base_score=55),
    "huggingface": ProviderSpec(
        "huggingface", "https://router.huggingface.co/v1",
        prefer=("glm", "deepseek", "qwen", "llama-3.3", "gpt-oss", "minimax", "gemma"),
        fallback_models=("zai-org/GLM-5.3", "deepseek-ai/DeepSeek-V4.1-Flash", "Qwen/Qwen3.8-27B",
                         "meta-llama/Llama-3.3-70B-Instruct", "openai/gpt-oss-120b",
                         "MiniMaxAI/MiniMax-M3", "google/gemma-4-31B-it"), discover=False, base_score=55),
    "llmtr": ProviderSpec(
        "llmtr", "https://llmtr.com/v1",
        prefer=("ultra", "super", "nemotron", "ling", "agnes", "gemma"),
        fallback_models=("nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-super-120b-a12b",
                         "qwen/qwen3.8-27b-free", "agnes-3.0-flash", "ling-3.1-flash",
                         "inclusionai/ling-3.1-flash", "poolside/laguna-xs-2.1", "agnes-2.5-flash"),
        discover=False, base_score=55),
    "modelscope": ProviderSpec(
        "modelscope", "https://api-inference.modelscope.cn/v1",
        prefer=("glm", "deepseek-v4-pro", "qwen3.5-397b", "qwen3.8", "minimax", "step"),
        fallback_models=("deepseek-ai/DeepSeek-V4.1-Flash", "Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3.8-27B",
                         "ZhipuAI/GLM-5.2", "ZhipuAI/GLM-4.7-Flash", "MiniMax/MiniMax-M3",
                         "deepseek-ai/DeepSeek-V4-Flash-0731", "stepfun-ai/Step-3.7-Flash"),
        discover=False, base_score=55),
    "freeinference": ProviderSpec(
        "freeinference", "https://freeinference.org/v1",
        prefer=("glm", "minimax", "qwen", "deepseek"),
        fallback_models=("glm-5.3-flash", "minimax-m3", "qwen3.6-35b", "deepseek-v4-flash"),
        discover=False, base_score=54),
    "moark": ProviderSpec(
        "moark", "https://api.moark.com/v1",
        prefer=("glm-5.3", "glm-5.2", "kimi", "deepseek", "qwen3.8-27b", "minimax", "step"),
        fallback_models=("GLM-5.3", "kimi-k3", "GLM-5.2", "deepseek-v4-flash-0731", "qwen3.8-27b",
                         "MiniMax-M3", "Step-3.7-Flash", "qwen3.8-flash"), discover=False, base_score=54),
    "tokenharbor": ProviderSpec(
        "tokenharbor", "https://tokenharbor.ai/v1",
        prefer=("deepseek",), require=":free", base_score=54),
    "nous": ProviderSpec(
        "nous", "https://inference-api.nousresearch.com/v1",
        prefer=("step", "laguna", "ling", "longcat"), require=":free", base_score=52),
    "routeway": ProviderSpec(
        "routeway", "https://api.routeway.ai/v1",
        prefer=("deepseek", "minimax", "muse", "gemma"), require=":free", base_score=52),
    "vercel": ProviderSpec(
        "vercel", "https://ai-gateway.vercel.sh/v1",
        prefer=("ultra", "super", "nemotron", "laguna", "glm"), free_only=True, base_score=55),
    "sealion": ProviderSpec(
        "sealion", "https://api.sea-lion.ai/v1",
        prefer=("nemotron", "llama", "qwen"),
        fallback_models=("aisingapore/Qwen-SEA-LION-v4-32B-IT", "aisingapore/Llama-SEA-LION-v3-70B-IT",
                         "aisingapore/Gemma-SEA-LION-v4-27B-IT"),
        discover=False, base_score=53),
    "siliconflow": ProviderSpec(
        "siliconflow", "https://api.siliconflow.cn/v1",
        prefer=("glm", "deepseek", "qwen3.5", "qwen2.5"),
        fallback_models=("glm-z1-9b-0414", "deepseek-r1-0528-qwen3-8b", "qwen3.5-4b", "qwen2.5-7b"),
        discover=False, base_score=52),
    "tencent": ProviderSpec(
        "tencent", "https://tokenhub.tencentmaas.com/v1",
        prefer=("hy3",), fallback_models=("hy3",), discover=False, base_score=52),
    "unturf": ProviderSpec(
        "unturf", "https://hermes.ai.unturf.com/v1",
        prefer=("qwen3.8",), keyless=True, base_score=54),
}
# Provider order only breaks ties between equal scores (live comparison on the user's keys, D-025).
# The providers added 2026-10-06 follow the same rule; a measured score always outranks this order.
DEFAULT_ORDER = ("gemini", "cohere", "groq", "cloudflare", "kilo", "nvidia", "zai", "llm7", "aionlabs",
                 "mistral", "dashscope", "ollama", "openrouter", "ovh", "llmtech", "requesty", "aihubmix",
                 "huggingface", "llmtr", "modelscope", "freeinference", "moark", "vercel", "nous", "routeway",
                 "tokenharbor", "sealion", "siliconflow", "tencent", "unturf")
# Not "billing": Gemini's ordinary per-minute 429 says "check your plan and billing details", and treating it as
# fatal disabled every Gemini model for the rest of the job (found in the tools/list_models.py run, D-036).
_NO_QUOTA = ("insufficient balance", "insufficient_quota", "recharge", "payment required",
             "quota exceeded for this account")
_TOKEN_LIMIT = ("max_tokens", "too_many_tokens", "too many tokens", "maximum context", "max_new_tokens",
                "estimated number of input and maximum output tokens")
_DAILY_QUOTA = ("perday", "per day", "per-day", "daily", "rpd")
# A quota that resets monthly (Cohere trial keys: "limited to 1000 API calls / month"): hopeless for this job and
# for the rest of the day, and it applies to the whole key, not to one model (D-086).
_MONTHLY_QUOTA = ("/ month", "per month", "/month", "monthly", "per-month")
# How long a later job remembers a failure (D-086). Timeouts and repeated failures are usually a bad hour, daily
# quotas reset at an unknown time, monthly quotas do not reset for weeks.
MEMORY_FAILURE_S = 30 * 60.0
MEMORY_DAILY_S = 3 * 3600.0
MEMORY_MONTHLY_S = 24 * 3600.0
# One reply of a subtitle block is short; a model that is silent for this long is dead or overloaded (D-086).
REQUEST_TIMEOUT_S = 60.0
# A measured model above the quality floor gets one longer second chance before it is dropped, and a model that has
# answered before is allowed three times its median reply time (never more than this, D-087).
LONG_TIMEOUT_S = 150.0
# The request itself exceeds the account's per-minute token limit (Groq 413/429 "Request too large"): waiting does
# not help, a smaller request might.
_TOO_LARGE = ("request too large", "request_too_large", "tokens per minute (tpm): limit")
_RETRY_DELAY = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
# Unmeasured models rank just below the quality floor: used only after every measured model above it (D-047).
UNMEASURED_MARGIN = 0.5
_SMALL = re.compile(r"(?<![\d.])(0\.\d+|1|2|3|4|7|8|9|12|14)b\b|(?<![a-z])(mini|nano|tiny|small|instant)(?![a-z])")
_LARGE = re.compile(r"(?<![\d.])(70|72|120|235|405|480|671)b\b|kimi|deepseek|command-a")


class ProviderCallError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    @property
    def too_large(self) -> bool:
        return self.status in (413, 429) and any(marker in str(self).lower() for marker in _TOO_LARGE)

    @property
    def rate_limited(self) -> bool:
        return self.status == 429 and not self.fatal and not self.not_free and not self.too_large

    @property
    def not_free(self) -> bool:
        """The model has no free quota for this key (Gemini answers 429 with "limit: 0")."""
        return self.status == 429 and "limit: 0" in str(self).lower()

    @property
    def daily_quota(self) -> bool:
        """Rate limit that will not clear within this job (e.g. Gemini's requests-per-day free-tier quota)."""
        text = str(self).lower()
        compact = text.replace("_", "")
        return self.rate_limited and (self.monthly_quota or any(marker in compact for marker in _DAILY_QUOTA))

    @property
    def monthly_quota(self) -> bool:
        """The rate limit is a monthly allowance that is used up (it applies to the whole key)."""
        return self.rate_limited and any(marker in str(self).lower() for marker in _MONTHLY_QUOTA)

    @property
    def timed_out(self) -> bool:
        """No answer within the read timeout (a dead or overloaded model, not a rejected request)."""
        return self.status is None and "timed out" in str(self).lower()

    @property
    def fatal(self) -> bool:
        """Errors that will not go away by retrying this provider (bad key, no access, no free quota)."""
        if self.status in (401, 402, 403):
            return True
        return any(marker in str(self).lower() for marker in _NO_QUOTA)

    @property
    def model_unavailable(self) -> bool:
        return self.status in (400, 404, 422) and not self.fatal


def _version_key(model_id: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", model_id)[:4])


def heuristic_score(spec: ProviderSpec, model: str) -> float:
    """Quality guess from the model name, used until the model has a measured benchmark score."""
    name = model.lower()
    score = spec.base_score
    if "flash" in name and "lite" not in name:
        score += 10
    elif "pro" in name.split("/")[-1].split("-"):
        score += 8                      # stronger, but thinks longer (more tokens) and has smaller free quotas
    if "lite" in name:
        score -= 20
    if "gemma" in name:
        score -= 10
    if _SMALL.search(name):
        score -= 20
    elif _LARGE.search(name):
        score += 8
    for rank, pattern in enumerate(spec.prefer):
        if pattern in name:
            score += max(0, 4 - rank)
            break
    version = re.search(r"(?:^|[-_/])(\d)(?:\.(\d))?(?=[-_.]|$)", name.split("/")[-1])
    if version:
        score += min(10.0, 2 * float(f"{version.group(1)}.{version.group(2) or 0}"))
    return round(score, 1)


def exclusion_reason(spec: ProviderSpec, model: str, free: bool | None = None) -> str | None:
    """Why a listed model is not used, or None when it is a candidate."""
    name = model.lower()
    marker = next((x for x in NON_TEXT if x in name), None)
    if marker:
        return f"not a text chat model ({marker})"
    marker = next((x for x in spec.exclude if x in name), None)
    if marker:
        return f"excluded for {spec.name} ({marker})"
    if spec.require is not None and spec.require not in model and free is not True:
        return f"not free (no {spec.require})"
    if spec.free_only and free is not True:
        return "not free (not priced at zero)"
    return None


def usable_model(spec: ProviderSpec, model: str, free: bool | None = None) -> bool:
    return exclusion_reason(spec, model, free) is None


def choose_models(spec: ProviderSpec, discovered: list[str] | None, limit: int | None = MAX_MODELS_PER_PROVIDER,
                  free_ids: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """All usable text models, best heuristic first (newest version breaks ties)."""
    if not discovered:
        return list(spec.fallback_models)[:limit]
    ids = list(dict.fromkeys(m.removeprefix("models/") for m in discovered))
    usable = [m for m in ids if usable_model(spec, m, m in free_ids)]
    usable.sort(key=lambda m: (heuristic_score(spec, m), _version_key(m)), reverse=True)
    return (usable or list(spec.fallback_models))[:limit]


def _is_free(entry: dict) -> bool | None:
    """True when a listed model is priced at zero (OpenRouter-style "pricing"), None when unknown."""
    pricing = entry.get("pricing")
    if not isinstance(pricing, dict):
        return None
    try:
        return all(float(pricing.get(k, 1)) == 0 for k in ("prompt", "completion"))
    except (TypeError, ValueError):
        return None


def reasoning_effort(provider: str, model: str) -> str | None:
    """Lowest thinking level the provider documents for this model (saves hidden reasoning tokens).

    Gemini OpenAI compatibility: "none" only for 2.5 models other than 2.5 Pro; Gemini 3 and 2.5 Pro cannot turn
    thinking off, "low" is their lowest level. Groq: "low" for gpt-oss. Other providers: not sent.
    A model that rejects the field gets requests without it (ChatClient)."""
    name = model.lower()
    if provider == "gemini" and "gemma" not in name:
        return "none" if "2.5" in name and "pro" not in name else "low"
    if provider == "groq":
        if "gpt-oss" in name:
            return "low"
        if "qwen3" in name:                            # e.g. qwen/qwen3-32b: thinking off
            return "none"
    return None


class ChatClient:
    """Minimal OpenAI-compatible client (stdlib only)."""

    def __init__(self, spec: ProviderSpec, api_key: str, account_id: str | None = None,
                 timeout_s: float = REQUEST_TIMEOUT_S):
        self.spec = spec
        self.name = spec.name
        self._key = api_key
        self.base_url = spec.base_url.format(account_id=account_id or "")
        self.models_url = spec.models_url.format(account_id=account_id or "") if spec.models_url else None
        self.timeout_s = timeout_s
        self.models: list[str] = []
        # Token counts reported by the provider ("usage" field of OpenAI-compatible replies).
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.usage_by_model: dict[str, dict[str, int]] = {}
        self._no_reasoning_field: set[str] = set()
        self._no_json_mode: set[str] = set()
        self._max_tokens: dict[str, int] = {}
        self.latency: dict[str, list[float]] = {}          # model -> seconds of its recent successful replies
        self.cancel: threading.Event | None = None         # set by the job: a cancelled job must not wait for a reply

    def timeout_for(self, model: str) -> float:
        """Read timeout for the next request: the base value, or 3x the model's median reply time (D-087)."""
        seen = sorted(self.latency.get(model, []))
        if len(seen) < 2:
            return self.timeout_s
        return min(max(self.timeout_s, 3 * seen[len(seen) // 2]), LONG_TIMEOUT_S)

    def _request(self, path: str, body: dict | None = None, url: str | None = None,
                 timeout_s: float | None = None) -> dict:
        """One HTTP request. When the job's cancel event is set, the call returns at once with JobCancelled; the
        blocked socket is left to finish in a daemon thread (urllib cannot be interrupted from outside)."""
        cancel = self.cancel
        if cancel is None:
            return self._request_blocking(path, body, url, timeout_s)
        if cancel.is_set():
            raise JobCancelled()
        box: dict = {}

        def run() -> None:
            try:
                box["value"] = self._request_blocking(path, body, url, timeout_s)
            except BaseException as exc:             # noqa: BLE001 - re-raised in the caller's thread
                box["error"] = exc

        thread = threading.Thread(target=run, name=f"llm-{self.name}", daemon=True)
        thread.start()
        while thread.is_alive():
            thread.join(0.2)
            if cancel.is_set():
                raise JobCancelled()
        if "error" in box:
            raise box["error"]
        return box["value"]

    def _request_blocking(self, path: str, body: dict | None = None, url: str | None = None,
                          timeout_s: float | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT, "Accept": "application/json"}
        # Providers that answer anonymous callers (ovh, unturf) reject any Authorization header.
        if self._key and not (self.spec.keyless and self._key in KEYLESS_KEYS):
            headers["Authorization"] = f"Bearer {self._key}"
        request = urllib.request.Request(url or self.base_url + path, data=data,
                                         method="POST" if body is not None else "GET", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:1500]     # quota details ("limit: 0", PerDay) come late
            retry = exc.headers.get("Retry-After") if exc.headers else None
            try:
                retry_s = float(retry) if retry else None
            except ValueError:
                retry_s = None
            if retry_s is None:                            # Gemini puts the wait into the error body
                match = _RETRY_DELAY.search(detail)
                retry_s = float(match.group(1)) if match else None
            raise ProviderCallError(f"{self.name} HTTP {exc.code}: {detail}", exc.code, retry_s) from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ProviderCallError(f"{self.name} request failed: {exc}") from exc

    def fetch_models(self) -> list[dict]:
        """Every model the provider lists for this key: [{"id", "free"}] ("free" None when not stated)."""
        if self.models_url:
            data = self._request("", url=self.models_url)
            items = data.get("result") or []
            return [{"id": m["name"], "free": None} for m in items if isinstance(m, dict) and m.get("name")]
        data = self._request("/models")
        items = data.get("data") or []
        return [{"id": m["id"], "free": _is_free(m)} for m in items if isinstance(m, dict) and m.get("id")]

    def discover_models(self) -> list[str]:
        discovered, free_ids = None, set()
        if self.spec.discover:
            try:
                entries = self.fetch_models()
                discovered = [e["id"] for e in entries]
                free_ids = {e["id"].removeprefix("models/") for e in entries if e["free"]}
            except ProviderCallError as exc:
                if exc.fatal:
                    raise
                log.info("Model discovery failed for %s (%s); using fallback names", self.name, exc)
        self.models = choose_models(self.spec, discovered, free_ids=free_ids)
        return self.models

    def chat(self, model: str, messages: list[dict], max_tokens: int = 4096, temperature: float = 0.2,
             json_mode: bool = False, timeout_s: float | None = None) -> str:
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": min(max_tokens, self._max_tokens.get(model, max_tokens)), "stream": False}
        if json_mode and model not in self._no_json_mode:
            body["response_format"] = {"type": "json_object"}   # valid JSON from the decoder (no lost blocks)
        effort = None if model in self._no_reasoning_field else reasoning_effort(self.name, model)
        if effort:
            body["reasoning_effort"] = effort
        for _ in range(5):
            try:
                started = time.monotonic()
                data = self._request("/chat/completions", body, timeout_s=timeout_s)
                history = self.latency.setdefault(model, [])
                history.append(time.monotonic() - started)
                del history[:-20]
                break
            except ProviderCallError as exc:
                text = str(exc).lower()
                if exc.too_large and body["max_tokens"] > 1024:
                    body["max_tokens"] = self._max_tokens[model] = body["max_tokens"] // 2
                    log.info("%s:%s request too large; max_tokens %d", self.name, model, body["max_tokens"])
                    continue
                if exc.status not in (400, 413, 422):
                    raise
                if "response_format" in body and ("response_format" in text or "json" in text):
                    log.info("%s:%s rejects response_format; sending requests without it", self.name, model)
                    self._no_json_mode.add(model)
                    body.pop("response_format")
                    continue
                if "reasoning_effort" in body and "reason" in text:
                    log.info("%s:%s rejects reasoning_effort; sending requests without it", self.name, model)
                    self._no_reasoning_field.add(model)
                    body.pop("reasoning_effort")
                elif body["max_tokens"] > 2048 and any(m in text for m in _TOKEN_LIMIT):
                    # Smaller models cap the reply length (Groq allam: 4096, Cohere command-r: TOO_MANY_TOKENS).
                    body["max_tokens"] = self._max_tokens[model] = 2048
                    log.info("%s:%s limits max_tokens; using 2048", self.name, model)
                else:
                    raise
        else:
            raise ProviderCallError(f"{self.name}:{model} rejected the request")
        self._count(model, data)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderCallError(f"{self.name}: unexpected response shape") from exc
        return message.get("content") or ""

    def _count(self, model: str, data) -> None:
        per_model = self.usage_by_model.setdefault(model, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0})
        for counter in (self.usage, per_model):
            counter["requests"] += 1
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens"):
                if isinstance(usage.get(key), int):
                    self.usage[key] += usage[key]
                    per_model[key] += usage[key]


@dataclass(frozen=True)
class Route:
    """One model of one provider, with its quality score."""

    client: ChatClient
    model: str
    score: float
    measured: bool = False

    @property
    def key(self) -> str:
        return f"{self.client.name}:{self.model}"

    @property
    def provider(self) -> str:
        return self.client.name


@dataclass
class ModelRanking:
    """Measured scores ("provider:model" -> 0..100) and the minimum score for correcting translations."""

    scores: dict[str, float] = field(default_factory=dict)
    estimates: dict[str, float] = field(default_factory=dict)    # name-based guesses, never measured (D-047)
    excluded: set[str] = field(default_factory=set)
    floor: float = 65.0

    @classmethod
    def load(cls, *paths) -> "ModelRanking":
        """Later files override earlier ones (shipped defaults, then the user's own benchmark)."""
        ranking = cls()
        for path in paths:
            try:
                data = json.loads(open(path, encoding="utf-8").read())
            except (OSError, ValueError):
                continue
            ranking.floor = float(data.get("floor", ranking.floor))
            for key, entry in (data.get("models") or {}).items():
                if not isinstance(entry, dict):
                    continue
                if entry.get("exclude"):
                    ranking.excluded.add(key)
                    ranking.scores.pop(key, None)
                    ranking.estimates.pop(key, None)
                elif isinstance(entry.get("score"), (int, float)):
                    ranking.excluded.discard(key)
                    if entry.get("estimated"):
                        # An estimate never replaces a measurement and never counts as one.
                        if key not in ranking.scores:
                            ranking.estimates[key] = float(entry["score"])
                    else:
                        ranking.scores[key] = float(entry["score"])
                        ranking.estimates.pop(key, None)
        return ranking


class HealthStore:
    """Failures that later jobs remember, so a model that was dead ten minutes ago is not tried again first (D-086).

    One small JSON file {key: {"until": epoch seconds, "reason": text}}; a key is a route ("provider:model") or a
    provider name. Reading and writing never raise: a broken file only means "no memory".
    """

    def __init__(self, path=None, clock=time.time):
        self.path = path
        self._clock = clock
        self._entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path is None:
            return
        try:
            data = json.loads(open(self.path, encoding="utf-8").read())
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            self._entries = {k: v for k, v in data.items()
                             if isinstance(v, dict) and isinstance(v.get("until"), (int, float))}

    def active(self) -> dict[str, float]:
        """key -> seconds still remembered."""
        now = self._clock()
        return {k: v["until"] - now for k, v in self._entries.items() if v["until"] > now}

    def record(self, key: str, seconds: float, reason: str = "") -> None:
        now = self._clock()
        until = now + seconds
        if self._entries.get(key, {}).get("until", 0) >= until:
            return
        self._entries[key] = {"until": until, "reason": reason[:120]}
        self._save(now)

    def _save(self, now: float) -> None:
        if self.path is None:
            return
        self._entries = {k: v for k, v in self._entries.items() if v["until"] > now}
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self._entries, handle)
            os.replace(tmp, self.path)
        except OSError:
            log.debug("Could not save the provider health file", exc_info=True)


@dataclass
class ProviderPool:
    """All usable models of all providers as one quality-ranked list, with per-model cool-down."""

    clients: list[ChatClient]
    ranking: ModelRanking = field(default_factory=ModelRanking)
    cooldown_until: dict[str, float] = field(default_factory=dict)      # route key or provider name
    disabled: dict[str, str] = field(default_factory=dict)              # provider name -> reason
    failures: dict[str, int] = field(default_factory=dict)              # route key -> failed requests
    health: HealthStore | None = None                                   # failures remembered across jobs

    def apply_health(self) -> list[str]:
        """Start this job with what earlier jobs learned: remembered failures begin as cool-downs."""
        applied = []
        if self.health is None:
            return applied
        now = time.monotonic()
        for key, remaining in self.health.active().items():
            self.cooldown_until[key] = max(self.cooldown_until.get(key, 0), now + remaining)
            applied.append(key)
        return applied

    def remember(self, key: str, seconds: float, reason: str) -> None:
        """Keep a failure for later jobs (the cool-down inside this job is set separately)."""
        if self.health is not None:
            self.health.record(key, seconds, reason)

    def routes(self) -> list[Route]:
        order = {c.name: i for i, c in enumerate(self.clients)}
        routes = []
        for client in self.clients:
            for position, model in enumerate(client.models):
                key = f"{client.name}:{model}"
                if key in self.ranking.excluded:
                    continue
                measured = key in self.ranking.scores
                score = self.ranking.scores[key] if measured else heuristic_score(client.spec, model)
                if not measured and self.ranking.floor > 0:       # floor 0: no quality gate (tests, tools)
                    score = min(score, self.ranking.floor - UNMEASURED_MARGIN)
                    if key in self.ranking.estimates:
                        # Estimated models stay below the floor (they can translate when nothing measured is left,
                        # never correct) but keep their estimated order among themselves.
                        score = self.ranking.floor - UNMEASURED_MARGIN - (100.0 - self.ranking.estimates[key]) / 100.0
                elif not measured and key in self.ranking.estimates:
                    score = self.ranking.estimates[key]
                routes.append((Route(client, model, score, measured), order[client.name], position))
        routes.sort(key=lambda item: (-item[0].score, item[1], item[2]))
        return [route for route, _, _ in routes]

    def available(self, exclude: set[str] | None = None, min_score: float | None = None) -> list[Route]:
        """Usable routes, best first. exclude: route keys or provider names."""
        now = time.monotonic()
        exclude = exclude or set()
        return [r for r in self.routes()
                if r.provider not in self.disabled
                and self.cooldown_until.get(r.key, 0) <= now and self.cooldown_until.get(r.provider, 0) <= now
                and r.key not in exclude and r.provider not in exclude
                and (min_score is None or r.score >= min_score)]

    def cool_down(self, key: str, seconds: float) -> None:
        self.cooldown_until[key] = time.monotonic() + seconds

    def disable(self, name: str, reason: str) -> None:
        self.disabled[name] = reason

    def strike(self, route: Route, reason: str) -> None:
        """A failed request (timeout, server error, unusable reply): longer pauses, removed after 3 failures."""
        count = self.failures[route.key] = self.failures.get(route.key, 0) + 1
        if count >= 3:
            self.drop(route)
            self.remember(route.key, MEMORY_FAILURE_S, reason)
            log.info("%s failed %d times (%s); removed for this job", route.key, count, reason)
        else:
            self.cool_down(route.key, 60.0 * count)

    def drop(self, route: Route) -> None:
        if route.model in route.client.models:
            route.client.models.remove(route.model)

    def timeout(self, route: Route, reason: str) -> None:
        """No reply within the read timeout: the model is removed at once, here and for the next half hour (D-086)."""
        self.drop(route)
        self.remember(route.key, MEMORY_FAILURE_S, reason)
        log.info("%s timed out; removed for this job", route.key)


def build_clients(keys: dict, order: tuple[str, ...] = DEFAULT_ORDER) -> list[ChatClient]:
    clients = []
    for name in order:
        entry = keys.get(name) or {}
        key = entry.get("api_key", "").strip() if isinstance(entry, dict) else ""
        if key and name in PROVIDERS:
            clients.append(ChatClient(PROVIDERS[name], key, entry.get("account_id")))
    return clients
