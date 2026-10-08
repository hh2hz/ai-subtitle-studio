"""Subtitle translation refinement with cloud LLMs (free API tiers).

Two modes per block of lines (with surrounding context, media details and a running glossary of names):
- "correct" (default, fewest tokens): the model receives each source line with the local model's translation
  and returns only the lines it corrects; all other lines are accepted unchanged (DECISIONS D-033).
- "translate": the model translates every line from scratch; optionally a second model from a different
  provider reviews the block and returns suggestions (never applied automatically).
Every model of every provider is a candidate, best measured/estimated quality first; a failure or rate limit
moves on to the next model. Correcting needs a model at or above the quality floor; in "translate" mode weaker
models are a last resort and their lines are flagged. Lines nobody handled keep the local translation and are
flagged.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from app.core import names
from app.core.errors import JobCancelled
from app.core.llm_providers import (
    LONG_TIMEOUT_S, MEMORY_DAILY_S, MEMORY_FAILURE_S, MEMORY_MONTHLY_S, ProviderCallError, ProviderPool, Route,
)
from app.core import style as style_guides
from app.core import risk
from app.core.brief import trim_brief
from app.core.verification import check_line
from app.utils.languages import english_name

log = logging.getLogger(__name__)

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
MAX_WAIT_S = 30.0
COOLDOWN_S = 120.0
DAY_S = 24 * 3600.0

def _source_line(line: dict) -> dict:
    d = {"id": line["id"], "source": line["source"]}
    if "max_chars" in line:
        d["max_chars"] = line["max_chars"]
    for key in ("speaker", "speaker_gender"):
        if line.get(key):
            d[key] = line[key]
    return d


CONDENSE_SYSTEM = """You are a senior professional subtitle editor for {tgt}.
Shorten the subtitle lines in "lines" so each line fits within its "max_chars" character limit.
Rules:
1. Condense without losing meaning. Never drop key facts, names, negation, questions or tone.
2. Return JSON only, no markdown:
{{"lines": [{{"id": <id>, "text": "<condensed text>"}}]}}"""


TRANSLATOR_SYSTEM = """You are a senior professional subtitle translator from {src} into {tgt}.
Translate the dialogue lines in "lines". Rules:
1. Convey the meaning, intent and tone of each line as a native {tgt} subtitler would. Never translate word by word. Idioms, slang, insults and colloquial address terms must become natural {tgt} equivalents.
2. Exactly one translation per id. Never merge, split, skip or reorder lines.
3. Use "previous_lines" and "next_lines" only as context to understand who is speaking and what is meant.
4. Person names stay names: transliterate them from their {src} spelling, consistently, following "glossary" when it has the name. Never translate a name as an ordinary word. Never use the names of a dubbed or localized version of the work, even if you know them: this subtitle is for the original.
5. Keep questions, imperatives, negation, numbers and the addressee correct.
6. Subtitle style: concise, easy to read; ideally at most 42 characters per line and two lines.
7. "source" comes from speech recognition and can contain misheard words: when a word makes no sense in context, translate what was most likely said.
8. "brief", when present, describes the episode: use its characters, relations and terms for gender, addressee, register and terms. The lines are the authority when they contradict it.
9. Lines may carry "speaker" (a character name or a voice id) and "speaker_gender": use the speaker's gender and the gender of the person addressed (usually the speaker of the neighbouring lines) for verb, pronoun and adjective agreement.
{style}
Reply with JSON only, no markdown:
{{"lines": [{{"id": <id>, "text": "<translation>"}}], "names": [{{"source": "<name as in source>", "target": "<spelling in {tgt}>"}}]}}"""

JUDGE_SYSTEM = """You are a senior professional subtitle editor for {tgt}, judging two candidate translations from {src}.
Each item in "lines" has the {src} "source" (speech recognition, may contain misheard words) and two {tgt} candidates, "A" and "B", written by different models. Decide which one is correct and natural: meaning (negation, questions, numbers, who is addressed, the speaker's gender), names (follow "glossary"), register and fluency. A different correct wording is fine: pick the better one, and "A" when they are equal. Answer "both_wrong" only when neither conveys the meaning of the source.
"brief", when present, describes the episode; the lines are the authority when they contradict it.
Reply with JSON only, no markdown:
{{"verdicts": [{{"id": <id>, "choice": "A" | "B" | "both_wrong", "reason": "<short English reason>"}}]}}"""

CORRECTOR_SYSTEM = """You are a senior {tgt} subtitle editor. Each item in "lines" has the {src} "source" and a draft "translation" made by a small local translation model.
Check every line against its source and correct only real errors: wrong or lost meaning, idioms, slang or address terms translated literally, wrong addressee/gender/number, questions or negation lost, wrong or inconsistent names, untranslated or missing words, broken grammar. An empty translation must be translated.
Do not rewrite acceptable lines for taste and do not list them. Use "previous_lines" and "next_lines" only as context. "source" comes from speech recognition and can contain misheard words: the translation must follow what was most likely said. Person names: transliterate them from their {src} spelling, consistently, following "glossary"; a name taken from a dubbed or localized version of the work is an error to correct.
Corrected lines keep the subtitle style: concise, ideally at most 42 characters per line and two lines.
"brief", when present, describes the episode: use its characters, relations and terms to check gender, addressee, register and terms. The lines are the authority when they contradict it.
Lines may carry "speaker" (a character name or a voice id) and "speaker_gender": check verb, pronoun and adjective agreement against the speaker's gender and the gender of the person addressed.
{style}
Reply with JSON only, no markdown:
{{"corrections": [{{"id": <id>, "text": "<corrected translation>"}}], "names": [{{"source": "<name as in source>", "target": "<spelling in {tgt}>"}}]}}
"corrections" is an empty list when every line is acceptable. "names" lists only person or place names in these lines that are not in the glossary."""

NAME_FIX_SYSTEM = """You fix person and place names in {tgt} subtitles translated from {src}. In each item of "lines", the names listed in "names" are missing or wrong in "translation" (for example replaced by a name that a dubbed version of the work uses). Rewrite each translation so that every listed name appears, transliterated from its {src} spelling; use "target" when it is given. Change nothing else unless the sentence needs it.
Reply with JSON only, no markdown:
{{"lines": [{{"id": <id>, "text": "<fixed translation>"}}]}}"""

MODES = ("correct", "translate")

# Per-target style rules live in app/resources/style_guides.json (loaded by app/core/style.py, D-060).


def extract_json(text: str) -> dict:
    """Parse the JSON object in a model reply (tolerates think blocks and code fences)."""
    text = _THINK.sub("", text).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    try:
        return json.loads(text[start:text.rfind("}") + 1])
    except ValueError:
        pass
    try:
        # Some models append a second object or a note after the JSON: take the first complete object.
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    # Broken JSON (an unescaped quote inside a line, a reply cut off at max_tokens): keep every complete
    # {"id", "text"} item instead of losing the whole block (D-047). Missing lines go to the next model.
    items = [{"id": int(m.group(1)), "text": m.group(2).replace('\\"', '"')} for m in _ITEM.finditer(text)]
    if not items:
        raise ValueError("no usable JSON in reply")
    return {"lines": items, "corrections": items, "salvaged": True}


_ITEM = re.compile(r'\{\s*"id"\s*:\s*"?(\d+)"?\s*,\s*"text"\s*:\s*"(.*?)"\s*(?:,\s*"reason"\s*:\s*"[^"]*"\s*)?\}'
                   r'(?=\s*[,\]}])', re.S)


def _clean(text: str) -> str:
    text = " ".join(str(text).replace("\\n", " ").split()).strip()
    # Remove quotes only when they wrap the whole line; a quotation inside the line keeps both of its marks
    # (stripping one side left 'labels saying "police' in the real run, D-037).
    if len(text) >= 2 and text[0] == text[-1] == '"' and text.count('"') == 2:
        text = text[1:-1].strip()
    return text


@dataclass
class BlockResult:
    translations: dict[int, str]
    translator: str | None
    reviewer: str | None = None
    corrections: dict[int, dict] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    changed: set[int] | None = None     # "correct" mode: ids whose local translation was corrected
    weak: set[int] = field(default_factory=set)   # ids translated by a model below the quality floor
    rejected_names: dict[str, str] = field(default_factory=dict)   # model renderings that fail the name check
    name_issues: dict[int, list[str]] = field(default_factory=dict)   # names still wrong after the fix attempt
    condense_requests: int = 0
    risky: dict[int, list[str]] = field(default_factory=dict)        # lines sent to the second model, with reasons
    disagreement: dict[int, dict] = field(default_factory=dict)      # {"choice", "reason", "a", "b"} (D-104)
    second: dict[int, str] = field(default_factory=dict)             # lines now written by the second model
    judge: str | None = None
    double_check_requests: int = 0


class LlmRefiner:
    def __init__(self, pool: ProviderPool, source_language: str, target_language: str, media: dict,
                 mode: str = "translate", review: bool = True, sleep: Callable[[float], None] = time.sleep):
        if mode not in MODES:
            raise ValueError(f"Unknown refine mode {mode!r}")
        self.pool = pool
        self.src, self.tgt = source_language, target_language
        self.media = media
        self.mode = mode
        self.review = review          # double-check risky lines with a second model (D-104)
        self._sleep = sleep
        self.cancel: threading.Event | None = None   # set by the pipeline; see set_cancel()
        self.on_step: Callable[[float], None] | None = None   # progress inside one block: 0.0 - 1.0 (D-115)
        style = style_guides.cloud_style(target_language)
        fields = dict(src=english_name(source_language), tgt=english_name(target_language), style=style)
        self._translator_system = TRANSLATOR_SYSTEM.format(**fields)
        self._judge_system = JUDGE_SYSTEM.format(**fields)
        self._corrector_system = CORRECTOR_SYSTEM.format(**fields)
        self._name_fix_system = NAME_FIX_SYSTEM.format(**fields)
        self._condense_system = CONDENSE_SYSTEM.format(tgt=english_name(target_language))
        self.known_names: set[str] = set()       # every source name seen in this job (checked on every line)
        self.rejected_names: dict[str, set[str]] = {}   # renderings that failed the transliteration check
        self._locked_route: Route | None = None
        self.brief: dict | None = None            # episode brief (D-055), trimmed per block   # style lock: first successful translator route

    # -- provider calls ------------------------------------------------------------------------

    def set_cancel(self, cancel: threading.Event) -> None:
        """Make every request and every wait of this refiner stop at once when `cancel` is set."""
        self.cancel = cancel
        for client in self.pool.clients:
            client.cancel = cancel
        plain_sleep = self._sleep

        def sleep(seconds: float) -> None:
            if cancel.wait(seconds):
                raise JobCancelled()
            plain_sleep(0)

        self._sleep = sleep

    def has_routes(self) -> bool:
        return bool(self.pool.available(min_score=self._min_score()))

    def _min_score(self) -> float | None:
        # Correcting needs a model good enough to judge the local translation; weak models only translate,
        # as a last resort, and their lines are flagged for review.
        return self.pool.ranking.floor if self.mode == "correct" else None

    def _release_lock(self, route: Route) -> None:
        """Drop the style lock when the locked route leaves the pool."""
        if self._locked_route is route:
            log.info("Style lock released (%s left the pool); next success will re-lock", route.key)
            self._locked_route = None

    def _add_brief(self, payload: dict, lines: list[dict], previous: list[dict], following: list[dict]) -> None:
        brief = trim_brief(self.brief, [x.get("source", "") for x in previous + lines + following])
        if brief:
            payload["brief"] = brief

    def _pick(self, routes: list[Route]) -> Route:
        """Style lock (D-051): the locked model writes every block it can; the best available route otherwise.
        A lock whose model has left the pool (dropped, or removed after repeated failures) is released."""
        lock = self._locked_route
        if lock is not None and lock.model not in lock.client.models:
            self._release_lock(lock)
            lock = None
        if lock is not None:
            for route in routes:
                if route.key == lock.key:
                    return route
        return routes[0]

    def _lock(self, route: Route) -> None:
        """Lock the first model that succeeds, unless it is a below-floor last resort."""
        floor = self.pool.ranking.floor
        if self._locked_route is None and not (floor and route.score < floor):
            self._locked_route = route
            log.info("Style lock: locked to %s", route.key)

    def _call(self, route: Route, system: str, payload: dict) -> dict:
        """One request to one model; raise ProviderCallError (after updating the pool) when it fails."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        timeout_s = route.client.timeout_for(route.model) if hasattr(route.client, "timeout_for") else None
        long_retry_used = False
        for attempt in range(3):
            try:
                reply = route.client.chat(route.model, messages, json_mode=True, temperature=0.0,
                                          **({"timeout_s": timeout_s} if timeout_s else {}))
                data = extract_json(reply)
                if data.get("salvaged"):
                    log.info("%s returned broken JSON; kept %d complete lines", route.key, len(data["lines"]))
                return {"model": route.key, "data": data}
            except ProviderCallError as exc:
                if exc.fatal:
                    self.pool.disable(route.provider, str(exc))
                    self.pool.remember(route.provider, MEMORY_FAILURE_S, "refused the key")
                    self._release_lock(route)
                    raise
                if exc.not_free:
                    self.pool.drop(route)
                    self.pool.remember(route.key, MEMORY_MONTHLY_S, "no free quota")
                    self._release_lock(route)
                    log.info("%s has no free quota for this key; removed", route.key)
                    raise
                if exc.timed_out:
                    # A measured model above the quality floor is worth one longer second chance (D-087).
                    worth_it = route.measured and route.score >= self.pool.ranking.floor
                    if worth_it and not long_retry_used and timeout_s is not None and timeout_s < LONG_TIMEOUT_S:
                        long_retry_used = True
                        timeout_s = LONG_TIMEOUT_S
                        log.info("%s timed out; one more try with a %.0f s timeout", route.key, timeout_s)
                        continue
                    self.pool.timeout(route, "read timeout")
                    self._release_lock(route)
                    raise
                if exc.rate_limited:
                    if exc.monthly_quota:
                        # The allowance belongs to the key, so every model of the provider is skipped (D-086).
                        log.info("%s: monthly quota used up; provider skipped", route.provider)
                        self.pool.cool_down(route.provider, DAY_S)
                        self.pool.remember(route.provider, MEMORY_MONTHLY_S, "monthly quota")
                        self._release_lock(route)
                        raise
                    if exc.daily_quota:
                        log.info("%s: daily quota used up; skipped for this job", route.key)
                        self.pool.cool_down(route.key, DAY_S)
                        self.pool.remember(route.key, MEMORY_DAILY_S, "daily quota")
                        raise
                    wait = exc.retry_after if exc.retry_after is not None else MAX_WAIT_S + 1
                    if wait <= MAX_WAIT_S and attempt == 0:
                        log.info("%s rate limited; waiting %.0f s", route.key, wait)
                        self._sleep(wait)
                        continue
                    self.pool.cool_down(route.key, max(wait, COOLDOWN_S))
                    raise
                if exc.model_unavailable or exc.too_large:
                    # Not available for this account, or requests that will never fit its limits.
                    self.pool.drop(route)
                    self._release_lock(route)
                    log.info("%s unavailable, removed (%s)", route.key, exc)
                    raise
                self.pool.strike(route, str(exc)[:120])           # timeout, server error, ...
                raise
            except (ValueError, KeyError, TypeError) as exc:
                self.pool.strike(route, "unusable JSON")
                raise ProviderCallError(f"{route.key} returned unusable JSON: {exc}") from exc
        raise ProviderCallError(f"{route.key}: no reply")

    # -- public ----------------------------------------------------------------------------------

    def translate_block(self, lines: list[dict], previous: list[dict], following: list[dict],
                        glossary: dict[str, str]) -> BlockResult:
        """lines: [{"id", "source", "draft"}]; previous: [{"source", "translation"}]; following: [{"source"}]."""
        self._step(0.0)
        if self.mode == "correct":
            result = self._correct_block(lines, previous, following, glossary)
        else:
            result = self._translate_block(lines, previous, following, glossary)
        self._step(0.4)
        if result.translations:
            self._enforce_names(lines, glossary, result)
            self._step(0.5)
            if self.review:
                self._double_check(lines, previous, following, glossary, result)
            self._step(0.85)
            self._condense(lines, result, glossary)
        self._step(1.0)
        return result

    def _step(self, fraction: float) -> None:
        """Tell the pipeline how far this block is, so the progress bar moves while a block is being worked on."""
        if self.on_step is not None:
            try:
                self.on_step(fraction)
            except Exception:                    # noqa: BLE001 - progress reporting must never break a translation
                log.debug("progress callback failed", exc_info=True)

    def _translate_block(self, lines: list[dict], previous: list[dict], following: list[dict],
                         glossary: dict[str, str]) -> BlockResult:
        wanted = {line["id"] for line in lines}
        result = BlockResult(translations={}, translator=None)
        used: set[str] = set()
        floor = self.pool.ranking.floor
        while wanted - set(result.translations):
            routes = self.pool.available(exclude=used)
            if not routes:
                break
            route = self._pick(routes)
            used.add(route.key)
            missing = [line for line in lines if line["id"] not in result.translations]
            payload = {
                "media": self.media,
                "glossary": [{"source": k, "target": v} for k, v in glossary.items()],
                "rule": "stay within max_chars by condensing, never by dropping meaning",
                "previous_lines": previous,
                "lines": [_source_line(l) for l in missing],
                "next_lines": following,
            }
            self._add_brief(payload, lines, previous, following)
            try:
                reply = self._call(route, self._translator_system, payload)
            except ProviderCallError as exc:
                result.notes.append(f"translator {route.key} failed: {exc}")
                log.warning("Translator %s failed: %s", route.key, exc)
                if route is self._locked_route:
                    log.info("Style lock: %s failed; another model does this block only", route.key)
                continue
            data = reply["data"]
            for item in data.get("lines", []) if isinstance(data.get("lines"), list) else []:
                try:
                    line_id = int(item["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                text = _clean(item.get("text", ""))
                if line_id in wanted and text and line_id not in result.translations:
                    result.translations[line_id] = text
                    if route.score < floor:
                        result.weak.add(line_id)
            self._collect_names(data, result)
            result.translator = result.translator or reply["model"]
            self._lock(route)
        return result

    def _correct_block(self, lines: list[dict], previous: list[dict], following: list[dict],
                       glossary: dict[str, str]) -> BlockResult:
        """Send source + local translation; apply the returned corrections, accept every other line."""
        result = BlockResult(translations={}, translator=None, changed=set())
        drafts = {line["id"]: _clean(line.get("draft") or "") for line in lines}
        payload = {
            "media": self.media,
            "glossary": [{"source": k, "target": v} for k, v in glossary.items()],
            "rule": "stay within max_chars by condensing, never by dropping meaning",
            "previous_lines": previous,
            "lines": [{**_source_line(l), "translation": drafts[l["id"]]} for l in lines],
            "next_lines": following,
        }
        self._add_brief(payload, lines, previous, following)
        used: set[str] = set()
        while True:
            routes = self.pool.available(exclude=used, min_score=self._min_score())
            if not routes:
                return result
            route = self._pick(routes)
            used.add(route.key)
            try:
                reply = self._call(route, self._corrector_system, payload)
            except ProviderCallError as exc:
                result.notes.append(f"corrector {route.key} failed: {exc}")
                log.warning("Corrector %s failed: %s", route.key, exc)
                if route is self._locked_route:
                    log.info("Style lock: %s failed; another model does this block only", route.key)
                continue
            data = reply["data"]
            corrections = data.get("corrections")
            if not isinstance(corrections, list):
                result.notes.append(f"corrector {route.key} returned no corrections list")
                continue
            fixes: dict[int, str] = {}
            for item in corrections:
                try:
                    line_id = int(item["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                text = _clean(item.get("text", ""))
                if line_id in drafts and text:
                    fixes[line_id] = text
            for line_id, draft in drafts.items():
                text = fixes.get(line_id, draft)
                if text:                      # an empty draft the model did not translate stays missing
                    result.translations[line_id] = text
                if line_id in fixes and fixes[line_id] != draft:
                    result.changed.add(line_id)
            self._collect_names(data, result)
            result.translator = reply["model"]
            self._lock(route)
            return result

    # -- names ---------------------------------------------------------------------------------------

    def _collect_names(self, data: dict, result: BlockResult) -> None:
        """Keep a model's name spellings only when they are transliterations of the source name."""
        for item in data.get("names", []) if isinstance(data.get("names"), list) else []:
            if not (isinstance(item, dict) and item.get("source") and item.get("target")):
                continue
            source, target = _clean(item["source"]), _clean(item["target"])
            if not names.is_name(source):              # a title alone ("Abi", "Bey") is not a name
                continue
            self.known_names.add(source)
            if not names.supported_target(self.tgt) or names.transliteration_ok(source, target):
                result.names[source] = target
            else:
                result.rejected_names[source] = target
                self.rejected_names.setdefault(source, set()).add(target)
                log.info("Name rendering rejected (not a transliteration): %s -> %s", source, target)

    def _name_issues(self, lines: list[dict], result: BlockResult, glossary: dict[str, str]) -> dict[int, list[str]]:
        known = self.known_names | set(glossary) | set(result.names)
        rejected = {k: set(v) for k, v in self.rejected_names.items()}
        for k, v in result.rejected_names.items():
            rejected.setdefault(k, set()).add(v)
        issues = {}
        for line in lines:
            text = result.translations.get(line["id"])
            if text:
                found = names.line_issues(line["source"], text, known, rejected)
                if found:
                    issues[line["id"]] = found
        return issues

    def _enforce_names(self, lines: list[dict], glossary: dict[str, str], result: BlockResult) -> None:
        """One extra request for lines whose names are missing or replaced (e.g. by dubbed names, D-039)."""
        if not names.supported_target(self.tgt):
            return                     # Chinese/Japanese targets: names are not transliterated letter by letter
        by_id = {line["id"]: line for line in lines}
        found_issues = self._name_issues(lines, result, glossary)
        # Rewrite automatically only where the word is clearly a name: capitalised mid-sentence, or reported as a
        # name by the translator for this block. Other lines ("Aslan gibi..." = "like a lion") are only flagged.
        listed = set(result.names) | set(result.rejected_names)
        issues = {i: found for i, found in found_issues.items()
                  if any(n in listed or names.certain_in(n, by_id[i]["source"]) for n in found)}
        if not issues:
            result.name_issues = found_issues
            return
        spelled = {**glossary, **result.names}
        payload = {"media": self.media,
                   "lines": [{"id": i, "source": by_id[i]["source"], "translation": result.translations[i],
                              "names": [{"source": n, **({"target": spelled[n]} if n in spelled else {})}
                                        for n in found]}
                             for i, found in issues.items()]}
        for route in self.pool.available(min_score=self.pool.ranking.floor)[:2]:
            try:
                reply = self._call(route, self._name_fix_system, payload)
            except ProviderCallError as exc:
                result.notes.append(f"name fix {route.key} failed: {exc}")
                continue
            items = reply["data"].get("lines")
            for item in items if isinstance(items, list) else []:
                try:
                    line_id = int(item["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                text = _clean(item.get("text", ""))
                if line_id in issues and text:
                    trial = BlockResult(translations={line_id: text}, translator=None,
                                        rejected_names=result.rejected_names)
                    if not self._name_issues([by_id[line_id]], trial, glossary):
                        result.translations[line_id] = text
                        if result.changed is not None:
                            result.changed.add(line_id)
                        log.info("Fixed names in line %d: %s", line_id, ", ".join(issues[line_id]))
            break
        result.name_issues = self._name_issues(lines, result, glossary)

    def _second_translation(self, lines: list[dict], previous: list[dict], following: list[dict],
                            glossary: dict[str, str], routes: list[Route], result: BlockResult) -> tuple[dict, str | None]:
        """The risky lines translated again by the best route of another provider (same payload, brief included)."""
        payload = {
            "media": self.media,
            "glossary": [{"source": k, "target": v} for k, v in glossary.items()],
            "rule": "stay within max_chars by condensing, never by dropping meaning",
            "previous_lines": previous,
            "lines": [_source_line(l) for l in lines],
            "next_lines": following,
        }
        self._add_brief(payload, lines, previous, following)
        wanted = {l["id"] for l in lines}
        for route in routes[:2]:
            result.double_check_requests += 1
            try:
                reply = self._call(route, self._translator_system, payload)
            except ProviderCallError as exc:
                result.notes.append(f"second translation {route.key} failed: {exc}")
                continue
            found: dict[int, str] = {}
            items = reply["data"].get("lines")
            for item in items if isinstance(items, list) else []:
                try:
                    line_id = int(item["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                text = _clean(item.get("text", ""))
                if line_id in wanted and text and line_id not in found:
                    found[line_id] = text
            if found:
                return found, reply["model"]
        return {}, None

    def _double_check(self, lines: list[dict], previous: list[dict], following: list[dict],
                      glossary: dict[str, str], result: BlockResult) -> None:
        """Second model on the risky lines only (task 2.3, D-104).

        A line is risky when a cheap rule suggests its translation is wrong (risk.py). The best route of ANOTHER
        provider above the quality floor translates only those lines again; a judge (a third provider when there is
        one) picks A, B or both_wrong. B replaces A only when it passes the rule check and the names check; every
        other disagreement keeps A, flagged `ai_disagreement`, with both candidates for the human reviewer."""
        by_id = {l["id"]: l for l in lines}
        previous_text = None
        reasons: dict[int, list[str]] = {}
        issues = self._name_issues(lines, result, glossary)
        for line in lines:
            text = result.translations.get(line["id"])
            if text is None:
                continue
            reasons[line["id"]] = risk.risk_reasons(
                line["source"], text, self.tgt, self.src, previous_pair=previous_text,
                weak=line["id"] in result.weak, name_issue=line["id"] in issues,
                audio_confidence=line.get("audio_confidence"))
            previous_text = (line["source"], text)
        risky = risk.select(reasons)
        result.risky = risky
        if not risky:
            return
        floor = self.pool.ranking.floor
        first = (result.translator or "").split(":")[0]
        second_routes = self.pool.available(exclude={first} if first else None, min_score=floor)
        if not second_routes:
            result.notes.append("no second provider above the quality floor for the double check")
            return
        subset = [by_id[i] for i in risky]
        second, second_model = self._second_translation(subset, previous, following, glossary, second_routes, result)
        pairs = {i: (result.translations[i], second[i]) for i in risky
                 if i in second and second[i].strip() != result.translations[i].strip()}
        if not pairs:
            return
        second_provider = (second_model or "").split(":")[0]
        judges = self.pool.available(exclude={first, second_provider} - {""}, min_score=floor) \
            or self.pool.available(exclude={first} - {""}, min_score=floor)
        verdicts: dict[int, dict] = {}
        payload = {"media": self.media,
                   "glossary": [{"source": k, "target": v} for k, v in glossary.items()],
                   "previous_lines": previous,
                   "lines": [{**{k: v for k, v in _source_line(by_id[i]).items() if k != "max_chars"},
                              "A": a, "B": b} for i, (a, b) in pairs.items()],
                   "next_lines": following}
        self._add_brief(payload, lines, previous, following)
        for route in judges[:2]:
            result.double_check_requests += 1
            try:
                reply = self._call(route, self._judge_system, payload)
            except ProviderCallError as exc:
                result.notes.append(f"judge {route.key} failed: {exc}")
                continue
            items = reply["data"].get("verdicts")
            for item in items if isinstance(items, list) else []:
                try:
                    line_id = int(item["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                choice = str(item.get("choice", "")).strip()
                if line_id in pairs and choice in ("A", "B", "both_wrong"):
                    verdicts[line_id] = {"choice": choice, "reason": _clean(item.get("reason", ""))}
            result.judge = reply["model"]
            break
        for line_id, verdict in verdicts.items():
            a, b = pairs[line_id]
            line = by_id[line_id]
            if verdict["choice"] == "A":
                continue
            record = {"choice": verdict["choice"], "reason": verdict["reason"], "a": a, "b": b}
            if verdict["choice"] == "B":
                trial = BlockResult(translations={line_id: b}, translator=None, names=result.names,
                                    rejected_names=result.rejected_names)
                if not check_line(line["source"], b, self.tgt) and not self._name_issues([line], trial, glossary):
                    result.translations[line_id] = b
                    result.second[line_id] = second_model or ""
                    if result.changed is not None:
                        result.changed.add(line_id)
                    log.info("Double check: line %d now uses %s (%s)", line_id, second_model, verdict["reason"])
                    continue
                record["choice"] = "B_rejected"
            result.disagreement[line_id] = record

    def _condense_adds_name_issue(self, line: dict, old_text: str, new_text: str, result: BlockResult,
                                  glossary: dict[str, str]) -> bool:
        """True when the condensed text has a name problem that the current text does not have."""
        if not names.supported_target(self.tgt):
            return False
        def issues(text: str) -> set[str]:
            probe = BlockResult(translations={line["id"]: text}, translator=None, names=result.names,
                                rejected_names=result.rejected_names)
            return set(self._name_issues([line], probe, glossary).get(line["id"], []))
        return bool(issues(new_text) - issues(old_text))

    def _condense(self, lines: list[dict], result: BlockResult, glossary: dict[str, str] | None = None,
                  factor: float = 1.2) -> None:
        """At most ONE extra condense request per call, only for lines over `factor` * max_chars (1.2 per block;
        1.0 for the lines that are still too fast after timing, D-100)."""
        by_id = {line["id"]: line for line in lines}
        over = [
            {"id": line["id"], "source": line["source"], "translation": result.translations[line["id"]],
             "max_chars": line["max_chars"]}
            for line in lines
            if line["id"] in result.translations
            and "max_chars" in line
            and len(result.translations[line["id"]]) > factor * line["max_chars"]
        ]
        if not over:
            return
        route = self._locked_route
        usable = {r.key for r in self.pool.available()}
        if route is None or route.key not in usable:
            # The locked model is rate limited, cooled down or gone: do not spend a request on it again.
            routes = self.pool.available(min_score=self._min_score())
            route = self._pick(routes) if routes else None
        if route is None:
            return
        payload = {
            "media": self.media,
            "instruction": "Shorten each translation to fit within max_chars without losing meaning.",
            "lines": over,
        }
        result.condense_requests += 1
        try:
            reply = self._call(route, self._condense_system, payload)
        except ProviderCallError as exc:
            result.notes.append(f"condense {route.key} failed: {exc}")
            log.warning("Condense %s failed: %s", route.key, exc)
            return
        over_ids = {line["id"] for line in over}
        items = reply["data"].get("lines")
        for item in items if isinstance(items, list) else []:
            try:
                line_id = int(item["id"])
            except (KeyError, TypeError, ValueError):
                continue
            text = _clean(item.get("text", ""))
            if line_id in over_ids and text:
                old_text = result.translations[line_id]
                if len(text) < len(old_text) and not check_line(by_id[line_id]["source"], text, self.tgt) \
                        and not self._condense_adds_name_issue(by_id[line_id], old_text, text, result, glossary or {}):
                    result.translations[line_id] = text
                    if result.changed is not None:
                        result.changed.add(line_id)
                    log.info("Condensed line %d: %d -> %d chars (budget %d)",
                             line_id, len(old_text), len(text), by_id[line_id]["max_chars"])

