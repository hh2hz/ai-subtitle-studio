"""Episode brief (task 2.1, D-055): one request per transcript part asks a strong model for the characters, how
they address each other and recurring terms, so block-by-block translation keeps gender, addressee and register
consistent. Never fatal: without a brief the translation works as before."""

from __future__ import annotations

import json
import logging
import time

from app.core.llm_providers import ProviderCallError
from app.utils.languages import english_name

log = logging.getLogger(__name__)

PART_CHARS = 24000          # transcript characters per brief request
MAX_PARTS = 4
MAX_ATTEMPTS = 6            # per part, each on a different provider
MAX_WAIT_S = 60.0           # longest retry_after honoured when no other provider is left
TIMEOUT_S = 90.0            # per brief request (a silent provider must not hold the job for minutes)
GENDERS = ("male", "female", "unknown")
REGISTERS = ("formal", "informal", "intimate", "hostile")

BRIEF_SYSTEM = """You prepare a translator's brief for subtitling a {src} video into {tgt}.
Read the whole transcript (speech recognition, may contain misheard words) and the media information.
Return JSON only, no markdown:
{{"summary": "<at most 10 sentences in English>",
 "characters": [{{"name": "<as written in the transcript>", "gender": "male|female|unknown",
                 "role": "<short English>", "aliases": ["<other forms of the name>"]}}],
 "relations": [{{"from": "<name>", "to": "<name>", "address": "<how 'from' addresses 'to'>",
                "register": "formal|informal|intimate|hostile"}}],
 "terms": [{{"source": "<recurring term, title or place>", "meaning": "<short English gloss>"}}]}}
List only people who speak or are spoken to or about. Use "unknown" gender when the transcript does not show it.
Do not invent facts that the transcript does not support."""


def _text(value, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def validate_brief(data) -> dict:
    """Keep only well-formed fields; invalid items are dropped."""
    data = data if isinstance(data, dict) else {}
    characters = []
    for c in data.get("characters") or []:
        if isinstance(c, dict) and _text(c.get("name"), 60):
            gender = c.get("gender") if c.get("gender") in GENDERS else "unknown"
            aliases = [_text(a, 60) for a in c.get("aliases") or [] if _text(a, 60)][:5]
            characters.append({"name": _text(c["name"], 60), "gender": gender, "role": _text(c.get("role"), 80),
                               "aliases": aliases})
    relations = []
    for r in data.get("relations") or []:
        if isinstance(r, dict) and _text(r.get("from"), 60) and _text(r.get("to"), 60):
            relations.append({"from": _text(r["from"], 60), "to": _text(r["to"], 60),
                              "address": _text(r.get("address"), 60),
                              "register": r.get("register") if r.get("register") in REGISTERS else "informal"})
    terms = [{"source": _text(t["source"], 60), "meaning": _text(t.get("meaning"), 120)}
             for t in data.get("terms") or [] if isinstance(t, dict) and _text(t.get("source"), 60)]
    return {"summary": _text(data.get("summary"), 1500), "characters": characters[:40],
            "relations": relations[:60], "terms": terms[:40]}


def is_empty(brief: dict | None) -> bool:
    return not brief or not (brief.get("characters") or brief.get("summary"))


def merge_briefs(parts: list[dict]) -> dict:
    """Union of characters (first entry per name wins), relations and terms; summaries joined in order."""
    merged = {"summary": " ".join(p["summary"] for p in parts if p["summary"])[:3000],
              "characters": [], "relations": [], "terms": []}
    seen: dict[str, set] = {"characters": set(), "relations": set(), "terms": set()}
    keys = {"characters": lambda x: x["name"].casefold(),
            "relations": lambda x: (x["from"].casefold(), x["to"].casefold()),
            "terms": lambda x: x["source"].casefold()}
    for part in parts:
        for field, key in keys.items():
            for item in part[field]:
                if key(item) not in seen[field]:
                    seen[field].add(key(item))
                    merged[field].append(item)
    return merged


def _parts(units: list[dict]) -> list[str]:
    lines = [f"[{int(u.get('start', 0)) // 60:02d}:{int(u.get('start', 0)) % 60:02d}] {u['text']}"
             for u in units if u.get("text", "").strip()]
    total = sum(len(line) + 1 for line in lines)
    count = min(MAX_PARTS, max(1, -(-total // PART_CHARS)))
    size = -(-len(lines) // count) if lines else 0
    return ["\n".join(lines[i:i + size]) for i in range(0, len(lines), size)] if size else []


def _ask(pool, system: str, payload: dict, parse, label: str, sleep):
    """Send `payload` to the best provider, then to others, until `parse(json)` returns something truthy."""
    from app.core.llm_translation import extract_json

    tried: set[str] = set()
    for _ in range(MAX_ATTEMPTS):
        routes = pool.available(exclude=tried)
        if not routes:
            break
        route = routes[0]
        tried.add(route.provider)             # the next attempt goes to another provider
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        saved = route.client.timeout_s
        route.client.timeout_s = min(saved, TIMEOUT_S)
        try:
            reply = route.client.chat(route.model, messages, max_tokens=4096, temperature=0.0, json_mode=True)
            parsed = parse(extract_json(reply))
            if parsed:
                return parsed
            log.warning("%s: %s returned nothing usable", label, route.key)
        except (ProviderCallError, ValueError) as exc:
            log.warning("%s: %s failed (%s)", label, route.key, str(exc)[:160])
            wait = getattr(exc, "retry_after", None)
            if wait and not pool.available(exclude=tried) and wait <= MAX_WAIT_S:
                sleep(wait)                     # no other provider left: wait once, then the same one again
                tried.discard(route.provider)
        finally:
            route.client.timeout_s = saved
    return None


def _ask_part(pool, system: str, media: dict, text: str, part: int, sleep) -> dict | None:
    def parse(data):
        brief = validate_brief(data)
        return None if is_empty(brief) else brief

    return _ask(pool, system, {"media": media, "transcript": text}, parse, f"Episode brief part {part}", sleep)


def build_brief(pool, source_language: str, target_language: str, units: list[dict], media: dict,
                sleep=time.sleep) -> dict | None:
    """The episode brief, or None when no part could be briefed."""
    system = BRIEF_SYSTEM.format(src=english_name(source_language), tgt=english_name(target_language))
    parts = [b for i, text in enumerate(_parts(units), 1)
             if (b := _ask_part(pool, system, media, text, i, sleep)) is not None]
    brief = merge_briefs(parts) if parts else None
    return None if is_empty(brief) else brief


def trim_brief(brief: dict | None, texts: list[str]) -> dict | None:
    """The part of the brief that concerns these lines: characters (and their relations) whose name or alias
    occurs in the texts, terms that occur; the summary always."""
    if not brief:
        return None
    haystack = " ".join(texts).casefold()

    def present(word: str) -> bool:
        return bool(word) and word.casefold() in haystack

    characters = [c for c in brief["characters"] if present(c["name"].split()[0])
                  or any(present(a) for a in c["aliases"])]
    names = {c["name"].casefold() for c in characters}
    relations = [r for r in brief["relations"] if r["from"].casefold() in names or r["to"].casefold() in names]
    terms = [t for t in brief["terms"] if present(t["source"])]
    return {"summary": brief["summary"], "characters": characters, "relations": relations, "terms": terms}


# -- speakers (task 3.2, D-099) ------------------------------------------------------------------------------

SPEAKER_SYSTEM = """You match the speakers of a {src} video to its characters.
"speakers" lists voice ids found by speaker detection (it can be wrong) with sample lines of each voice (speech
recognition, may contain misheard words). "characters" comes from the episode brief and may be empty.
Decide for every voice id who it is and whether the speaker is male or female, from how the lines address the speaker
and refer to the speaker, verb endings, self-reference and the characters list.
Return JSON only, no markdown:
{{"speakers": [{{"id": "<voice id>", "name": "<character name as in the characters list or the transcript, or unknown>",
                "gender": "male|female|unknown"}}]}}
Use "unknown" when the lines do not show it. Never guess."""
SAMPLES_PER_SPEAKER = 8
SAMPLE_CHARS = 160


def speaker_samples(units: list[dict]) -> list[dict]:
    """[{"id": "S1", "samples": [...]}] with the longest lines of each voice, in time order."""
    by_voice: dict[str, list[dict]] = {}
    for unit in units:
        if unit.get("speaker") and unit.get("text", "").strip():
            by_voice.setdefault(unit["speaker"], []).append(unit)
    result = []
    for voice in sorted(by_voice, key=lambda v: (len(v), v)):
        longest = sorted(by_voice[voice], key=lambda u: -len(u["text"]))[:SAMPLES_PER_SPEAKER]
        result.append({"id": voice, "samples": [u["text"][:SAMPLE_CHARS] for u in sorted(longest, key=lambda u: u["start"])]})
    return result


def validate_speakers(data, ids: set[str]) -> dict[str, dict]:
    """{voice id: {"name": str, "gender": male|female|unknown}} for known ids only."""
    result: dict[str, dict] = {}
    for item in (data.get("speakers") if isinstance(data, dict) else None) or []:
        if isinstance(item, dict) and item.get("id") in ids:
            name = _text(item.get("name"), 60)
            result[item["id"]] = {"name": "" if name.casefold() == "unknown" else name,
                                  "gender": item.get("gender") if item.get("gender") in GENDERS else "unknown"}
    return result


def build_speaker_map(pool, source_language: str, brief: dict | None, units: list[dict],
                      sleep=time.sleep) -> dict[str, dict] | None:
    """One request: voice id -> character and gender. None when it failed (the lines then carry only the id)."""
    samples = speaker_samples(units)
    if not samples:
        return None
    ids = {s["id"] for s in samples}
    payload = {"characters": [{"name": c["name"], "gender": c["gender"], "aliases": c["aliases"]}
                              for c in (brief or {}).get("characters", [])],
               "speakers": samples}
    system = SPEAKER_SYSTEM.format(src=english_name(source_language))
    return _ask(pool, system, payload, lambda data: validate_speakers(data, ids) or None, "Speaker map", sleep)


def speaker_fields(label: str | None, speakers: dict[str, dict] | None) -> dict:
    """The "speaker" and "speaker_gender" payload fields of a line (empty when the line has no voice)."""
    if not label:
        return {}
    info = (speakers or {}).get(label) or {}
    fields = {"speaker": info.get("name") or label}
    if info.get("gender") and info["gender"] != "unknown":
        fields["speaker_gender"] = info["gender"]
    return fields
