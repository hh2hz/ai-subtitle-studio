"""Automatic evaluation of AI Subtitle Studio output (docs/EXECUTOR_HANDOFF.md, phase 1; DECISIONS D-049).

No human reference is needed. Quality is measured by:
- pairwise judging: 2-3 AI judges from different providers compare two runs line by line (A/B order randomised
  per line, majority vote);
- an absolute judge that lists errors per line (safety net);
- metrics without a judge: flagged lines, subtitle QA flags, name issues, time and tokens;
- optional human references made with tools/make_reference.py (WER, CER, chrF).

Usage (from the project folder, with the project venv):
    python tools/evaluate.py make-set                       # build evaluation/set.json from your finished jobs
    python tools/evaluate.py run --label baseline           # run the app on every set item (current settings)
    python tools/evaluate.py run --label rerun --reuse-asr baseline   # same settings, transcription reused
    python tools/evaluate.py score baseline                 # metrics + absolute judge for one run
    python tools/evaluate.py compare baseline rerun         # pairwise + absolute judge, report in evaluation/reports
    python tools/evaluate.py reference reference/<name> --run baseline   # compare a run with a human reference

A settings file (`run --settings x.json`) changes the app settings for one run: {"mode": "balanced",
"extras": {"llm_review": true}}. Without it the run uses the settings saved in the app.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("evaluate")

EVAL_DIR = ROOT / "evaluation"
PROMPT_VERSION = 2
JUDGE_BLOCK = 15              # lines per judge request (small requests answer faster and time out less)
JUDGE_TIMEOUT_S = 60
MAX_CANDIDATES = 16           # measured models probed as judges or spares
# Lines judged per item. A random sample keeps a comparison to a few minutes on free providers: 80 differing lines
# per item (240 for the set) give a win rate within about +-7 % (95 %). `--all` judges every line.
SAMPLE_PAIRWISE = 80
SAMPLE_ABSOLUTE = 60
CONTEXT_LINES = 3
WINDOW_S = 15 * 60
# Settings the GUI passes to a job (app/ui/main_window.py).
EXTRAS_KEYS = ("fetch_platform_subtitles", "opensubtitles_api_key", "subdl_api_key", "llm_refine", "llm_correct_only",
               "llm_review", "translation_engine", "local_model", "audio_enhance")
FLAGGED = ("LOW", "MEDIUM")
ERROR_TYPES = ("meaning", "omission", "addition", "gender_or_addressee", "name", "register", "grammar", "fluency",
               "untranslated")


# -- results of a run ------------------------------------------------------------------------------------------

@dataclass
class Unit:
    id: int
    start: float
    end: float
    source: str
    target: str
    confidence: str = ""
    flags: list[str] = field(default_factory=list)
    engine: str = ""


@dataclass
class Result:
    name: str
    source_language: str
    target_language: str
    units: list[Unit]
    stats: dict


def load_result(path: Path, name: str | None = None) -> Result:
    """A finished episode: its output folder, its work/ folder or MasterTranscript.json itself."""
    path = Path(path)
    if path.is_dir():
        candidates = [path / "MasterTranscript.json", path / "work" / "MasterTranscript.json"]
        path = next((c for c in candidates if c.is_file()), candidates[-1])
    doc = json.loads(path.read_text(encoding="utf-8"))
    units = []
    for u in doc.get("translation_units") or []:
        target = u.get("final_text") if u.get("final_text") is not None else u.get("translation", "")
        units.append(Unit(int(u["id"]), float(u["start"]), float(u["end"]), u.get("text", ""), target or "",
                          u.get("confidence") or "", list(u.get("flags") or []) + list(u.get("qa_flags") or []),
                          u.get("engine") or ""))
    language = (doc.get("language") or {}).get("code") or ""
    target_language = (doc.get("job") or {}).get("target_language") or ""
    return Result(name or path.parent.parent.name, language, target_language, units, doc.get("stats") or {})


def run_results(label: str, eval_dir: Path | None = None) -> dict[str, Result]:
    """{set item name: Result} of a stored run."""
    eval_dir = eval_dir or EVAL_DIR
    out = eval_dir / "runs" / label / "out"
    if not out.is_dir():
        raise SystemExit(f"No run named {label!r} in {eval_dir / 'runs'}")
    results = {}
    for transcript in sorted(out.glob("*/work/MasterTranscript.json")):
        name = transcript.parent.parent.name
        results[name] = load_result(transcript, name)
    if not results:
        raise SystemExit(f"Run {label!r} has no finished items")
    return results


# -- metrics without a judge ------------------------------------------------------------------------------------

def basic_metrics(result: Result) -> dict:
    units = result.units
    n = len(units) or 1
    flags: dict[str, int] = {}
    for u in units:
        for f in set(u.flags):
            flags[f] = flags.get(f, 0) + 1
    stats = result.stats or {}
    media = float(stats.get("media_duration_s") or 0) or (max((u.end for u in units), default=0.0))
    stages = {k: v.get("elapsed_s") for k, v in (stats.get("stages") or {}).items() if isinstance(v, dict)}
    tokens = {}
    for provider, used in ((stats.get("refine") or {}).get("tokens") or {}).items():
        tokens[provider] = int(used.get("prompt_tokens", 0)) + int(used.get("completion_tokens", 0))
    return {
        "lines": len(units),
        "flagged_share": round(sum(u.confidence in FLAGGED for u in units) / n, 4),
        "low_share": round(sum(u.confidence == "LOW" for u in units) / n, 4),
        "flag_counts": dict(sorted(flags.items(), key=lambda kv: -kv[1])),
        "name_mismatch_share": round(flags.get("name_mismatch", 0) / n, 4),
        "media_minutes": round(media / 60, 2),
        "seconds_per_media_minute": (round(float(stats["total_elapsed_s"]) / (media / 60), 1)
                                     if stats.get("total_elapsed_s") and media else None),
        "stage_seconds": stages,
        "tokens": tokens,
    }


# -- text normalisation, alignment, reference metrics ----------------------------------------------------------

def normalize(text: str, language: str = "") -> str:
    text = unicodedata.normalize("NFC", text.replace("‏", "").replace("‎", ""))
    if language.split("-")[0] in ("tr", "az"):
        text = text.replace("I", "ı").replace("İ", "i")
    text = text.casefold()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(text.split())


def edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def wer(hypothesis: str, reference: str, language: str = "") -> tuple[int, int]:
    """(word errors, reference words)."""
    ref = normalize(reference, language).split()
    return edit_distance(normalize(hypothesis, language).split(), ref), len(ref)


def cer(hypothesis: str, reference: str, language: str = "") -> tuple[int, int]:
    ref = normalize(reference, language).replace(" ", "")
    return edit_distance(list(normalize(hypothesis, language).replace(" ", "")), list(ref)), len(ref)


def chrf(hypotheses: list[str], references: list[str], order: int = 6, beta: float = 2.0) -> float:
    """Corpus chrF (character n-grams 1..6, beta 2, spaces removed; Popovic 2015), 0..100."""
    precisions, recalls = [], []
    for n in range(1, order + 1):
        match = hyp_total = ref_total = 0
        for hyp, ref in zip(hypotheses, references):
            h, r = _ngrams(hyp, n), _ngrams(ref, n)
            match += sum(min(c, r.get(g, 0)) for g, c in h.items())
            hyp_total += sum(h.values())
            ref_total += sum(r.values())
        if hyp_total and ref_total:
            precisions.append(match / hyp_total)
            recalls.append(match / ref_total)
    if not precisions:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    return round(100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r), 2)


def _ngrams(text: str, n: int) -> dict[str, int]:
    text = "".join(text.split())
    grams: dict[str, int] = {}
    for i in range(len(text) - n + 1):
        grams[text[i:i + n]] = grams.get(text[i:i + n], 0) + 1
    return grams


def align(a: list, b: list) -> list[tuple[object | None, object | None]]:
    """Pairs of items with .start/.end by largest time overlap (one-to-one, in time order); unmatched items are
    paired with None (omissions / insertions)."""
    pairs, used = [], set()
    j0 = 0
    for x in a:
        best, best_overlap = None, 0.0
        for j in range(j0, len(b)):
            y = b[j]
            if y.start >= x.end:
                break
            overlap = min(x.end, y.end) - max(x.start, y.start)
            if overlap > best_overlap and j not in used:
                best, best_overlap = j, overlap
        if best is None:
            pairs.append((x, None))
        else:
            used.add(best)
            pairs.append((x, b[best]))
            while j0 < len(b) and b[j0].end <= x.start:
                j0 += 1
    pairs += [(None, y) for j, y in enumerate(b) if j not in used]
    pairs.sort(key=lambda p: (p[0] or p[1]).start)
    return pairs


def wilson(wins: int, losses: int, z: float = 1.96) -> tuple[float, float, float]:
    """Win rate over decided comparisons and its 95 % Wilson interval."""
    n = wins + losses
    if not n:
        return 0.5, 0.0, 1.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(p, 4), round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


# -- AI judges --------------------------------------------------------------------------------------------------

PAIRWISE_SYSTEM = """You are an expert subtitle quality judge for {src} -> {tgt}.
For each item, compare the two candidate {tgt} subtitles "A" and "B" for the {src} "source" line (with context).
Judge in this order: faithful meaning (nothing lost, added or changed), correct gender/addressee/number, correct
names, natural and idiomatic {tgt}, appropriate register, concise subtitle style. Ignore punctuation and spacing
differences. Answer "tie" when both are equally good or equally bad. Do not prefer a candidate for its position.
Return one entry for EVERY item id.
Reply with JSON only: {{"results": [{{"id": <id>, "winner": "A"|"B"|"tie", "reason": "<short English>"}}]}}"""

ASR_SYSTEM = """You compare two speech-recognition transcripts of the same {src} audio line, without the audio.
For each item, decide which transcript ("A" or "B") is more plausible as what was actually said, using grammar,
meaning and the context lines. Answer "tie" when you cannot tell. Return one entry for EVERY item id.
Reply with JSON only: {{"results": [{{"id": <id>, "winner": "A"|"B"|"tie", "reason": "<short English>"}}]}}"""

ABSOLUTE_SYSTEM = """You are an expert {src} -> {tgt} subtitle reviewer. For each item, check the {tgt} "subtitle"
against the {src} "source" line, using the context lines. List only real errors. Types: meaning, omission, addition,
gender_or_addressee, name, register, grammar, fluency, untranslated. Severity "major" = a viewer would misunderstand
or notice a clear mistake; "minor" = small awkwardness. A different but correct wording is not an error.
Return one entry for EVERY item id; use "errors": [] for a correct subtitle.
Reply with JSON only: {{"results": [{{"id": <id>, "errors": [{{"type": "<type>", "severity": "major"|"minor"}}]}}]}}"""


class JudgeCache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                    self.data[record["key"]] = record["reply"]
                except (ValueError, KeyError):
                    continue

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, reply: dict) -> None:
        with self.lock:
            self.data[key] = reply
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "reply": reply}, ensure_ascii=False) + "\n")


def _responds(route, timeout_s: float = 30.0) -> bool:
    """A tiny request first: providers that are down or very slow (seen: NVIDIA free endpoints timing out) are
    not used as judges, so a comparison is not held up by timeouts."""
    started = time.monotonic()
    saved = route.client.timeout_s
    route.client.timeout_s = timeout_s
    try:
        route.client.chat(route.model, [{"role": "user", "content": 'Reply with the JSON {"ok": true}'}],
                          max_tokens=200, temperature=0.0)
        print(f"  judge candidate {route.key}: answered in {time.monotonic() - started:.0f} s", flush=True)
        return True
    except Exception as exc:
        print(f"  judge candidate {route.key}: skipped ({str(exc)[:80]})", flush=True)
        return False
    finally:
        route.client.timeout_s = saved


class Panel:
    """Judge models from different providers, best first. Every responding measured model (score >= floor) of
    every provider is a spare: a judge that fails is replaced at once and the block is asked again."""

    def __init__(self, routes: list, cache: JudgeCache, spares: list | None = None):
        self.routes = routes
        self.cache = cache
        self.spares = list(spares or [])
        self.replaced: dict[str, object] = {}
        self.lock = threading.Lock()
        self._start = (list(routes), list(self.spares))

    def reset(self) -> None:
        """Back to the starting judges (called per evaluation item), so a judge that failed once for a rate
        limit is not lost for the whole comparison and every item starts with the same panel."""
        with self.lock:
            self.routes[:] = self._start[0]
            self.spares = list(self._start[1])
            self.replaced = {}

    def _replace(self, route):
        """Swap a failed judge for the best spare, preferring a provider not already on the panel. Returns the
        judge to ask now, or None when no spare is left."""
        with self.lock:
            while route not in self.routes and route.key in self.replaced:
                route = self.replaced[route.key]        # another thread already replaced it
                if route in self.routes:
                    return route
            if route not in self.routes or not self.spares:
                return None
            busy = {r.provider for r in self.routes if r is not route}
            spare = next((r for r in self.spares if r.provider not in busy), self.spares[0])
            self.spares.remove(spare)
            self.routes[self.routes.index(route)] = spare
            self.replaced[route.key] = spare
            print(f"  judge {route.key} failed; replaced by {spare.key}", flush=True)
            return spare

    @classmethod
    def from_app(cls, exclude_providers: set[str], cache: JudgeCache, size: int = 3) -> "Panel":
        from app.core.llm_providers import ModelRanking, ProviderCallError, ProviderPool, build_clients
        from app.services.api_keys import load_keys
        from app.utils.paths import model_ranking_files

        clients = []
        for client in build_clients(load_keys()):
            try:
                if client.discover_models():
                    clients.append(client)
            except ProviderCallError as exc:
                log.warning("Provider %s unusable: %s", client.name, exc)
        pool = ProviderPool(clients, ModelRanking.load(*model_ranking_files()))
        # All measured models above the floor, best first (not only one per provider).
        candidates = [r for r in pool.routes() if r.measured and r.score >= pool.ranking.floor][:MAX_CANDIDATES]
        # Probe all candidates at once: a silent model costs one probe timeout in total.
        routes = [r for r, ok in zip(candidates, _parallel(_responds, candidates)) if ok]
        for route in routes:
            route.client.timeout_s = JUDGE_TIMEOUT_S
        # Judges: one model per provider; providers that did not translate the runs first.
        chosen, used = [], set()
        for route in sorted(routes, key=lambda r: r.provider in exclude_providers):
            if route.provider not in used and len(chosen) < size:
                chosen.append(route)
                used.add(route.provider)
        if len(chosen) < 2:
            raise SystemExit("At least two responding AI providers above the quality floor are needed for judging")
        spares = [r for r in routes if r not in chosen]
        log.info("Judges: %s (%d spare models)", ", ".join(r.key for r in chosen), len(spares))
        return cls(chosen, cache, spares)

    def ask(self, route, system: str, payload: dict) -> dict:
        """One block to one judge. On failure the judge is replaced by a spare and the block asked again."""
        from app.core.llm_translation import extract_json

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        retried_same = False
        while route is not None:
            key = JudgeCache.key(route.key, PROMPT_VERSION, system, payload)
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            started = time.monotonic()
            try:
                reply = extract_json(route.client.chat(route.model, messages, temperature=0.0, json_mode=True))
                self.cache.put(key, reply)
                print(f"  judge {route.key}: {len(payload.get('items', []))} lines in "
                      f"{time.monotonic() - started:.0f} s", flush=True)
                return reply
            except Exception as exc:              # provider error or unusable JSON: next model
                print(f"  judge {route.key} failed ({str(exc)[:100]})", flush=True)
                replacement = self._replace(route)
                if replacement is None and not retried_same:
                    # No spare left: one more try with the same judge after a short wait.
                    retried_same = True
                    time.sleep(min(30.0, float(getattr(exc, "retry_after", None) or 10)))
                    continue
                route = replacement
        log.warning("No judge answered a block")
        return {}


def _context(units: list[Unit], index: int) -> tuple[list[str], list[str]]:
    return ([u.source for u in units[max(0, index - CONTEXT_LINES):index]],
            [u.source for u in units[index + 1:index + 1 + CONTEXT_LINES]])


def _results_by_id(reply: dict) -> dict[int, dict]:
    out = {}
    if isinstance(reply, dict) and "results" not in reply and "id" in reply:
        reply = {"results": [reply]}           # a one-item reply without the wrapper
    elif isinstance(reply, list):
        reply = {"results": reply}
    for item in reply.get("results") or [] if isinstance(reply, dict) else []:
        try:
            out[int(item["id"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    return out


def pairwise(a: Result, b: Result, panel: Panel, seed: int = 7) -> dict:
    """Translation and transcript comparison of two runs of the same item; winner counts are from B's view
    (a "win" means B, the new run, is better)."""
    rng = random.Random(f"{seed}:{a.name}")
    pairs = [(x, y) for x, y in align(a.units, b.units) if x is not None and y is not None]
    translation_items, asr_items = [], []
    for k, (x, y) in enumerate(pairs):
        index = a.units.index(x)
        before, after = _context(a.units, index)
        if normalize(x.target) != normalize(y.target):
            swap = rng.random() < 0.5
            translation_items.append((k, swap, {"id": k, "previous": before, "source": x.source, "next": after,
                                                "A": y.target if swap else x.target,
                                                "B": x.target if swap else y.target}))
        if normalize(x.source, a.source_language) != normalize(y.source, a.source_language):
            swap = rng.random() < 0.5
            asr_items.append((k, swap, {"id": k, "previous": before, "next": after,
                                        "A": y.source if swap else x.source, "B": x.source if swap else y.source}))
    sampler = random.Random(f"{seed}:sample:{a.name}")
    sampled = {}
    for kind, items in (("translation", translation_items), ("transcript", asr_items)):
        sampled[kind] = len(items)
        if SAMPLE_PAIRWISE and len(items) > SAMPLE_PAIRWISE:
            items[:] = sorted(sampler.sample(items, SAMPLE_PAIRWISE), key=lambda item: item[0])
    src, tgt = _language_name(a.source_language), _language_name(a.target_language)
    report = {"aligned_lines": len(pairs), "differing_lines": sampled, "unmatched": sum(1 for x, y in align(a.units, b.units) if x is None or y is None)}
    for kind, items, system in (("translation", translation_items, PAIRWISE_SYSTEM.format(src=src, tgt=tgt)),
                                ("transcript", asr_items, ASR_SYSTEM.format(src=src))):
        votes: dict[int, list[str]] = {}
        reasons: dict[int, list[str]] = {}
        blocks = [items[s:s + JUDGE_BLOCK] for s in range(0, len(items), JUDGE_BLOCK)]
        for number, block in enumerate(blocks, 1):
            print(f"[{a.name}] {kind}: block {number}/{len(blocks)}", flush=True)
            payload = {"items": [item for _, _, item in block]}
            # The judges are different providers: ask them at the same time.
            replies = _parallel(lambda route: panel.ask(route, system, payload), panel.routes)
            for reply in replies:
                answers = _results_by_id(reply)
                for k, swap, _ in block:
                    if k not in answers:
                        continue                  # no answer from this judge: no vote (not a tie)
                    winner = str(answers[k].get("winner", "")).strip().upper()
                    if winner in ("A", "B"):
                        new_wins = (winner == "A") if swap else (winner == "B")
                        votes.setdefault(k, []).append("new" if new_wins else "old")
                    else:
                        votes.setdefault(k, []).append("tie")
                    reasons.setdefault(k, []).append(str((answers.get(k) or {}).get("reason", ""))[:160])
        wins = losses = ties = 0
        agreement = []
        disputed = []
        unjudged = 0
        for k, _, item in items:
            v = votes.get(k, [])
            if not v:
                unjudged += 1                     # no judge answered for this line
                continue
            new, old = v.count("new"), v.count("old")
            agreement.append(max(new, old, v.count("tie")) / len(v) if v else 0)
            if new > old and new > len(v) / 2:
                wins += 1
            elif old > new and old > len(v) / 2:
                losses += 1
            else:
                ties += 1
            if new and old:
                x, y = pairs[k]
                disputed.append({"source": x.source, "old": x.target if kind == "translation" else x.source,
                                 "new": y.target if kind == "translation" else y.source, "votes": v,
                                 "reasons": reasons.get(k, [])})
        rate, low, high = wilson(wins, losses)
        report[kind] = {"compared": len(items) - unjudged, "unjudged": unjudged, "new_better": wins, "old_better": losses, "ties": ties,
                        "win_rate": rate, "ci95": [low, high],
                        "judge_agreement": round(sum(agreement) / len(agreement), 3) if agreement else None,
                        "disputed": disputed[:20]}
    report["judges"] = [r.key for r in panel.routes]
    return report


def _parallel(fn, items: list, workers: int | None = None) -> list:
    """executor.map that does not wait for running requests when interrupted (Ctrl+C)."""
    executor = concurrent.futures.ThreadPoolExecutor(max(1, workers or len(items)))
    try:
        return list(executor.map(fn, items))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _absolute_tasks(result: Result, panel: Panel) -> tuple[list, list]:
    """Errors per line. Every block goes to every judge; a line has a major error when more than half of the
    judges that answered for it say so (one strict or lenient judge cannot move the rate alone)."""
    system = ABSOLUTE_SYSTEM.format(src=_language_name(result.source_language),
                                    tgt=_language_name(result.target_language))
    items = []
    for i, u in enumerate(result.units):
        if u.target.strip():
            before, after = _context(result.units, i)
            items.append({"id": u.id, "previous": before, "source": u.source, "next": after, "subtitle": u.target})
    if SAMPLE_ABSOLUTE and len(items) > SAMPLE_ABSOLUTE:
        # Same seed per item name: the old and the new run of an item are checked at the same positions.
        positions = sorted(random.Random(f"absolute:{result.name}").sample(range(len(items)), SAMPLE_ABSOLUTE))
        items = [items[i] for i in positions]
    blocks = [items[s:s + JUDGE_BLOCK] for s in range(0, len(items), JUDGE_BLOCK)]
    print(f"[{result.name}] absolute check: {len(blocks)} blocks x {len(panel.routes)} judges", flush=True)
    tasks = [(judge, system, block) for block in blocks for judge in range(len(panel.routes))]
    return items, tasks


def absolute(result: Result, panel: Panel) -> dict:
    return absolute_many([result], panel)[0]


def absolute_many(results: list[Result], panel: Panel) -> list[dict]:
    """Absolute checks of several runs in one batch: the old and the new run of an item are judged at the same
    time, by the same panel."""
    prepared = [_absolute_tasks(r, panel) for r in results]
    tasks = [t for _, ts in prepared for t in ts]
    replies = _parallel(lambda t: panel.ask(panel.routes[t[0]], t[1], {"items": t[2]}), tasks, len(panel.routes))
    out, start = [], 0
    for result, (items, ts) in zip(results, prepared):
        out.append(_absolute_score(result, items, ts, replies[start:start + len(ts)], panel))
        start += len(ts)
    return out


def _absolute_score(result: Result, items: list, tasks: list, replies: list, panel: Panel) -> dict:
    answers_by_line: dict[int, list[list[dict]]] = {}
    for (_, _, block), reply in zip(tasks, replies):
        answers = _results_by_id(reply)
        for item in block:
            if item["id"] in answers:
                errors = [e for e in answers[item["id"]].get("errors") or [] if isinstance(e, dict)]
                answers_by_line.setdefault(item["id"], []).append(errors)
    major_ids = set()
    minor = 0.0
    types: dict[str, float] = {}
    for line_id, per_judge in answers_by_line.items():
        if sum(any(e.get("severity") == "major" for e in errors) for errors in per_judge) > len(per_judge) / 2:
            major_ids.add(line_id)
        for errors in per_judge:
            minor += sum(e.get("severity") == "minor" for e in errors) / len(per_judge)
            for e in errors:
                if e.get("type") in ERROR_TYPES:
                    types[e["type"]] = types.get(e["type"], 0) + 1 / len(per_judge)
    judged_ids = set(answers_by_line)
    judged = len(judged_ids)
    flagged = {u.id for u in result.units if u.confidence in FLAGGED}
    precision = (round(len(flagged & major_ids) / len(flagged & judged_ids), 4) if flagged & judged_ids else None)
    recall = round(len(flagged & major_ids) / len(major_ids), 4) if major_ids else None
    return {"judge": ", ".join(r.key for r in panel.routes), "sampled_lines": len(items), "judged_lines": judged,
            "judges_per_line": round(sum(map(len, answers_by_line.values())) / judged, 2) if judged else None,
            "major_error_rate": round(len(major_ids) / judged, 4) if judged else None,
            "minor_errors_per_line": round(minor / judged, 4) if judged else None,
            "error_types": {k: round(v, 1) for k, v in sorted(types.items(), key=lambda kv: -kv[1])},
            "flag_precision": precision, "flag_recall": recall}


def _language_name(code: str) -> str:
    try:
        from app.utils.languages import english_name
        return english_name(code) or code
    except Exception:
        return code


def _translators(result: Result) -> dict[str, int]:
    """Lines per translating model (engine) in a run item."""
    counts: dict[str, int] = {}
    for u in result.units:
        if ":" in u.engine:
            counts[u.engine] = counts.get(u.engine, 0) + 1
    return counts


def translator_providers(results: list[Result]) -> set[str]:
    return {u.engine.split(":")[0] for r in results for u in r.units if ":" in u.engine}


# -- human references (optional calibration) ------------------------------------------------------------------

@dataclass
class RefLine:
    start: float
    end: float
    source: str
    target: str


def load_reference(folder: Path) -> list[RefLine]:
    import csv

    with open(Path(folder) / "lines.csv", encoding="utf-8-sig", newline="") as f:
        return [RefLine(float(r["start"]), float(r["end"]), r.get("source_ref") or "", r.get("target_ref") or "")
                for r in csv.DictReader(f)]


def reference_metrics(result: Result, reference: list[RefLine]) -> dict:
    lo = min((r.start for r in reference), default=0.0)
    hi = max((r.end for r in reference), default=0.0)
    units = [u for u in result.units if u.end > lo and u.start < hi]
    pairs = align(reference, units)
    word_err = words = char_err = chars = 0
    hyps, refs = [], []
    for ref, unit in pairs:
        hyp_source = unit.source if unit else ""
        hyp_target = unit.target if unit else ""
        if ref is None:
            continue
        e, n = wer(hyp_source, ref.source, result.source_language)
        word_err, words = word_err + e, words + n
        e, n = cer(hyp_source, ref.source, result.source_language)
        char_err, chars = char_err + e, chars + n
        hyps.append(normalize(hyp_target))
        refs.append(normalize(ref.target))
    return {"lines": len([p for p in pairs if p[0] is not None]),
            "omissions": sum(1 for r, u in pairs if r is not None and u is None),
            "insertions": sum(1 for r, u in pairs if r is None),
            "wer": round(word_err / words, 4) if words else None,
            "cer": round(char_err / chars, 4) if chars else None,
            "chrf": chrf(hyps, refs)}


# -- evaluation set -------------------------------------------------------------------------------------------

def _job_media(job_dir: Path) -> Path | None:
    stage = job_dir / "stages" / "download.json"
    if stage.is_file():
        media = (json.loads(stage.read_text(encoding="utf-8")).get("info") or {}).get("media")
        if media and Path(media).is_file():
            return Path(media)
    log_file = job_dir / "job.log"
    if log_file.is_file():
        # repr() quotes with " when the path contains an apostrophe ("Konseyi'ne").
        match = re.search(r"input_path=(?:Windows|Posix)Path\((['\"])(.+?)\1\)",
                          log_file.read_text(encoding="utf-8", errors="replace"))
        if match:
            path = Path(match.group(2))
            if path.is_file():
                return path
    return None


def _windows(segments: list[dict], duration: float) -> list[dict]:
    """Candidate 15-minute windows (whole file when shorter) with their selection measures."""
    if duration <= WINDOW_S:
        starts = [0.0]
    else:
        starts = [float(s) for s in range(0, int(duration - WINDOW_S) + 1, 60)]
    out = []
    for start in starts:
        end = min(duration, start + WINDOW_S)
        inside = [s for s in segments if s["start"] >= start and s["end"] <= end]
        if not inside:
            continue
        speech = sum(s["end"] - s["start"] for s in inside) or 1.0
        confidence = [s["audio_confidence"] for s in inside if s.get("audio_confidence") is not None]
        out.append({"from": start, "to": end, "segments": len(inside),
                    "audio_confidence": sum(confidence) / len(confidence) if confidence else 1.0,
                    "words_per_second": sum(len(s["text"].split()) for s in inside) / speech})
    return out


def make_set(jobs_dir: Path) -> list[dict]:
    """Three items from three different finished jobs: dialogue-heavy, noisiest, fastest speech."""
    candidates = []
    for job in sorted(p for p in Path(jobs_dir).iterdir() if p.is_dir()):
        masters = sorted(job.glob("master.*.json"), key=lambda p: p.stat().st_mtime)
        media = _job_media(job)
        if not masters or media is None or not list(job.glob("refined.*.json")):
            continue
        doc = json.loads(masters[-1].read_text(encoding="utf-8"))
        series = {}
        exported = media.parent / "work" / "MasterTranscript.json"
        if exported.is_file():
            stored = json.loads(exported.read_text(encoding="utf-8")).get("series") or {}
            series = {k: stored.get(k) for k in ("series_name", "season", "episode") if stored.get(k) is not None}
        duration = float((doc.get("media") or {}).get("duration") or 0)
        for w in _windows(doc.get("segments") or [], duration):
            candidates.append({**w, "job": job.name, "media": str(media), "series": series,
                               "source_language": (doc.get("language") or {}).get("code"),
                               "target_language": (doc.get("job") or {}).get("target_language")})
    chosen, used = [], set()
    for kind, key in (("dialogue", lambda c: c["segments"]), ("noise", lambda c: -c["audio_confidence"]),
                      ("fast_speech", lambda c: c["words_per_second"])):
        # Different videos for the three items (the same video can have several job folders).
        pool = [c for c in candidates if c["media"] not in used] or candidates
        if not pool:
            break
        best = max(pool, key=key)
        used.add(best["media"])
        chosen.append({"name": f"{kind}_{best['job'][:8]}", "kind": kind, "job": best["job"], "media": best["media"],
                       "from": best["from"], "to": best["to"], "source_language": best["source_language"],
                       "target_language": best["target_language"], "series": best["series"],
                       "measures": {k: round(best[k], 3) for k in ("segments", "audio_confidence", "words_per_second")}})
    return chosen


def cut_audio(source: Path, start: float, end: float, target: Path) -> Path:
    """16 kHz mono WAV of [start, end) of any media file (PyAV)."""
    import av
    import numpy as np

    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part.wav")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    with av.open(str(source)) as container, wave.open(str(tmp), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        stream = container.streams.audio[0]
        container.seek(int(max(0.0, start - 1.0) * 1_000_000), any_frame=False)
        written_until = start
        for frame in container.decode(stream):
            if frame.time is None or frame.time + frame.samples / frame.sample_rate <= start:
                continue
            if frame.time >= end:
                break
            for chunk in resampler.resample(frame):
                data = chunk.to_ndarray().reshape(-1)
                t0 = chunk.time if chunk.time is not None else written_until
                first = max(0, int(round((start - t0) * 16000)))
                last = min(len(data), int(round((end - t0) * 16000)))
                if last > first:
                    out.writeframes(np.ascontiguousarray(data[first:last]).astype("<i2").tobytes())
                    written_until = t0 + last / 16000
    tmp.replace(target)
    return target


def _app_settings() -> dict:
    from app.database.database import Database
    from app.database.settings import Settings
    from app.utils.paths import AppPaths, default_data_root

    paths = AppPaths.from_root(default_data_root())
    with Database(paths.db_file) as db:
        return Settings(db).all()


def run_set(label: str, settings_file: Path | None, reuse_asr: str | None, eval_dir: Path | None = None) -> None:
    eval_dir = eval_dir or EVAL_DIR
    from app.core.metadata import SeriesInfo
    from app.core.modes import Mode
    from app.core.pipeline import JobConfig
    from app.services.job_manager import default_pipeline_factory
    from app.utils.paths import AppPaths, default_data_root

    items = json.loads((eval_dir / "set.json").read_text(encoding="utf-8"))
    current = _app_settings()
    override = json.loads(Path(settings_file).read_text(encoding="utf-8")) if settings_file else {}
    mode = Mode(override.get("mode") or current["mode"])
    extras = {k: current[k] for k in EXTRAS_KEYS}
    extras.update(override.get("extras") or {})
    run_dir = eval_dir / "runs" / label
    if run_dir.exists():
        raise SystemExit(f"Run {label!r} already exists; use another label")
    jobs_dir = run_dir / "jobs"
    jobs_dir.mkdir(parents=True)
    if reuse_asr:
        _copy_transcription(eval_dir / "runs" / reuse_asr / "jobs", jobs_dir)
    (run_dir / "settings.json").write_text(json.dumps({"mode": mode.value, "extras": {
        k: v for k, v in extras.items() if not k.endswith("_api_key")}, "reuse_asr": reuse_asr}, indent=2),
        encoding="utf-8")
    paths = AppPaths.from_root(default_data_root())
    factory = default_pipeline_factory(jobs_dir, paths.models_dir, paths.root)
    for item in items:
        media = cut_audio(Path(item["media"]), float(item["from"]), float(item["to"]),
                          eval_dir / "media" / f"{item['name']}.wav")
        # The series name keeps the series glossary (character names) in use, as in a real run.
        series = SeriesInfo(**item["series"], source="user") if item.get("series") else None
        config = JobConfig(media, item["source_language"] or "auto", item["target_language"], mode,
                           output_dir=run_dir / "out", series_override=series)
        started = time.monotonic()
        print(f"[{label}] {item['name']} ...", flush=True)
        pipeline = factory(config, threading.Event(),
                           lambda stage, frac, overall, msg: None, lambda *a: None, extras)
        pipeline.run()
        print(f"[{label}] {item['name']} done in {time.monotonic() - started:.0f} s", flush=True)


def _copy_transcription(source_jobs: Path, target_jobs: Path) -> None:
    """Copy audio and transcription results (not translation) so a run with the same speech settings starts at
    the translation stages."""
    keep = ("input.json", "media_info.json", "audio.wav", "language.", "master.", "transcribe.", "enhanced.")
    for job in source_jobs.iterdir():
        if not job.is_dir():
            continue
        (target_jobs / job.name / "stages").mkdir(parents=True, exist_ok=True)
        for f in job.iterdir():
            if f.is_file() and f.name.startswith(keep):
                shutil.copy2(f, target_jobs / job.name / f.name)
        for stage in ("audio", "transcribe"):
            manifest = job / "stages" / f"{stage}.json"
            if manifest.is_file():
                shutil.copy2(manifest, target_jobs / job.name / "stages" / manifest.name)


# -- reports ----------------------------------------------------------------------------------------------------

def write_report(label: str, report: dict, eval_dir: Path | None = None) -> Path:
    eval_dir = eval_dir or EVAL_DIR
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = eval_dir / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    # Not with_suffix(): a label with a dot (p2.2) would lose its end.
    json_path, md_path = folder / f"{stamp}_{label}.json", folder / f"{stamp}_{label}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(label, report), encoding="utf-8")
    with open(eval_dir / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": stamp, "label": label, "summary": _summary(report)}, ensure_ascii=False) + "\n")
    return md_path


def _summary(report: dict) -> dict:
    out = {}
    for name, item in (report.get("items") or {}).items():
        for part in ("translation", "transcript"):
            if part in (item.get("pairwise") or {}):
                out[f"{name}.{part}.win_rate"] = item["pairwise"][part]["win_rate"]
        for side in ("absolute", "absolute_old", "absolute_new"):
            if side in item:
                out[f"{name}.{side}.major_error_rate"] = item[side].get("major_error_rate")
    if "total" in report:
        out.update({f"total.{k}": v for k, v in report["total"].items() if not isinstance(v, (dict, list))})
    return out


def _markdown(label: str, report: dict) -> str:
    lines = [f"# Evaluation: {label}", "", f"Created {datetime.now():%Y-%m-%d %H:%M}", ""]
    if "total" in report:
        lines += ["## Total", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in report["total"].items() if not isinstance(v, (dict, list))]
        lines.append("")
    for name, item in (report.get("items") or {}).items():
        lines += [f"## {name}", ""]
        for part in ("translation", "transcript"):
            p = (item.get("pairwise") or {}).get(part)
            if p:
                lines.append(f"- **{part}**: new better {p['new_better']}, old better {p['old_better']}, ties "
                             f"{p['ties']} of {p['compared']} differing lines; win rate {p['win_rate']:.1%} "
                             f"(95 % CI {p['ci95'][0]:.1%}-{p['ci95'][1]:.1%}), judge agreement {p['judge_agreement']}")
        for side in ("absolute", "absolute_old", "absolute_new"):
            a = item.get(side)
            if a:
                lines.append(f"- **{side}** ({a['judge']}): major-error rate {a['major_error_rate']}, minor per line "
                             f"{a['minor_errors_per_line']}, types {a['error_types']}, flag precision "
                             f"{a['flag_precision']}, recall {a['flag_recall']}")
        for side in ("basic", "basic_old", "basic_new"):
            b = item.get(side)
            if b:
                lines.append(f"- **{side}**: lines {b['lines']}, flagged {b['flagged_share']:.1%}, name issues "
                             f"{b['name_mismatch_share']:.1%}, s per media min {b['seconds_per_media_minute']}, "
                             f"tokens {sum(b['tokens'].values())}")
        if item.get("reference"):
            lines.append(f"- **reference**: {item['reference']}")
        disputed = ((item.get("pairwise") or {}).get("translation") or {}).get("disputed") or []
        if disputed:
            lines += ["", "| source | old | new | votes |", "|---|---|---|---|"]
            lines += [f"| {d['source']} | {d['old']} | {d['new']} | {' '.join(d['votes'])} |" for d in disputed]
        lines.append("")
    return "\n".join(lines) + "\n"


def _total(items: dict) -> dict:
    total: dict = {}
    for part in ("translation", "transcript"):
        wins = sum(i.get("pairwise", {}).get(part, {}).get("new_better", 0) for i in items.values())
        losses = sum(i.get("pairwise", {}).get(part, {}).get("old_better", 0) for i in items.values())
        if wins + losses:
            rate, low, high = wilson(wins, losses)
            total[f"{part}_win_rate"], total[f"{part}_ci95_low"], total[f"{part}_ci95_high"] = rate, low, high
    for side in ("absolute", "absolute_old", "absolute_new"):
        judged = sum((i.get(side) or {}).get("judged_lines") or 0 for i in items.values())
        major = sum(((i.get(side) or {}).get("major_error_rate") or 0) * ((i.get(side) or {}).get("judged_lines") or 0)
                    for i in items.values())
        if judged:
            total[f"{side}_major_error_rate"] = round(major / judged, 4)
    return total


# -- command line -----------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("make-set")
    p.add_argument("--jobs", type=Path, help="jobs folder (default: the app's data folder)")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--label", required=True)
    p.add_argument("--settings", type=Path)
    p.add_argument("--reuse-asr", help="copy the transcription of this earlier run (same speech settings only)")
    p = sub.add_parser("score")
    p.add_argument("label")
    p = sub.add_parser("compare")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--all", action="store_true", help="judge every line instead of a random sample (slow)")
    p = sub.add_parser("reference")
    p.add_argument("folder", type=Path)
    p.add_argument("--run", required=True)
    args = parser.parse_args(argv)

    if args.command == "make-set":
        from app.utils.paths import default_data_root
        target = EVAL_DIR / "set.json"
        if target.exists() and not args.force:
            print(f"{target} exists; use --force to rebuild it")
            return 1
        items = make_set(args.jobs or default_data_root() / "jobs")
        if not items:
            print("No finished job with its media file was found")
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        for item in items:
            print(f"{item['name']}: {item['media']} {item['from']:.0f}-{item['to']:.0f} s {item['measures']}")
        return 0
    if args.command == "run":
        run_set(args.label, args.settings, args.reuse_asr)
        return 0
    cache = JudgeCache(EVAL_DIR / "judge_cache.jsonl")
    if args.command == "score":
        results = run_results(args.label)
        panel = Panel.from_app(translator_providers(list(results.values())), cache)
        items = {}
        for name, r in results.items():
            panel.reset()
            items[name] = {"basic": basic_metrics(r), "absolute": absolute(r, panel)}
        print(f"Report: {write_report(args.label, {'items': items, 'total': _total(items)})}")
        return 0
    if args.command == "compare":
        if args.all:
            global SAMPLE_PAIRWISE, SAMPLE_ABSOLUTE
            SAMPLE_PAIRWISE = SAMPLE_ABSOLUTE = 0
        old, new = run_results(args.old), run_results(args.new)
        panel = Panel.from_app(translator_providers(list(old.values()) + list(new.values())), cache)
        items = {}
        for name in sorted(set(old) & set(new)):
            panel.reset()
            absolute_old, absolute_new = absolute_many([old[name], new[name]], panel)
            mix = [_translators(old[name]), _translators(new[name])]
            if mix[0] and mix[1] and max(mix[0], key=mix[0].get) != max(mix[1], key=mix[1].get):
                print(f"WARNING [{name}]: the runs were translated by different models ({mix[0]} vs {mix[1]}); "
                      "the comparison measures the model change, not only the code change", flush=True)
            items[name] = {"translators": mix, "pairwise": pairwise(old[name], new[name], panel),
                           "absolute_old": absolute_old, "absolute_new": absolute_new,
                           "basic_old": basic_metrics(old[name]), "basic_new": basic_metrics(new[name])}
        print(f"Report: {write_report(f'{args.old}_vs_{args.new}', {'items': items, 'total': _total(items)})}")
        return 0
    if args.command == "reference":
        reference = load_reference(args.folder)
        results = run_results(args.run)
        meta = json.loads((args.folder / "meta.json").read_text(encoding="utf-8"))
        # A reference made from an evaluation run item (make_reference --transcript evaluation/runs/<label>/out/
        # <item>/work/MasterTranscript.json) has that item's cut audio as media: compare with that item only.
        item = Path(str(meta.get("media_path") or "")).stem
        items = {name: {"reference": reference_metrics(r, reference)} for name, r in results.items()
                 if item not in results or name == item}
        print(f"Report: {write_report(f'{args.run}_reference', {'items': items})}")
        return 0
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Judge answers received so far are cached; the next run reuses them.", flush=True)
        # Exit at once: do not wait for judge requests still running in worker threads.
        os._exit(130)
