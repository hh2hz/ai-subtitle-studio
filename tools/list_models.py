"""List every model of every provider in api_keys, and check which ones this key can really use for free.

For each provider: all models it lists, which ones the app ignores and why (image/speech/embedding models, models
that are not free), and - with one tiny request per remaining model (a few tokens) - whether the key can use it:
    usable        answered
    rate limited  busy right now (kept; it has a quota)
    not free      no free quota for this key (Gemini "limit: 0", payment required, no balance)
    unavailable   the provider refused the model for this key (e.g. 404/400)
    error         network or server error (not excluded; try again later)
Models that are not free or unavailable are written as exclusions to the user's model ranking file
(<data folder>/model_ranking.json), so jobs do not spend a request on them; a later run that finds them usable
removes the exclusion again. Measured scores in that file are kept.

    .venv\\Scripts\\python.exe tools\\list_models.py              (list and check)
    .venv\\Scripts\\python.exe tools\\list_models.py --no-probe   (list only, no requests to the models)

Writes tools/models_report.txt and tools/models_report.json (no API keys are written).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.llm_providers import (ProviderCallError, build_clients, exclusion_reason,  # noqa: E402
                                    heuristic_score)
from app.services.api_keys import load_keys  # noqa: E402
from app.utils.paths import default_data_root  # noqa: E402

REPORT_JSON = ROOT / "tools" / "models_report.json"
REPORT_TXT = ROOT / "tools" / "models_report.txt"
PAUSE_S = 1.5
PROBE = [{"role": "user", "content": "Reply with the single word OK."}]


def classify(exc: ProviderCallError) -> str:
    if exc.not_free or exc.status == 402 or "balance" in str(exc).lower() or "payment" in str(exc).lower():
        return "not free"
    if exc.rate_limited:
        return "rate limited"
    if exc.status is None or exc.status >= 500:
        return "error"                   # network or server trouble: not a property of the model, not excluded
    return "unavailable"


def probe(client, model: str) -> tuple[str, str]:
    try:
        reply = client.chat(model, PROBE, max_tokens=16, temperature=0.0)
        return "usable", reply.strip()[:40]
    except ProviderCallError as exc:
        return classify(exc), str(exc)[:200]


def update_ranking(results: list[dict], path: Path) -> tuple[int, int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {"version": 1, "models": {}}
    models = data.setdefault("models", {})
    added = removed = 0
    for r in results:
        entry = models.get(r["key"], {})
        if r["status"] in ("not free", "unavailable"):
            if not entry.get("exclude"):
                added += 1
            models[r["key"]] = {**entry, "exclude": True, "reason": f"{r['status']}: {r['detail'][:120]}"}
        elif r["status"] in ("usable", "rate limited") and entry.get("exclude"):
            entry.pop("exclude", None)
            entry.pop("reason", None)
            removed += 1
            if entry:
                models[r["key"]] = entry
            else:
                models.pop(r["key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return added, removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--providers", help="comma-separated provider names (default: all with keys)")
    args = parser.parse_args()

    clients = build_clients(load_keys())
    if args.providers:
        clients = [c for c in clients if c.name in set(args.providers.split(","))]
    if not clients:
        raise SystemExit("No API keys found (api_keys.local.json or %LOCALAPPDATA%\\AISubtitleStudio\\api_keys.json)")
    providers, results, lines = [], [], []
    for client in clients:
        try:
            entries = client.fetch_models() if (client.spec.discover or client.models_url) else []
            listing_error = None
        except ProviderCallError as exc:
            entries, listing_error = [], str(exc)[:200]
        if not entries:
            entries = [{"id": m, "free": None} for m in client.spec.fallback_models]
        ids = list(dict.fromkeys(e["id"].removeprefix("models/") for e in entries))
        free = {e["id"].removeprefix("models/"): e["free"] for e in entries}
        ignored = {m: exclusion_reason(client.spec, m, free.get(m)) for m in ids}
        kept = sorted((m for m in ids if not ignored[m]), key=lambda m: -heuristic_score(client.spec, m))
        providers.append({"provider": client.name, "listed": len(ids), "kept": len(kept), "listing_error": listing_error,
                          "ignored": {m: r for m, r in ignored.items() if r}})
        head = f"\n=== {client.name}: {len(ids)} models listed, {len(kept)} text models"
        if listing_error:
            head += f" (model list failed: {listing_error}; using built-in names)"
        print(head, flush=True)
        lines.append(head)
        for model in kept:
            if args.no_probe:
                status, detail = "not checked", ""
            else:
                status, detail = probe(client, model)
                time.sleep(PAUSE_S)
            key = f"{client.name}:{model}"
            results.append({"key": key, "provider": client.name, "model": model, "status": status, "detail": detail,
                            "heuristic": heuristic_score(client.spec, model)})
            row = f"  {status:<12} {model}" + (f"   ({detail[:90]})" if status not in ("usable", "not checked") else "")
            print(row, flush=True)
            lines.append(row)
        for model, reason in ignored.items():
            if reason:
                lines.append(f"  ignored      {model}   ({reason})")
    usable = [r for r in results if r["status"] == "usable"]
    summary = f"\n{len(usable)} usable models: " + ", ".join(r["key"] for r in usable)
    lines.append(summary)
    print(summary)
    if not args.no_probe:
        ranking_file = default_data_root() / "model_ranking.json"
        added, removed = update_ranking(results, ranking_file)
        note = f"Exclusions in {ranking_file}: {added} added, {removed} removed"
        lines.append(note)
        print(note)
    REPORT_JSON.write_text(json.dumps({"providers": providers, "models": results}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
