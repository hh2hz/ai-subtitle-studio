"""Benchmark every available cloud AI model on the same subtitle block ("translate" or "correct" mode).

    --mode translate (default): each model translates the Turkish lines from scratch.
    --mode correct: each model corrects the local model's translation.

Each model receives the same Turkish lines with the local model's translation (taken from a finished job's
MasterTranscript.json) and returns its corrections. The script records the final lines, which lines were changed,
tokens and time per model, so the models can be scored and ranked (app/resources/model_ranking.json, D-034).
One request per model; models that are rate limited or unavailable are reported and skipped, models excluded
by tools/list_models.py (not free / unavailable for this key) are not asked. Run list_models.py first.

    .venv\\Scripts\\python.exe tools\\benchmark_models.py
    .venv\\Scripts\\python.exe tools\\benchmark_models.py --lines 25 --start 0 --transcript "output\\...\\MasterTranscript.json"

Writes tools/benchmark_report.json and tools/benchmark_report.txt (no API keys are written).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.llm_providers import ModelRanking, ProviderCallError, ProviderPool, build_clients, heuristic_score  # noqa: E402
from app.core.llm_translation import LlmRefiner  # noqa: E402
from app.services.api_keys import load_keys  # noqa: E402
from app.utils.paths import model_ranking_files  # noqa: E402

REPORT_JSON = ROOT / "tools" / "benchmark_report.json"       # the mode is added to the name
REPORT_TXT = ROOT / "tools" / "benchmark_report.txt"
PAUSE_S = 3.0          # between requests to the same provider (per-minute limits)


def newest_transcript() -> Path:
    found = sorted((ROOT / "output").rglob("MasterTranscript.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not found:
        raise SystemExit("No MasterTranscript.json under output/ - run a job first or pass --transcript")
    return found[0]


def load_block(path: Path, start: int, count: int) -> tuple[list[dict], dict, str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    units = doc["translation_units"][start:start + count]
    lines = []
    for unit in units:
        draft = unit.get("draft") or unit.get("translation") or ""
        lines.append({"id": unit["id"], "source": unit["text"], "draft": draft})
    if not any(line["draft"] for line in lines):
        raise SystemExit("The transcript has no local translations (draft) for these lines")
    title = (doc.get("media_info") or {}).get("title") or path.parent.parent.name
    return lines, doc.get("series") or {}, title


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--lines", type=int, default=25)
    parser.add_argument("--providers", help="comma-separated provider names (default: all with keys)")
    parser.add_argument("--mode", choices=("translate", "correct"), default="translate")
    args = parser.parse_args()

    transcript = args.transcript or newest_transcript()
    lines, series, title = load_block(transcript, args.start, args.lines)
    media = {"title": title, **{k: v for k, v in series.items() if v and k != "source"}}
    print(f"{len(lines)} lines from {transcript}")

    clients = build_clients(load_keys())
    if args.providers:
        wanted = set(args.providers.split(","))
        clients = [c for c in clients if c.name in wanted]
    excluded = ModelRanking.load(*model_ranking_files()).excluded     # set by tools/list_models.py
    results = []
    for client in clients:
        try:
            models = list(client.discover_models())
        except ProviderCallError as exc:
            print(f"{client.name}: unusable ({exc})")
            results.append({"provider": client.name, "error": str(exc)})
            continue
        models = [m for m in models if f"{client.name}:{m}" not in excluded]
        print(f"\n{client.name}: {len(models)} models")
        for model in models:
            client.models = [model]
            pool = ProviderPool([client], ModelRanking(floor=0))
            refiner = LlmRefiner(pool, "tr", "ar", media, mode=args.mode, review=False)
            before = dict(client.usage_by_model.get(model, {}))
            started = time.monotonic()
            result = refiner.translate_block(lines, [], [], {})
            elapsed = time.monotonic() - started
            used = client.usage_by_model.get(model, {})
            tokens = {k: used.get(k, 0) - before.get(k, 0) for k in ("requests", "prompt_tokens", "completion_tokens")}
            entry = {"key": f"{client.name}:{model}", "provider": client.name, "model": model,
                     "heuristic": heuristic_score(client.spec, model), "elapsed_s": round(elapsed, 1),
                     "tokens": tokens, "ok": bool(result.translator),
                     "changed": sorted(result.changed or ()) if args.mode == "correct" else [], "final": {str(k): v for k, v in result.translations.items()},
                     "names": result.names, "notes": result.notes}
            results.append(entry)
            status = (f"{len(entry['changed'])} corrected, {tokens['prompt_tokens']}+{tokens['completion_tokens']} "
                      f"tokens, {elapsed:.1f} s") if entry["ok"] else "FAILED: " + "; ".join(result.notes)[:160]
            print(f"  {model}: {status}")
            time.sleep(PAUSE_S)
        client.models = models

    report_json = REPORT_JSON.with_name(f"{REPORT_JSON.stem}_{args.mode}.json")
    report_txt = REPORT_TXT.with_name(f"{REPORT_TXT.stem}_{args.mode}.txt")
    report_json.write_text(json.dumps({"transcript": str(transcript), "mode": args.mode, "lines": lines, "results": results},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
    ok = [r for r in results if r.get("ok")]
    out = [f"{len(ok)} of {len([r for r in results if 'key' in r])} models answered\n"]
    for r in ok:
        out.append(f"{r['key']}: {len(r['changed'])} corrected, {r['tokens']['prompt_tokens']} in + "
                   f"{r['tokens']['completion_tokens']} out, {r['elapsed_s']} s")
    for line in lines:
        out.append(f"\n[{line['id']}] {line['source']}\n  local: {line['draft']}")
        variants: dict[str, list[str]] = {}
        for r in ok:
            text = r["final"].get(str(line["id"]), "")
            variants.setdefault(text, []).append(r["key"])
        for text, keys in sorted(variants.items(), key=lambda item: -len(item[1])):
            out.append(f"  {text}\n      <- {', '.join(keys)}")
    report_txt.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nReport: {report_txt}")


if __name__ == "__main__":
    main()
