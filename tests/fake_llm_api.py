"""A local HTTP server imitating an OpenAI-compatible chat API (tests only)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


NEVZAT_AR = "\u0646\u064a\u0641\u0632\u0627\u062a"      # Arabic transliteration of "Nevzat"


def fake_text(source: str) -> str:
    return f"AR({source})".replace("Nevzat", NEVZAT_AR)


class FakeLlmApi:
    """behaviour: "ok", "rate_limit", "unauthorized", "garbage", "drop_first" (omits the first line id)."""

    def __init__(self, models=("good-model",), behaviour="ok", reviewer_fix=None, retry_after="0",
                 reject_reasoning=False, pricing=None, max_tokens_cap=None, translations=None, judge_choice="A"):
        self.translations = translations or {}       # source text -> translation (default: fake_text)
        self.judge_choice = judge_choice             # verdict of the double-check judge (D-104)
        self.max_tokens_cap = max_tokens_cap
        self.reject_reasoning = reject_reasoning
        self.pricing = pricing or {}            # model id -> OpenRouter-style pricing dict
        self.models = list(models)
        self.behaviour = behaviour
        self.reviewer_fix = reviewer_fix          # id to "correct" when acting as reviewer
        self.retry_after = retry_after
        self.requests: list[dict] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, payload, status=200, headers=None):
                body = json.dumps(payload).encode()
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if "/ai/models/search" in self.path:          # Cloudflare model search
                    self._send({"success": True, "result": [{"name": m, "task": {"name": "Text Generation"}}
                                                            for m in owner.models]})
                elif self.path.endswith("/models"):
                    self._send({"data": [{"id": m, **({"pricing": owner.pricing[m]} if m in owner.pricing else {})}
                                         for m in owner.models]})
                else:
                    self._send({"error": "nf"}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append({"auth": self.headers.get("Authorization"), **body})
                if owner.behaviour == "unauthorized":
                    return self._send({"error": "bad key"}, 401)
                if owner.max_tokens_cap and body.get("max_tokens", 0) > owner.max_tokens_cap:
                    return self._send({"error": {"message": "`max_tokens` must be less than or equal to 4096"}}, 400)
                if owner.reject_reasoning and "reasoning_effort" in body:
                    return self._send({"error": {"message": "Unknown name \"reasoning_effort\""}}, 400)
                if owner.behaviour == "rate_limit":
                    return self._send({"error": "slow down"}, 429, {"Retry-After": owner.retry_after})
                content = owner.reply(body)
                self._send({"choices": [{"message": {"role": "assistant", "content": content}}],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def reply(self, body: dict) -> str:
        if self.behaviour == "garbage":
            return "Sorry, I cannot help."
        if len(body["messages"]) < 2:                       # availability probe (tools/list_models.py)
            return "OK"
        if "translator's brief" in body["messages"][0]["content"]:                 # episode brief (D-055)
            return json.dumps({"summary": "A test episode.", "characters": [
                {"name": "Nevzat", "gender": "male", "role": "officer", "aliases": []}], "relations": [], "terms": []})
        if "judging two candidate translations" in body["messages"][0]["content"]:   # double-check judge (D-104)
            items = json.loads(body["messages"][1]["content"])["lines"]
            return json.dumps({"verdicts": [{"id": l["id"], "choice": self.judge_choice, "reason": "test reason"}
                                            for l in items]})
        if "match the speakers" in body["messages"][0]["content"]:                 # speaker map (D-099)
            voices = json.loads(body["messages"][1]["content"])["speakers"]
            return json.dumps({"speakers": [{"id": v["id"], "name": "Nevzat" if v["id"] == "S1" else "unknown",
                                             "gender": "male" if v["id"] == "S1" else "female"} for v in voices]})
        if "small local translation model" in body["messages"][0]["content"]:      # correct mode
            fixes = [{"id": self.reviewer_fix, "text": "FIXED"}] if self.reviewer_fix is not None else []
            return json.dumps({"corrections": fixes, "names": [{"source": "Nevzat", "target": NEVZAT_AR}]})
        if "corrections" in body["messages"][0]["content"]:
            fixes = [{"id": self.reviewer_fix, "text": "FIXED", "reason": "meaning"}] if self.reviewer_fix else []
            return "<think>checking</think>```json\n" + json.dumps({"corrections": fixes}) + "\n```"
        lines = json.loads(body["messages"][1]["content"])["lines"]
        if self.behaviour == "drop_first":
            lines = lines[1:]
        return json.dumps({"lines": [{"id": l["id"], "text": self.translations.get(l["source"], fake_text(l["source"]))}
                                     for l in lines],
                           "names": [{"source": "Nevzat", "target": NEVZAT_AR}]})

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
