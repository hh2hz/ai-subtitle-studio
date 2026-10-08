"""Imitates the llama-server command line and HTTP API for tests (no model involved)."""

import argparse
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

_MARKER = re.compile(r"^\[(\d+)\]\s*(.*)$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("-c")
    parser.add_argument("-ngl")
    parser.add_argument("-np")
    parser.add_argument("--device")
    parser.add_argument("--fit")
    parser.add_argument("--no-jinja", action="store_true")
    args, _ = parser.parse_known_args()
    if not args.no_jinja:
        print("chat template parsing error", flush=True)      # what the real TranslateGemma GGUF does
        return 1
    on_cpu = args.device == "none" and args.ngl == "0"
    if not on_cpu and args.fit != "on":
        print("GPU start without --fit on", flush=True)
        return 1
    if os.environ.get("FAKE_LLAMA_FAIL_GPU") and not on_cpu:
        print("ggml_cuda_init: failed to initialize CUDA", flush=True)
        return 1
    if os.environ.get("FAKE_LLAMA_EXPECT_PATH") and os.environ["FAKE_LLAMA_EXPECT_PATH"] not in os.environ["PATH"]:
        print("cudart64_12.dll not found", flush=True)
        return 1
    if not os.path.isfile(args.m):
        print("model not found", flush=True)
        return 1

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"status": "ok"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = body["prompt"]
            assert prompt.startswith("<start_of_turn>user\n") and prompt.endswith("<start_of_turn>model\n")
            text = prompt.split("\n\n\n", 1)[1].split("<end_of_turn>")[0]
            out = []
            for raw in text.splitlines():
                match = _MARKER.match(raw)
                out.append(f"[{match.group(1)}] L({match.group(2)})" if match else f"L({raw})")
            self._send({"content": "\n".join(out) + f"\n(ngl={args.ngl})" * 0})

    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
