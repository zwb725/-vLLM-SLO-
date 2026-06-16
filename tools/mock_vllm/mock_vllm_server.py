#!/usr/bin/env python3
"""Local mock OpenAI-compatible SSE server.

This is not vLLM. It exists only for local proxy development without GPU,
model weights, CUDA, or Python web framework dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


class MockHandler(BaseHTTPRequestHandler):
    server_version = "mock-vllm-sse/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[mock-vllm] " + fmt % args + "\n")
        sys.stdout.flush()

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/health":
            self.send_json(404, {"error": "unsupported path"})
            return
        self.send_json(200, {"status": "ok", "mock": True, "vllm": False})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "unsupported path"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        self.log_message("received POST /v1/chat/completions bytes=%d", len(body))

        mode = env_str("MOCK_VLLM_MODE", "ok").lower()
        if mode == "http500":
            self.send_json(500, {"error": "mock http500"})
            return
        if mode == "hang":
            time.sleep(env_float("MOCK_VLLM_HANG_SECONDS", 3600.0))
            return

        first_delay = env_float("MOCK_VLLM_FIRST_TOKEN_DELAY_SECONDS", 0.05)
        token_delay = env_float("MOCK_VLLM_TOKEN_DELAY_SECONDS", 0.05)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        events = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":" from"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":" mock"}}]}\n\n',
            "data: [DONE]\n\n",
        ]

        time.sleep(first_delay)
        for index, event in enumerate(events):
            if mode == "drop" and index == 1:
                self.log_message("dropping connection by mock mode")
                self.close_connection = True
                return
            self.wfile.write(event.encode("utf-8"))
            self.wfile.flush()
            self.log_message("sent SSE event %d", index)
            time.sleep(token_delay)

    def send_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local mock vLLM SSE server.")
    parser.add_argument("--host", default=os.getenv("MOCK_VLLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MOCK_VLLM_PORT", "8000")))
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MockHandler)
    print(f"[mock-vllm] not vLLM; listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[mock-vllm] stopping", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
