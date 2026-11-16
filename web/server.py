#!/usr/bin/env python3
"""Local web server for the ORB audio demo.

Standard library only: ThreadingHTTPServer + BaseHTTPRequestHandler.
Run:  .venv-torch/bin/python web/server.py [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.engine import BadInput, Engine  # noqa: E402

STATIC_DIR = ROOT / "web" / "static"
OUT_DIR = ROOT / "web" / "out"
EXAMPLES_PATH = ROOT / "web" / "examples.json"
MAX_BODY_BYTES = 1_000_000

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".wav": "audio/wav",
}

AUDIO_NAME = re.compile(r"^[A-Za-z0-9_-]+\.wav$")
STATIC_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class Handler(BaseHTTPRequestHandler):
    server_version = "ORBWeb/1.0"
    protocol_version = "HTTP/1.1"

    engine: Engine | None = None

    # -- logging -----------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        # Silence the default per-request chatter.
        pass

    def log_request(self, code="-", size="-") -> None:
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = 0
        if status >= 400:
            sys.stderr.write(f"{self.command} {self.path} -> {code}\n")

    def log_error(self, fmt: str, *args) -> None:
        sys.stderr.write("http error: " + (fmt % args) + "\n")

    def handle_one_request(self) -> None:
        # Clients (browsers, curl) routinely drop keep-alive sockets; that is
        # not a server error and should not print a traceback.
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    # -- verbs -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        self._guard(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._guard(head=True)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._route_post()
        except BadInput as exc:
            self._json({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": str(exc) or "internal server error"}, 500)

    def _guard(self, head: bool) -> None:
        try:
            self._route_get(head)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            try:
                self._json({"error": str(exc) or "internal server error"}, 500)
            except Exception:  # noqa: BLE001
                pass

    # -- routing -----------------------------------------------------------

    def _route_get(self, head: bool) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._serve_file(STATIC_DIR / "index.html", head=head)
            return
        if path.startswith("/static/"):
            name = path[len("/static/") :]
            if not STATIC_NAME.fullmatch(name):
                self._not_found(head)
                return
            self._serve_file(STATIC_DIR / name, head=head)
            return
        if path.startswith("/audio/"):
            name = path[len("/audio/") :]
            if not AUDIO_NAME.fullmatch(name):
                self._not_found(head)
                return
            self._serve_audio(name, head=head)
            return
        if path == "/api/health":
            assert self.engine is not None
            self._json(
                {
                    "ok": True,
                    "planner": self.engine.planner_path,
                    "detail": self.engine.detail_path,
                    "detail_available": self.engine.detail_available,
                    "device": str(self.engine.device),
                },
                head=head,
            )
            return
        if path == "/api/examples":
            self._json(self._examples(), head=head)
            return
        if path == "/api/generate":
            self._json({"error": "use POST"}, 405, head=head)
            return
        self._not_found(head)

    def _route_post(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path != "/api/generate":
            self._json({"error": "not found"}, 404)
            return
        body = self._read_json()
        caption = body.get("caption")
        detail = body.get("detail", False)
        strength = body.get("strength", 1.0)

        if not isinstance(caption, str):
            raise BadInput("caption must be a string")
        if not isinstance(detail, bool):
            if isinstance(detail, (int, float)) and not isinstance(detail, bool):
                detail = bool(detail)
            else:
                raise BadInput("detail must be a boolean")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise BadInput("strength must be a number between 0 and 1")

        assert self.engine is not None
        result = self.engine.generate(caption, detail=detail, strength=float(strength))
        self._json(result, 200)

    # -- responses ---------------------------------------------------------

    def _json(self, payload: dict, status: int = 200, head: bool = False) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", head=head)

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        head: bool = False,
        extra: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if extra:
            for key, value in extra.items():
                self.send_header(key, value)
        self.end_headers()
        if not head and body:
            self.wfile.write(body)

    def _not_found(self, head: bool = False) -> None:
        body = b"not found\n"
        self._send(404, body, "text/plain; charset=utf-8", head=head)

    def _serve_file(self, path: Path, head: bool = False) -> None:
        try:
            resolved = path.resolve()
            resolved.relative_to(STATIC_DIR.resolve())
        except (OSError, ValueError):
            self._not_found(head)
            return
        if not resolved.is_file():
            self._not_found(head)
            return
        try:
            body = resolved.read_bytes()
        except OSError:
            self._not_found(head)
            return
        ctype = MIME_TYPES.get(resolved.suffix.lower(), "application/octet-stream")
        self._send(200, body, ctype, head=head)

    def _serve_audio(self, name: str, head: bool = False) -> None:
        path = (OUT_DIR / name).resolve()
        try:
            path.relative_to(OUT_DIR.resolve())
        except ValueError:
            self._not_found(head)
            return
        if not path.is_file():
            self._not_found(head)
            return

        size = path.stat().st_size
        byte_range = self._parse_range(size)
        if byte_range == "unsatisfiable":
            self._send(
                416,
                b"",
                "audio/wav",
                head=head,
                extra={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
            )
            return
        if byte_range == "invalid":
            self._send(400, b"bad range\n", "text/plain; charset=utf-8", head=head)
            return

        if byte_range is None:
            start, end = 0, size - 1
            status = 200
            extra = {"Accept-Ranges": "bytes"}
        else:
            start, end = byte_range
            status = 206
            extra = {
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{size}",
            }

        try:
            with path.open("rb") as fh:
                fh.seek(start)
                body = fh.read(max(0, end - start + 1))
        except OSError:
            self._not_found(head)
            return

        self._send(status, body, "audio/wav", head=head, extra=extra)

    def _parse_range(self, size: int) -> tuple[int, int] | None | str:
        header = self.headers.get("Range")
        if not header:
            return None
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
        if not match:
            return "invalid"
        first, last = match.group(1), match.group(2)
        if first == "" and last == "":
            return "invalid"
        if first == "":
            suffix = int(last)
            if suffix <= 0:
                return "invalid"
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(first)
            if start >= size:
                return "unsatisfiable"
            end = int(last) if last != "" else size - 1
            end = min(end, size - 1)
            if end < start:
                return "invalid"
        return start, end

    def _read_json(self) -> dict:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise BadInput("missing Content-Length")
        try:
            length = int(raw_length)
        except ValueError:
            raise BadInput("invalid Content-Length") from None
        if length <= 0:
            raise BadInput("empty request body")
        if length > MAX_BODY_BYTES:
            raise BadInput("request body too large")
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise BadInput("invalid JSON body") from None
        if not isinstance(body, dict):
            raise BadInput("JSON body must be an object")
        return body

    @staticmethod
    def _examples() -> dict:
        try:
            raw = EXAMPLES_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(data.get("examples"), list):
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return {"examples": []}


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ORB audio demo locally.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 in a container)")
    args = parser.parse_args()

    print("loading planner and detail checkpoints...", flush=True)
    started = time.perf_counter()
    engine = Engine()
    engine.warmup()
    elapsed = time.perf_counter() - started
    Handler.engine = engine

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"models ready in {elapsed:.1f}s", flush=True)
    print(f"ORB web ready — http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
