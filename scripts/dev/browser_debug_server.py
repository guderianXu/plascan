#!/usr/bin/env python3
"""Serve the loopback-only PlaScan browser debug hub and structured API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from browser_agent import capabilities
from browser_agent_protocol import filter_items, modal_view, paginate, revision_for, summary, view_items
from browser_debug_client import BridgeError, call_bridge


STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/styles.css": "styles.css",
}
MAXIMUM_POST_BYTES = 64 * 1024


class DebugHubContext:
    def __init__(
        self,
        state_file: Path,
        web_root: Path,
        socket_path: Path,
        token: str,
        novnc_url: str,
    ) -> None:
        self.state_file = state_file
        self.web_root = web_root
        self.socket_path = socket_path
        self.token = token
        self.novnc_url = novnc_url

    def bridge_call(self, method: str, parameters: dict[str, Any] | None = None) -> Any:
        return call_bridge(self.socket_path, self.token, method, parameters)

    def session(self) -> dict[str, Any]:
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        processes = {}
        for name, record in state.get("processes", {}).items():
            if isinstance(record, dict):
                processes[name] = {"pid": record.get("pid"), "log": record.get("log")}
        return {
            "schema_version": 1,
            "run_id": state.get("run_id", ""),
            "url": state.get("url", ""),
            "novnc_url": self.novnc_url,
            "log_directory": state.get("log_directory", ""),
            "run_directory": state.get("run_directory", ""),
            "fixture": state.get("fixture", ""),
            "fixture_mode": state.get("fixture_mode", ""),
            "project_path": state.get("project_path", ""),
            "project_source_path": state.get("project_source_path", ""),
            "project_is_copy": bool(state.get("project_is_copy")),
            "project_read_only": bool(state.get("project_read_only")),
            "processes": processes,
        }

    def agent_view(
        self, view: str, query: str = "", offset: int = 0, limit: int = 25
    ) -> dict[str, Any]:
        snapshot = self.bridge_call("snapshot")
        if not isinstance(snapshot, dict):
            raise BridgeError("bridge snapshot is not an object", "invalid_snapshot")
        if view == "summary":
            data = summary(snapshot)
        elif view == "modal":
            data = modal_view(snapshot)
        else:
            data = paginate(filter_items(view_items(snapshot, view), query), offset, limit)
        return {"schema_version": 1, "revision": revision_for(snapshot), "view": view, "data": data}


class DebugHubHandler(BaseHTTPRequestHandler):
    server_version = "PlaScanDebugHub/1"

    @property
    def context(self) -> DebugHubContext:
        return self.server.context  # type: ignore[attr-defined]

    def log_message(self, format_string: str, *arguments: object) -> None:
        message = format_string % arguments
        print(f"[{self.log_date_time_string()}] {self.client_address[0]} {message}", flush=True)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; frame-src http://127.0.0.1:* http://localhost:*; "
            "connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'",
        )
        super().end_headers()

    def token_is_valid(self) -> bool:
        supplied = self.headers.get("X-PlaScan-Debug-Token", "")
        if not supplied:
            supplied = parse_qs(urlparse(self.path).query).get("token", [""])[0]
        return secrets.compare_digest(supplied, self.context.token)

    def require_token(self) -> bool:
        if self.token_is_valid():
            return True
        self.send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "invalid session token"})
        return False

    def send_json(self, status: HTTPStatus, body: Any) -> None:
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_bridge_result(self, method: str, parameters: dict[str, Any] | None = None) -> None:
        try:
            result = self.context.bridge_call(method, parameters)
        except BridgeError as error:
            self.send_json(
                HTTPStatus.BAD_GATEWAY,
                {"ok": False, "error": {"code": error.code, "message": str(error)}},
            )
            return
        self.send_json(HTTPStatus.OK, {"ok": True, "result": result})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in STATIC_FILES:
            self.serve_static(STATIC_FILES[path])
            return
        if path == "/api/health":
            try:
                result = self.context.bridge_call("ping")
            except BridgeError as error:
                self.send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"ok": False, "error": {"code": error.code, "message": str(error)}},
                )
                return
            self.send_json(HTTPStatus.OK, {"ok": True, "bridge": result})
            return
        if not self.require_token():
            return
        if path == "/api/session":
            self.send_json(HTTPStatus.OK, {"ok": True, "result": self.context.session()})
        elif path == "/api/agent/capabilities":
            root = self.context.web_root.parents[2]
            self.send_json(HTTPStatus.OK, {"ok": True, "result": capabilities(root)})
        elif path.startswith("/api/agent/"):
            view = path.rsplit("/", 1)[-1]
            if view not in {"summary", "modal", "controls", "logs", "artifacts", "tasks"}:
                self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown agent view"})
                return
            query = parse_qs(parsed.query)
            try:
                result = self.context.agent_view(
                    view,
                    query.get("query", [""])[0],
                    int(query.get("offset", ["0"])[0]),
                    int(query.get("limit", ["25"])[0]),
                )
            except (BridgeError, ValueError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
                return
            self.send_json(HTTPStatus.OK, {"ok": True, "result": result})
        elif path == "/api/snapshot":
            self.send_bridge_result("snapshot")
        elif path == "/api/ui-tree":
            self.send_bridge_result("ui_tree")
        elif path == "/api/logs":
            self.send_bridge_result("logs")
        elif path == "/api/screenshot":
            self.send_bridge_result("screenshot")
        elif path == "/api/events":
            self.serve_events()
        else:
            self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not self.require_token():
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0 or length > MAXIMUM_POST_BYTES:
            self.send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid request size"})
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid JSON: {error}"})
            return
        if not isinstance(body, dict):
            self.send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "JSON object required"})
            return
        if path == "/api/interact":
            allowed = {"object_name", "operation", "value"}
            self.send_bridge_result("interact", {key: body[key] for key in allowed if key in body})
        elif path == "/api/close-dialog":
            self.send_bridge_result("close_dialog")
        else:
            self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def serve_static(self, file_name: str) -> None:
        path = self.context.web_root / file_name
        try:
            payload = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def serve_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for _ in range(120):
                snapshot = self.context.bridge_call("snapshot")
                event = {
                    "revision": revision_for(snapshot),
                    "data": {"timestamp": snapshot.get("timestamp", ""), **summary(snapshot)},
                } if isinstance(snapshot, dict) else {}
                payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                self.wfile.write(f"event: status\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(1)
        except (BridgeError, BrokenPipeError, ConnectionResetError, TimeoutError):
            return


class DebugHubServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], context: DebugHubContext) -> None:
        super().__init__(address, DebugHubHandler)
        self.context = context


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--web-root", type=Path, required=True)
    parser.add_argument("--bridge-socket", type=Path, required=True)
    parser.add_argument("--novnc-url", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    token = os.environ.get("PLASCAN_BROWSER_DEBUG_TOKEN", "")
    if not args.web_root.is_dir() or len(token) < 24:
        raise SystemExit("invalid debug hub web root or token")
    context = DebugHubContext(
        args.state_file.resolve(),
        args.web_root.resolve(),
        args.bridge_socket.resolve(),
        token,
        args.novnc_url,
    )
    server = DebugHubServer((args.host, args.port), context)
    print(f"PlaScan debug hub listening on http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
