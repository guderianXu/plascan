#!/usr/bin/env python3
"""Client helpers for PlaScan's test-only local browser debug bridge."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


MAXIMUM_RESPONSE_BYTES = 32 * 1024 * 1024


class BridgeError(RuntimeError):
    """Raised when the local bridge rejects or cannot complete a request."""

    def __init__(self, message: str, code: str = "bridge_error") -> None:
        super().__init__(message)
        self.code = code


def sanitize_runtime_state(state: dict[str, Any]) -> dict[str, Any]:
    token = str(state.get("token", ""))

    def scrub(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {name: scrub(item, name) for name, item in value.items() if name != "token"}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, str):
            cleaned = value.replace(token, "<redacted>") if token else value
            if key == "url":
                parsed = urlsplit(cleaned)
                return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
            return cleaned
        return value

    return scrub(state)


def call_bridge(
    socket_path: Path,
    token: str,
    method: str,
    parameters: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> Any:
    request = {
        "id": 1,
        "token": token,
        "method": method,
        "params": parameters or {},
    }
    payload = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(str(socket_path))
            connection.sendall(payload)
            response = bytearray()
            while b"\n" not in response:
                chunk = connection.recv(65536)
                if not chunk:
                    break
                response.extend(chunk)
                if len(response) > MAXIMUM_RESPONSE_BYTES:
                    raise BridgeError("debug bridge response exceeds 32 MiB", "response_too_large")
    except OSError as error:
        raise BridgeError(
            f"cannot contact PlaScan debug bridge at {socket_path}: {error}", "bridge_unavailable"
        ) from error

    line, _, _ = bytes(response).partition(b"\n")
    try:
        decoded = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BridgeError(f"debug bridge returned invalid JSON: {error}", "invalid_response") from error
    if not isinstance(decoded, dict):
        raise BridgeError("debug bridge returned a non-object response", "invalid_response")
    if not decoded.get("ok"):
        failure = decoded.get("error") or {}
        code = failure.get("code", "bridge_error")
        message = failure.get("message", "request failed")
        raise BridgeError(message, str(code))
    return decoded.get("result")


def load_runtime_credentials(state_path: Path) -> tuple[dict[str, Any], Path, str]:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BridgeError(
            f"PlaScan browser runtime state does not exist: {state_path}", "runtime_not_running"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise BridgeError(
            f"cannot read PlaScan browser runtime state {state_path}: {error}", "invalid_runtime_state"
        ) from error
    socket_path = Path(str(state.get("bridge_socket", "")))
    token = str(state.get("token", ""))
    if not socket_path.is_absolute() or len(token) < 24:
        raise BridgeError(
            "runtime state does not contain valid bridge credentials", "invalid_runtime_state"
        )
    return state, socket_path, token
