#!/usr/bin/env python3
"""Run declarative semantic scenarios against PlaScan's browser debug bridge."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from browser_debug_client import BridgeError, call_bridge, load_runtime_credentials, sanitize_runtime_state
from browser_agent_protocol import condition_matches, wait_for_condition


def write_diagnostics(
    output_directory: Path,
    state: dict[str, Any],
    snapshot: dict[str, Any] | None,
    screenshot: dict[str, Any] | None,
    result: dict[str, Any],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    sanitized_state = sanitize_runtime_state(state)
    documents = {
        "result.json": result,
        "runtime.json": sanitized_state,
    }
    if snapshot is not None:
        documents["snapshot.json"] = snapshot
    for filename, payload in documents.items():
        (output_directory / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if screenshot and screenshot.get("available"):
        (output_directory / "window.png").write_bytes(
            base64.b64decode(str(screenshot.get("data_base64", "")), validate=True)
        )


def run_scenario(
    scenario: dict[str, Any],
    state: dict[str, Any],
    bridge_call: Callable[[str, dict[str, Any] | None, float], Any],
    output_directory: Path,
) -> dict[str, Any]:
    steps = scenario.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("scenario must contain a non-empty 'steps' array")
    default_timeout = float(scenario.get("default_timeout", 10.0))
    if default_timeout <= 0:
        raise ValueError("default_timeout must be positive")

    last_snapshot: dict[str, Any] | None = None
    completed: list[dict[str, Any]] = []

    def snapshot_provider() -> dict[str, Any]:
        result = bridge_call("snapshot", None, default_timeout)
        if not isinstance(result, dict):
            raise RuntimeError("snapshot response is not an object")
        return result

    try:
        for index, step in enumerate(steps):
            if not isinstance(step, dict) or len(step) != 1:
                raise ValueError(f"step {index + 1} must contain exactly one operation")
            operation, parameters = next(iter(step.items()))
            parameters = parameters or {}
            if not isinstance(parameters, dict):
                raise ValueError(f"step {index + 1} parameters must be an object")
            timeout = float(parameters.get("timeout", default_timeout))
            if operation == "wait":
                condition = {key: value for key, value in parameters.items() if key != "timeout"}
                last_snapshot = wait_for_condition(snapshot_provider, condition, timeout)
            elif operation == "assert":
                last_snapshot = snapshot_provider()
                matched, reason = condition_matches(last_snapshot, parameters)
                if not matched:
                    raise AssertionError(reason)
            elif operation == "action":
                bridge_call("interact", parameters, timeout)
            elif operation == "close_dialog":
                bridge_call("close_dialog", parameters, timeout)
            elif operation == "capture":
                last_snapshot = snapshot_provider()
                screenshot = bridge_call("screenshot", None, timeout)
                write_diagnostics(output_directory / f"step-{index + 1:02d}", state,
                                  last_snapshot, screenshot, {"ok": True, "step": index + 1})
            else:
                raise ValueError(f"unsupported scenario operation: {operation}")
            completed.append({"step": index + 1, "operation": operation, "ok": True})
    except Exception as error:
        try:
            last_snapshot = snapshot_provider()
        except Exception:
            pass
        try:
            screenshot = bridge_call("screenshot", None, default_timeout)
        except Exception:
            screenshot = None
        result = {
            "ok": False,
            "name": str(scenario.get("name", "unnamed")),
            "completed": completed,
            "failed_step": len(completed) + 1,
            "error": str(error),
        }
        write_diagnostics(output_directory, state, last_snapshot, screenshot, result)
        return result

    result = {
        "ok": True,
        "name": str(scenario.get("name", "unnamed")),
        "completed": completed,
        "diagnostics": str(output_directory),
    }
    write_diagnostics(output_directory, state, last_snapshot, None, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="JSON scenario file")
    parser.add_argument(
        "--state-file", type=Path, default=root / "build/tmp/browser-gui/state.json"
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
        if not isinstance(scenario, dict):
            raise ValueError("scenario root must be a JSON object")
        state, socket_path, token = load_runtime_credentials(args.state_file)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_directory = (
            args.output_dir
            or Path(str(state["run_directory"])) / "diagnostics" / f"scenario-{timestamp}"
        ).resolve()

        def bridge_call(method: str, parameters: dict[str, Any] | None, timeout: float) -> Any:
            return call_bridge(socket_path, token, method, parameters, timeout)

        result = run_scenario(scenario, state, bridge_call, output_directory)
    except (BridgeError, OSError, ValueError, json.JSONDecodeError) as error:
        result = {"ok": False, "error": str(error)}
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
