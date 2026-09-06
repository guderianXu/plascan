#!/usr/bin/env python3
"""Agent-first CLI for deterministic PlaScan GUI inspection and operation."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from browser_agent_protocol import (
    agent_capabilities,
    compact_state,
    condition_matches,
    diff_states,
    filter_items,
    modal_view,
    named_objects,
    paginate,
    revision_for,
    summary,
    view_items,
    wait_for_condition,
)
from browser_debug_client import BridgeError, call_bridge, load_runtime_credentials, sanitize_runtime_state
from browser_fixtures import fixture_catalog


MAXIMUM_HISTORY = 8
PROJECT_WRITE_TOKENS = (
    "save", "addphoto", "addfolder", "import", "delete", "remove", "create", "generate", "start", "run",
)


class AgentProtocolError(RuntimeError):
    def __init__(self, code: str, message: str, suggestions: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.suggestions = suggestions or []


def json_argument(value: str) -> Any:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def capabilities(root: Path) -> dict[str, Any]:
    return agent_capabilities(fixture_catalog(root))


def history_path(state: dict[str, Any]) -> Path:
    return Path(str(state["run_directory"])) / "agent" / "history.json"


def read_history(state: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        document = json.loads(history_path(state).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    return document if isinstance(document, list) else []


def record_history(state: dict[str, Any], snapshot: dict[str, Any]) -> str:
    revision = revision_for(snapshot)
    history = [entry for entry in read_history(state) if entry.get("revision") != revision]
    history.append({"revision": revision, "state": compact_state(snapshot)})
    path = history_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(history[-MAXIMUM_HISTORY:], ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return revision


def bridge_snapshot(socket_path: Path, token: str, timeout: float) -> dict[str, Any]:
    result = call_bridge(socket_path, token, "snapshot", timeout=timeout)
    if not isinstance(result, dict):
        raise AgentProtocolError("invalid_snapshot", "bridge snapshot is not an object")
    return result


def inspect_snapshot(snapshot: dict[str, Any], view: str, query: str, offset: int, limit: int) -> Any:
    if view == "summary":
        return summary(snapshot)
    if view == "modal":
        return modal_view(snapshot)
    return paginate(filter_items(view_items(snapshot, view), query), offset, limit)


def find_target(snapshot: dict[str, Any], object_name: str) -> dict[str, Any]:
    matches = [item for item in named_objects(snapshot) if item.get("object_name") == object_name]
    if not matches:
        raise AgentProtocolError("object_not_found", f"object {object_name!r} was not found", ["run query first"])
    if len(matches) != 1:
        raise AgentProtocolError("ambiguous_object", f"object {object_name!r} matched {len(matches)} objects")
    return matches[0]


def is_project_write(object_name: str, operation: str) -> bool:
    lowered = object_name.casefold()
    return operation == "activate" and any(token in lowered for token in PROJECT_WRITE_TOKENS)


def validate_operation(
    state: dict[str, Any], target: dict[str, Any], operation: str, allow_project_write: bool
) -> None:
    if operation not in target.get("operations", []):
        raise AgentProtocolError(
            "unsupported_operation",
            f"{operation!r} is not supported by {target.get('object_name')!r}",
            ["run describe to list supported operations"],
        )
    if operation == "activate" and (not target.get("visible", True) or not target.get("enabled", True)):
        raise AgentProtocolError("disabled", "target is disabled or hidden")
    if is_project_write(str(target["object_name"]), operation):
        if state.get("project_read_only"):
            raise AgentProtocolError(
                "project_write_blocked",
                "the active fixture is read-only and this action may change project data",
                ["restart with a complete writable project copy"],
            )
        if not allow_project_write:
            raise AgentProtocolError(
                "project_write_confirmation_required",
                "this action may change project data",
                ["pass --allow-project-write only after confirming the active project is disposable"],
            )


def infer_form_operation(target: dict[str, Any], value: Any) -> str:
    operations = target.get("operations", [])
    for candidate in ("set_checked", "set_text", "set_value", "select_index"):
        if candidate in operations:
            if candidate == "set_checked" and not isinstance(value, bool):
                continue
            return candidate
    raise AgentProtocolError("unsupported_form_control", f"cannot infer a setter for {target.get('object_name')!r}")


def previous_value(target: dict[str, Any], operation: str) -> Any:
    return {
        "set_checked": target.get("checked"),
        "set_text": target.get("text", ""),
        "set_value": target.get("value"),
        "select_index": target.get("current_index"),
    }.get(operation)


def apply_form(
    state: dict[str, Any], snapshot: dict[str, Any], bridge, values: dict[str, Any], allow_project_write: bool
) -> dict[str, Any]:
    prepared = []
    for object_name, specification in values.items():
        target = find_target(snapshot, object_name)
        if isinstance(specification, dict) and "value" in specification:
            value = specification["value"]
            operation = str(specification.get("operation") or infer_form_operation(target, value))
        else:
            value = specification
            operation = infer_form_operation(target, value)
        validate_operation(state, target, operation, allow_project_write)
        prepared.append((object_name, operation, value, previous_value(target, operation)))
    applied = []
    try:
        for object_name, operation, value, old_value in prepared:
            bridge("interact", {"object_name": object_name, "operation": operation, "value": value})
            applied.append((object_name, operation, old_value))
    except Exception:
        for object_name, operation, old_value in reversed(applied):
            if old_value is not None:
                try:
                    bridge("interact", {"object_name": object_name, "operation": operation, "value": old_value})
                except Exception:
                    pass
        raise
    return {"validated": len(prepared), "applied": [item[0] for item in prepared], "rollback_available": True}


def validate_form_scope(state: dict[str, Any], allow_project_write: bool) -> None:
    if not state.get("project_is_copy") and not allow_project_write:
        raise AgentProtocolError(
            "project_write_confirmation_required",
            "form changes can persist to the directly opened project",
            ["use a named fixture/copy or pass --allow-project-write for a disposable project"],
        )


def diagnostic_bundle(state: dict[str, Any], snapshot: dict[str, Any], screenshot: Any) -> Path:
    output = Path(str(state["run_directory"])) / "diagnostics" / f"agent-{time.strftime('%Y%m%d-%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    sanitized = sanitize_runtime_state(state)
    documents = {
        "summary.json": summary(snapshot),
        "snapshot.json": snapshot,
        "runtime.json": sanitized,
        "replay.json": {
            "project": state.get("project_path", ""),
            "fixture": state.get("fixture", ""),
            "commands": ["browser_agent.py inspect", "browser_agent.py query --query <text>"],
        },
    }
    for filename, document in documents.items():
        (output / filename).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if isinstance(screenshot, dict) and screenshot.get("available"):
        (output / "window.png").write_bytes(base64.b64decode(str(screenshot.get("data_base64", "")), validate=True))
    return output


def build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, default=root / "build/tmp/browser-gui/state.json")
    parser.add_argument("--pretty", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities")
    inspect = commands.add_parser("inspect")
    inspect.add_argument(
        "--view",
        choices=("summary", "modal", "controls", "logs", "artifacts", "tasks"),
        default="summary",
    )
    inspect.add_argument("--query", default="")
    inspect.add_argument("--offset", type=int, default=0)
    inspect.add_argument("--limit", type=int, default=25)
    inspect.add_argument("--since")
    query = commands.add_parser("query")
    query.add_argument("--query", required=True)
    query.add_argument("--class-name")
    query.add_argument("--offset", type=int, default=0)
    query.add_argument("--limit", type=int, default=25)
    describe = commands.add_parser("describe")
    describe.add_argument("--target", required=True)
    act = commands.add_parser("act")
    act.add_argument("--target", required=True)
    act.add_argument("--operation", required=True)
    act.add_argument("--value")
    act.add_argument("--allow-project-write", action="store_true")
    commands.add_parser("close-dialog")
    cancel_task = commands.add_parser("cancel-task")
    cancel_task.add_argument("--target", required=True)
    task_command = commands.add_parser("task-command")
    task_command.add_argument(
        "--action",
        required=True,
        choices=("pause", "resume", "cancel", "set_priority", "move_before", "move_after"),
    )
    task_command.add_argument("--run-id", required=True)
    task_command.add_argument("--reference-run-id")
    task_command.add_argument("--priority", type=int, default=0)
    task_command.add_argument("--revision", type=int, default=0)
    form = commands.add_parser("form")
    form.add_argument("--values", required=True, help="JSON object or @file")
    form.add_argument("--allow-project-write", action="store_true")
    for name in ("wait", "assert"):
        command = commands.add_parser(name)
        command.add_argument("--condition", required=True, help="JSON object or @file")
        command.add_argument("--timeout", type=float, default=10.0)
    watch = commands.add_parser("watch")
    watch.add_argument("--duration", type=float, default=30.0)
    watch.add_argument("--interval", type=float, default=0.5)
    watch.add_argument("--max-events", type=int, default=100)
    commands.add_parser("diagnose")
    return parser


def execute(args: argparse.Namespace, root: Path) -> tuple[int, Any]:
    if args.command == "capabilities":
        return 0, capabilities(root)
    state, socket_path, token = load_runtime_credentials(args.state_file)
    bridge = lambda method, parameters=None: call_bridge(socket_path, token, method, parameters, timeout=10.0)
    snapshot = bridge_snapshot(socket_path, token, 10.0)
    revision = revision_for(snapshot)
    if args.command in {"inspect", "query"}:
        view = "controls" if args.command == "query" else args.view
        if args.command == "query" and args.class_name:
            controls = filter_items(view_items(snapshot, "controls"), args.query)
            controls = [item for item in controls if item.get("class") == args.class_name]
            data = paginate(controls, args.offset, args.limit)
        else:
            data = inspect_snapshot(snapshot, view, args.query, args.offset, args.limit)
        result: dict[str, Any] = {"schema_version": 1, "revision": revision, "view": view, "data": data}
        if args.command == "inspect" and args.since:
            old = next((entry for entry in read_history(state) if entry.get("revision") == args.since), None)
            if old is None:
                raise AgentProtocolError("revision_not_found", f"revision {args.since!r} is not in bounded history")
            result["unchanged"] = args.since == revision
            result["changes"] = diff_states(old.get("state", {}), compact_state(snapshot))
        record_history(state, snapshot)
        return 0, result
    if args.command == "describe":
        return 0, {"revision": revision, "target": find_target(snapshot, args.target)}
    if args.command == "act":
        target = find_target(snapshot, args.target)
        validate_operation(state, target, args.operation, args.allow_project_write)
        parameters = {"object_name": args.target, "operation": args.operation}
        if args.value is not None:
            parameters["value"] = json_argument(args.value)
        return 0, {"accepted": True, "target": args.target, "result": bridge("interact", parameters)}
    if args.command == "close-dialog":
        return 0, {"accepted": True, "result": bridge("close_dialog")}
    if args.command == "cancel-task":
        target = find_target(snapshot, args.target)
        validate_operation(state, target, "cancel_task", allow_project_write=False)
        result = bridge(
            "interact", {"object_name": args.target, "operation": "cancel_task"}
        )
        return 0, {"accepted": True, "target": args.target, "result": result}
    if args.command == "task-command":
        task = next(
            (item for item in snapshot.get("tasks", []) if item.get("run_id") == args.run_id),
            None,
        )
        if task is None:
            raise AgentProtocolError("run_not_found", f"run {args.run_id!r} is not visible")
        if not task.get("scheduler_managed"):
            raise AgentProtocolError(
                "task_not_scheduler_managed",
                "the selected legacy task does not implement the scheduler protocol",
            )
        capability = {
            "pause": "can_pause",
            "resume": "can_resume",
            "cancel": "can_cancel",
            "set_priority": "can_reorder",
            "move_before": "can_reorder",
            "move_after": "can_reorder",
        }[args.action]
        if not task.get(capability):
            raise AgentProtocolError(
                "task_capability_unavailable",
                f"run {args.run_id!r} does not currently expose {capability}",
            )
        if args.action in {"move_before", "move_after"} and not args.reference_run_id:
            raise AgentProtocolError(
                "reference_run_required", "queue movement requires --reference-run-id"
            )
        parameters = {
            "action": args.action,
            "run_id": args.run_id,
            "priority": args.priority,
            "revision": args.revision or int(task.get("revision", 0)),
        }
        if args.reference_run_id:
            parameters["reference_run_id"] = args.reference_run_id
        return 0, {
            "accepted": True,
            "run_id": args.run_id,
            "result": bridge("task_command", parameters),
        }
    if args.command == "form":
        validate_form_scope(state, args.allow_project_write)
        values = json_argument(args.values)
        if not isinstance(values, dict):
            raise AgentProtocolError("invalid_form", "form values must be a JSON object")
        return 0, apply_form(state, snapshot, bridge, values, args.allow_project_write)
    if args.command in {"wait", "assert"}:
        condition = json_argument(args.condition)
        if not isinstance(condition, dict):
            raise AgentProtocolError("invalid_condition", "condition must be a JSON object")
        if args.command == "wait":
            try:
                matched_snapshot = wait_for_condition(
                    lambda: bridge_snapshot(socket_path, token, args.timeout), condition, args.timeout
                )
            except TimeoutError as error:
                raise AgentProtocolError(
                    "condition_timeout", str(error), ["run inspect --view summary", "run diagnose"]
                ) from error
            return 0, {
                "matched": True,
                "revision": record_history(state, matched_snapshot),
                "summary": summary(matched_snapshot),
            }
        matched, reason = condition_matches(snapshot, condition)
        if not matched:
            raise AgentProtocolError("assertion_failed", reason, ["run inspect --view summary", "run diagnose"])
        return 0, {"matched": True, "reason": reason, "revision": revision}
    if args.command == "diagnose":
        output = diagnostic_bundle(state, snapshot, bridge("screenshot"))
        return 0, {"diagnostics": str(output), "revision": revision}
    raise AgentProtocolError("unsupported_command", f"unsupported command: {args.command}")


def run_watch(args: argparse.Namespace) -> int:
    state, socket_path, token = load_runtime_credentials(args.state_file)
    deadline = time.monotonic() + max(0.1, min(args.duration, 3600.0))
    previous = ""
    emitted = 0
    while time.monotonic() < deadline and emitted < max(1, args.max_events):
        snapshot = bridge_snapshot(socket_path, token, 10.0)
        revision = revision_for(snapshot)
        if revision != previous:
            event = {"event": "state.changed", "revision": revision, "data": summary(snapshot)}
            print(
                json.dumps(event, ensure_ascii=False, separators=(",", ":")),
                flush=True,
            )
            record_history(state, snapshot)
            previous = revision
            emitted += 1
        time.sleep(max(0.1, min(args.interval, 10.0)))
    return 0


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    args = build_parser(root).parse_args(argv)
    if args.command == "watch":
        try:
            return run_watch(args)
        except (AgentProtocolError, BridgeError, OSError, ValueError) as error:
            payload = {
                "ok": False,
                "error": {
                    "code": getattr(error, "code", "agent_error"),
                    "message": str(error),
                },
            }
            print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            return 1
    try:
        code, result = execute(args, root)
        payload = {"ok": True, "result": result}
    except (AgentProtocolError, BridgeError, OSError, ValueError, json.JSONDecodeError) as error:
        code = 1
        payload = {
            "ok": False,
            "error": {
                "code": getattr(error, "code", "agent_error"),
                "message": str(error),
                "suggested_actions": getattr(error, "suggestions", []),
            },
        }
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
