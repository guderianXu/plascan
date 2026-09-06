#!/usr/bin/env python3
"""Low-token snapshot and condition helpers for PlaScan browser agents."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


STATE_FIELDS = (
    "text",
    "placeholder",
    "checked",
    "checkable",
    "value",
    "minimum",
    "maximum",
    "current_index",
    "current_text",
    "count",
    "row_count",
    "column_count",
)


def agent_capabilities(fixtures: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "commands": {
            "inspect": "summary or a paginated controls/logs/artifacts/tasks/modal view",
            "query": "search named controls and actions",
            "describe": "show one unique target and its supported operations",
            "act": "perform one allow-listed semantic operation",
            "close-dialog": "reject the active modal dialog without coordinate input",
            "cancel-task": "request cancellation through a named active task status widget",
            "task-command": "pause, resume, cancel, reprioritize, or reorder a scheduler-managed run",
            "form": "validate and apply a reversible batch of form values",
            "wait": "poll until a nested condition matches",
            "assert": "evaluate a nested condition once",
            "watch": "emit changed summaries as JSONL",
            "diagnose": "write a bounded replayable diagnostic bundle",
        },
        "views": ["summary", "modal", "controls", "logs", "artifacts", "tasks"],
        "operations": [
            "activate", "focus", "set_text", "set_value", "set_checked", "select_index", "cancel_task"
        ],
        "conditions": {
            "combinators": ["all", "any", "not"],
            "path_operators": ["exists", "equals", "not_equals", "contains", "gt", "gte", "lt", "lte"],
            "legacy": [
                "object_name", "visible", "enabled", "text_contains", "modal_present",
                "modal_title_contains", "project_open", "project_dirty", "image_count", "task_count",
                "recent_error_contains",
            ],
        },
        "fixtures": fixtures,
        "safety": {
            "transport": "local IPC with per-run token",
            "direct_project_form_requires_flag": True,
            "project_write_actions_require_flag": True,
            "sparse_resource_fixture_actions_are_read_only": True,
            "arbitrary_qt_methods": False,
            "arbitrary_file_io": False,
            "true_copy_on_write_sandbox": False,
        },
    }


def named_objects(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(item: Any, window_title: str = "") -> None:
        if not isinstance(item, dict):
            return
        current_window = str(item.get("window_title", "")) or window_title
        if item.get("object_name"):
            copy = {key: value for key, value in item.items() if key not in ("children", "actions")}
            copy["window"] = current_window
            copy["operations"] = supported_operations(copy)
            geometry = copy.get("geometry", {})
            identity = json.dumps(
                [
                    copy.get("object_name"),
                    copy.get("class"),
                    current_window,
                    geometry.get("global_x"),
                    geometry.get("global_y"),
                    geometry.get("width"),
                    geometry.get("height"),
                    copy.get("text"),
                    copy.get("current_text"),
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if identity not in seen:
                found.append(copy)
                seen.add(identity)
        for child in item.get("children", []):
            visit(child, current_window)
        for action in item.get("actions", []):
            visit(action, current_window)

    for window in snapshot.get("windows", []):
        visit(window)
    return found


def supported_operations(item: dict[str, Any]) -> list[str]:
    class_name = str(item.get("class", ""))
    operations: list[str] = []
    if class_name == "QAction" or class_name.endswith("Button"):
        operations.append("activate")
    if class_name != "QAction":
        operations.append("focus")
    if class_name == "QLineEdit":
        operations.append("set_text")
    if class_name in {"QSpinBox", "QDoubleSpinBox", "QSlider", "QScrollBar"}:
        operations.append("set_value")
    if bool(item.get("checkable")):
        operations.append("set_checked")
    if class_name in {"QComboBox", "QTabWidget", "QStackedWidget"}:
        operations.append("select_index")
    if class_name == "TaskStatusWidget":
        operations.append("cancel_task")
    return operations


def compact_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    project = dict(snapshot.get("project", {}))
    artifacts = project.pop("artifacts", [])
    controls = {}
    for item in named_objects(snapshot):
        name = str(item["object_name"])
        state = {
            key: item[key]
            for key in ("class", "visible", "enabled", "window", *STATE_FIELDS)
            if key in item
        }
        controls.setdefault(name, []).append(state)
    return {
        "application": snapshot.get("application", {}),
        "project": project,
        "tasks": snapshot.get("tasks", []),
        "recent_error": snapshot.get("recent_error", ""),
        "log_count": len(snapshot.get("logs", [])),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "controls": controls,
    }


def revision_for(snapshot: dict[str, Any]) -> str:
    payload = json.dumps(
        compact_state(snapshot), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    project = dict(snapshot.get("project", {}))
    artifacts = project.pop("artifacts", [])
    controls = named_objects(snapshot)
    return {
        "application": snapshot.get("application", {}),
        "project": project,
        "tasks": snapshot.get("tasks", []),
        "recent_error": snapshot.get("recent_error", ""),
        "counts": {
            "controls": len(controls),
            "logs": len(snapshot.get("logs", [])),
            "artifacts": len(artifacts),
        },
    }


def modal_view(snapshot: dict[str, Any]) -> dict[str, Any]:
    title = str(snapshot.get("application", {}).get("modal_window", ""))
    windows = [
        item
        for item in snapshot.get("windows", [])
        if isinstance(item, dict) and (item.get("modal") or item.get("window_title") == title)
    ]
    return {"present": bool(title), "title": title, "windows": windows}


def view_items(snapshot: dict[str, Any], view: str) -> list[dict[str, Any]]:
    if view == "controls":
        return named_objects(snapshot)
    if view == "logs":
        return [item for item in snapshot.get("logs", []) if isinstance(item, dict)]
    if view == "artifacts":
        return [
            item
            for item in snapshot.get("project", {}).get("artifacts", [])
            if isinstance(item, dict)
        ]
    if view == "tasks":
        return [item for item in snapshot.get("tasks", []) if isinstance(item, dict)]
    raise ValueError(f"view {view!r} is not pageable")


def filter_items(items: Iterable[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    lowered = query.casefold().strip()
    if not lowered:
        return list(items)
    return [
        item
        for item in items
        if lowered in json.dumps(item, ensure_ascii=False, sort_keys=True).casefold()
    ]


def paginate(items: list[dict[str, Any]], offset: int, limit: int) -> dict[str, Any]:
    if offset < 0 or limit < 1 or limit > 200:
        raise ValueError("offset must be non-negative and limit must be between 1 and 200")
    page = items[offset : offset + limit]
    next_offset = offset + len(page)
    return {
        "total": len(items),
        "offset": offset,
        "limit": limit,
        "next_offset": next_offset if next_offset < len(items) else None,
        "items": page,
    }


def diff_states(before: Any, after: Any, path: str = "", maximum: int = 100) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []

    def visit(old: Any, new: Any, current: str) -> None:
        if len(changes) >= maximum or old == new:
            return
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                visit(old.get(key), new.get(key), f"{current}.{key}" if current else key)
            return
        changes.append({"path": current or "$", "before": old, "after": new})

    visit(before, after, path)
    return changes


def value_at_path(document: Any, path: str) -> tuple[bool, Any]:
    current = document
    for part in path.split(".") if path else []:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "equals":
        return actual == expected
    if operator == "not_equals":
        return actual != expected
    if operator == "contains":
        return str(expected) in str(actual)
    if operator in {"gt", "gte", "lt", "lte"}:
        left, right = float(actual), float(expected)
        return {"gt": left > right, "gte": left >= right, "lt": left < right, "lte": left <= right}[operator]
    raise ValueError(f"unsupported comparison operator: {operator}")


def condition_matches(snapshot: dict[str, Any], condition: dict[str, Any]) -> tuple[bool, str]:
    if "all" in condition:
        clauses = condition["all"]
        if not isinstance(clauses, list):
            raise ValueError("all must be an array")
        for clause in clauses:
            matched, reason = condition_matches(snapshot, clause)
            if not matched:
                return False, reason
        return True, "all conditions matched"
    if "any" in condition:
        clauses = condition["any"]
        if not isinstance(clauses, list):
            raise ValueError("any must be an array")
        reasons = []
        for clause in clauses:
            matched, reason = condition_matches(snapshot, clause)
            if matched:
                return True, "one condition matched"
            reasons.append(reason)
        return False, "; ".join(reasons)
    if "not" in condition:
        matched, reason = condition_matches(snapshot, condition["not"])
        return (not matched, "negated condition matched" if matched else "negated condition did not match")
    if "path" in condition:
        allowed = {"path", "exists", "equals", "not_equals", "contains", "gt", "gte", "lt", "lte"}
        unknown = set(condition) - allowed
        if unknown:
            raise ValueError("unsupported path condition fields: " + ", ".join(sorted(unknown)))
        path = str(condition["path"])
        exists, actual = value_at_path(snapshot, path)
        if "exists" in condition and exists != bool(condition["exists"]):
            return False, f"{path} existence is {exists}"
        for operator in allowed - {"path", "exists"}:
            if operator in condition and (not exists or not _compare(actual, operator, condition[operator])):
                return False, f"{path} is {actual!r}; {operator} {condition[operator]!r} failed"
        return True, f"{path} matched"
    return legacy_condition_matches(snapshot, condition)


def legacy_condition_matches(
    snapshot: dict[str, Any], condition: dict[str, Any]
) -> tuple[bool, str]:
    supported = {
        "object_name", "exists", "visible", "enabled", "text_contains", "modal_present",
        "modal_title_contains", "project_open", "project_dirty", "image_count", "task_count",
        "recent_error_contains",
    }
    unknown = set(condition) - supported
    if unknown:
        raise ValueError("unsupported condition fields: " + ", ".join(sorted(unknown)))
    object_name = condition.get("object_name")
    if object_name is not None:
        matches = [item for item in named_objects(snapshot) if item.get("object_name") == object_name]
        should_exist = bool(condition.get("exists", True))
        if bool(matches) != should_exist:
            return False, f"object {object_name!r} existence is {bool(matches)}, expected {should_exist}"
        if should_exist:
            if len(matches) != 1:
                return False, f"object {object_name!r} is not unique ({len(matches)} matches)"
            item = matches[0]
            for field in ("visible", "enabled"):
                if field in condition and item.get(field) != condition[field]:
                    return False, f"object {object_name!r} {field} is {item.get(field)!r}"
            if "text_contains" in condition:
                text = str(item.get("text", item.get("current_text", "")))
                if str(condition["text_contains"]) not in text:
                    return False, f"object {object_name!r} text does not contain {condition['text_contains']!r}"
    application = snapshot.get("application", {})
    modal_title = str(application.get("modal_window", ""))
    if "modal_present" in condition and bool(modal_title) != bool(condition["modal_present"]):
        return False, f"modal presence is {bool(modal_title)}"
    if "modal_title_contains" in condition and str(condition["modal_title_contains"]) not in modal_title:
        return False, f"modal title {modal_title!r} does not contain {condition['modal_title_contains']!r}"
    project = snapshot.get("project", {})
    for field, source in (("project_open", "open"), ("project_dirty", "dirty")):
        if field in condition and project.get(source) != condition[field]:
            return False, f"{field} is {project.get(source)!r}"
    if "image_count" in condition and int(project.get("image_count", -1)) != int(condition["image_count"]):
        return False, f"image count is {project.get('image_count', -1)}"
    if "task_count" in condition and len(snapshot.get("tasks", [])) != int(condition["task_count"]):
        return False, f"task count is {len(snapshot.get('tasks', []))}"
    if "recent_error_contains" in condition:
        if str(condition["recent_error_contains"]) not in str(snapshot.get("recent_error", "")):
            return False, "recent error does not contain expected text"
    return True, "condition matched"


def wait_for_condition(snapshot_provider, condition: dict[str, Any], timeout: float, poll_interval: float = 0.2):
    import time

    deadline = time.monotonic() + timeout
    last_reason = "no snapshot received"
    while True:
        snapshot = snapshot_provider()
        matched, last_reason = condition_matches(snapshot, condition)
        if matched:
            return snapshot
        if time.monotonic() >= deadline:
            raise TimeoutError(f"condition did not match within {timeout:.1f}s: {last_reason}")
        time.sleep(poll_interval)
