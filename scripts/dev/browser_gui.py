#!/usr/bin/env python3
"""Run the real PlaScan Qt GUI in an isolated browser-accessible X display."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from browser_debug_client import BridgeError, call_bridge, load_runtime_credentials, sanitize_runtime_state
from browser_fixtures import FIXTURE_PROJECTS, fixture_project


STATE_VERSION = 2
DEFAULT_DISPLAY = 91
DEFAULT_VNC_PORT = 5901
DEFAULT_WEB_PORT = 6080
DEFAULT_NOVNC_WEB_PORT = 6081
DEFAULT_SCREEN = "1440x900"
PROCESS_ORDER = ("xvfb", "window_manager", "vnc", "novnc", "hub", "web", "plascan")
MAXIMUM_DEFAULT_PROJECT_COPY_BYTES = 5 * 1024 * 1024 * 1024


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_screen(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", value)
    if match is None:
        raise argparse.ArgumentTypeError("screen must use WIDTHxHEIGHT, for example 1440x900")
    width, height = (int(part) for part in match.groups())
    if width < 800 or height < 600:
        raise argparse.ArgumentTypeError("screen must be at least 800x600")
    return width, height


def loopback_host(value: str) -> str:
    if value in {"127.0.0.1", "localhost"}:
        return value
    raise argparse.ArgumentTypeError("browser gateway host must be 127.0.0.1 or localhost")


def novnc_url(host: str, port: int) -> str:
    query = urlencode({"autoconnect": "1", "resize": "scale", "view_only": "0"})
    return f"http://{host}:{port}/vnc.html?{query}"


def browser_url(host: str, port: int, token: str) -> str:
    return f"http://{host}:{port}/?{urlencode({'token': token})}"


def process_start_time(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    fields_after_name = text.rsplit(")", 1)[1].split()
    return fields_after_name[19] if len(fields_after_name) > 19 else None


def process_command(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace")


def process_matches(record: dict[str, Any]) -> bool:
    pid = int(record.get("pid", 0))
    expected_start = str(record.get("start_time", ""))
    marker = str(record.get("marker", ""))
    return (
        pid > 1
        and bool(expected_start)
        and process_start_time(pid) == expected_start
        and bool(marker)
        and marker in process_command(pid)
    )


def find_novnc_root(explicit: Path | None = None) -> Path | None:
    candidates = [explicit] if explicit else []
    candidates.extend(
        [
            Path("/usr/share/novnc"),
            Path("/usr/local/share/novnc"),
            repository_root() / "build" / "browser-runtime" / "novnc",
        ]
    )
    for candidate in candidates:
        if candidate is not None and (candidate / "vnc.html").is_file():
            return candidate.resolve()
    return None


def dependency_report(novnc_root: Path | None = None) -> tuple[dict[str, str], list[str]]:
    tools = {name: shutil.which(name) or "" for name in ("Xvfb", "x11vnc", "websockify")}
    missing = [name for name, path in tools.items() if not path]
    root = find_novnc_root(novnc_root)
    if root is None:
        missing.append("noVNC web files")
    else:
        tools["novnc_root"] = str(root)
    tools["window_manager"] = shutil.which("openbox") or ""
    return tools, missing


def geospatial_environment(root: Path) -> dict[str, str]:
    environment: dict[str, str] = {}
    proj_candidates = sorted(
        root.glob("build/*source-deps*/vcpkg_installed/*/share/proj/proj.db")
    )
    gdal_candidates = sorted(root.glob("build/*source-deps*/install/share/gdal/gdalvrt.xsd"))
    if proj_candidates:
        environment["PROJ_DATA"] = str(proj_candidates[0].parent)
        environment["PROJ_LIB"] = environment["PROJ_DATA"]
    if gdal_candidates:
        environment["GDAL_DATA"] = str(gdal_candidates[0].parent)
    return environment


def port_is_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
    except OSError:
        return False
    return True


def wait_for(predicate, description: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for {description}")


def wait_for_port(host: str, port: int) -> None:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET

    def connected() -> bool:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as client:
                client.settimeout(0.2)
                return client.connect_ex((host, port)) == 0
        except OSError:
            return False

    wait_for(connected, f"{host}:{port}")


def load_state(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"cannot read runtime state {path}: {error}") from error


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def print_result(args: argparse.Namespace, payload: dict[str, Any], lines: list[str]) -> None:
    if args.json_output:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return
    for line in lines:
        print(line)


def bridge_is_ready(socket_path: Path, token: str) -> bool:
    try:
        response = call_bridge(socket_path, token, "ping", timeout=0.5)
    except BridgeError:
        return False
    return isinstance(response, dict) and response.get("schema_version") == 1


def copy_project_for_case(project: Path, run_directory: Path) -> Path:
    case_directory = run_directory / "case"
    case_directory.mkdir(parents=True, exist_ok=True)
    destination = case_directory / project.name
    shutil.copy2(project, destination)
    sidecar = project.with_name(f"{project.stem}.files")
    if sidecar.is_dir():
        destination_sidecar = case_directory / sidecar.name
        shutil.copytree(
            sidecar,
            destination_sidecar,
            ignore=shutil.ignore_patterns(".plascan_tmp"),
        )
    return destination


def copy_project_for_agent_fixture(project: Path, run_directory: Path) -> Path:
    case_directory = run_directory / "case"
    case_directory.mkdir(parents=True, exist_ok=True)
    destination = case_directory / project.name
    shutil.copy2(project, destination)
    sidecar = project.with_name(f"{project.stem}.files")
    if not sidecar.is_dir():
        return destination
    destination_sidecar = case_directory / sidecar.name

    def copy_resource_directory(source: Path, target: Path, sparse: bool) -> None:
        target.mkdir(parents=True, exist_ok=True)
        for entry in source.iterdir():
            child_target = target / entry.name
            if entry.is_dir() and not entry.is_symlink():
                copy_resource_directory(entry, child_target, sparse)
            elif entry.is_symlink():
                continue
            elif sparse:
                with child_target.open("wb") as placeholder:
                    placeholder.truncate(entry.stat().st_size)
                shutil.copystat(entry, child_target)
            else:
                shutil.copy2(entry, child_target)

    def copy_directory(source: Path, target: Path, relative: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        for entry in source.iterdir():
            child_relative = relative / entry.name
            child_target = target / entry.name
            if entry.name in {".plascan_tmp", ".plascan.lock"}:
                continue
            should_skip = child_relative.parts == ("shared",) or (
                len(child_relative.parts) == 2
                and child_relative.parts[1] in {"assets", "mvs_output"}
            )
            if should_skip:
                if entry.is_dir():
                    copy_resource_directory(
                        entry,
                        child_target,
                        sparse=child_relative.parts != ("shared",),
                    )
                continue
            if entry.is_dir() and not entry.is_symlink():
                copy_directory(entry, child_target, child_relative)
            elif entry.is_symlink():
                continue
            else:
                shutil.copy2(entry, child_target)

    copy_directory(sidecar, destination_sidecar, Path())
    return destination


def project_copy_size(project: Path) -> int:
    size = project.stat().st_size
    sidecar = project.with_name(f"{project.stem}.files")
    if sidecar.is_dir():
        size += sum(path.stat().st_size for path in sidecar.rglob("*") if path.is_file())
    return size


def launch_process(
    name: str,
    command: list[str],
    environment: dict[str, str],
    log_directory: Path,
    working_directory: Path,
) -> dict[str, Any]:
    log_path = log_directory / f"{name}.log"
    with log_path.open("ab", buffering=0) as log_file:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    wait_for(lambda: process_start_time(process.pid) is not None, f"{name} process")
    return {
        "pid": process.pid,
        "start_time": process_start_time(process.pid),
        "marker": Path(command[0]).name,
        "command": command,
        "log": str(log_path),
    }


def terminate_process(record: dict[str, Any], timeout: float = 5.0) -> None:
    if not process_matches(record):
        return
    pid = int(record["pid"])
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and process_matches(record):
        time.sleep(0.1)
    if process_matches(record):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def cleanup_project_lock(state: dict[str, Any]) -> bool:
    project_path_text = str(state.get("project_path", "")).strip()
    plascan_record = state.get("processes", {}).get("plascan")
    if not project_path_text or not isinstance(plascan_record, dict):
        return False
    if process_matches(plascan_record):
        return False
    lock_path = Path(project_path_text).with_name(
        f"{Path(project_path_text).stem}.files"
    ) / ".plascan.lock"
    if lock_path.is_symlink():
        return False
    try:
        lock_pid = int(lock_path.read_text(encoding="utf-8").splitlines()[0])
        expected_pid = int(plascan_record.get("pid", 0))
    except (FileNotFoundError, IndexError, OSError, TypeError, ValueError):
        return False
    if lock_pid != expected_pid or expected_pid <= 1:
        return False
    lock_path.unlink()
    return True


def stop_runtime(state_path: Path, quiet: bool = False) -> bool:
    state = load_state(state_path)
    if state is None:
        if not quiet:
            print("PlaScan browser runtime is not running")
        return False
    processes = state.get("processes", {})
    for name in reversed(PROCESS_ORDER):
        record = processes.get(name)
        if isinstance(record, dict):
            terminate_process(record)
    cleanup_project_lock(state)
    state_path.unlink(missing_ok=True)
    if not quiet:
        print("PlaScan browser runtime stopped")
    return True


def command_doctor(args: argparse.Namespace) -> int:
    tools, missing = dependency_report(args.novnc_root)
    payload = {
        "ok": not missing,
        "tools": tools,
        "missing": missing,
        "platform": sys.platform,
    }
    if args.json_output:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return 1 if missing else 0
    print("PlaScan browser GUI dependencies:")
    for name in ("Xvfb", "x11vnc", "websockify", "novnc_root", "window_manager"):
        fallback = "missing (optional)" if name == "window_manager" else "missing (required)"
        print(f"  {name}: {tools.get(name) or fallback}")
    if missing:
        print("\nMissing required dependencies: " + ", ".join(missing), file=sys.stderr)
        print(
            "Install on Ubuntu with:\n"
            "  sudo apt-get update\n"
            "  sudo apt-get install -y xvfb x11vnc novnc websockify openbox",
            file=sys.stderr,
        )
        return 1
    return 0


def command_status(args: argparse.Namespace) -> int:
    state = load_state(args.state_file)
    if state is None:
        print_result(args, {"ok": False, "running": False},
                     ["PlaScan browser runtime is not running"])
        return 1
    processes = state.get("processes", {})
    all_running = True
    process_status: dict[str, Any] = {}
    for name in PROCESS_ORDER:
        record = processes.get(name)
        if record is None:
            continue
        running = isinstance(record, dict) and process_matches(record)
        all_running = all_running and running
        process_status[name] = {"running": running, "pid": record.get("pid")}
    bridge_ready = False
    try:
        _, socket_path, token = load_runtime_credentials(args.state_file)
        bridge_ready = bridge_is_ready(socket_path, token)
    except BridgeError:
        pass
    healthy = all_running and bridge_ready
    payload = {
        "ok": healthy,
        "running": all_running,
        "bridge_ready": bridge_ready,
        "run_id": state.get("run_id", ""),
        "url": state.get("url", ""),
        "novnc_url": state.get("novnc_url", ""),
        "log_directory": state.get("log_directory", ""),
        "processes": process_status,
    }
    lines = [
        *(f"{name}: {'running' if item['running'] else 'stopped'} (pid={item['pid']})"
          for name, item in process_status.items()),
        f"bridge: {'ready' if bridge_ready else 'unavailable'}",
        f"URL: {state.get('url', '')}",
        f"Logs: {state.get('log_directory', '')}",
    ]
    print_result(args, payload, lines)
    return 0 if healthy else 1


def command_stop(args: argparse.Namespace) -> int:
    stopped = stop_runtime(args.state_file, quiet=True)
    print_result(
        args,
        {"ok": stopped, "stopped": stopped},
        ["PlaScan browser runtime stopped" if stopped
         else "PlaScan browser runtime is not running"],
    )
    return 0 if stopped else 1


def command_inspect(args: argparse.Namespace) -> int:
    _, socket_path, token = load_runtime_credentials(args.state_file)
    result = call_bridge(socket_path, token, "snapshot", timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=None if args.json_output else 2))
    return 0


def command_action(args: argparse.Namespace) -> int:
    _, socket_path, token = load_runtime_credentials(args.state_file)
    parameters: dict[str, Any] = {
        "object_name": args.object_name,
        "operation": args.operation,
    }
    if args.value is not None:
        try:
            parameters["value"] = json.loads(args.value)
        except json.JSONDecodeError:
            parameters["value"] = args.value
    result = call_bridge(socket_path, token, "interact", parameters, timeout=args.timeout)
    print_result(args, {"ok": True, "result": result}, ["PlaScan browser action accepted"])
    return 0


def command_capture(args: argparse.Namespace) -> int:
    state, socket_path, token = load_runtime_credentials(args.state_file)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    default_directory = Path(str(state.get("run_directory", args.runtime_dir))) / "diagnostics" / timestamp
    output_directory = (args.output_dir or default_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    snapshot = call_bridge(socket_path, token, "snapshot", timeout=args.timeout)
    screenshot = call_bridge(socket_path, token, "screenshot", timeout=args.timeout)
    (output_directory / "snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sanitized_state = sanitize_runtime_state(state)
    (output_directory / "runtime.json").write_text(
        json.dumps(sanitized_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if isinstance(screenshot, dict) and screenshot.get("available"):
        (output_directory / "window.png").write_bytes(
            base64.b64decode(str(screenshot.get("data_base64", "")), validate=True))
    payload = {"ok": True, "diagnostics": str(output_directory)}
    print_result(args, payload, [f"Diagnostics: {output_directory}"])
    return 0


def command_start(args: argparse.Namespace) -> int:
    tools, missing = dependency_report(args.novnc_root)
    if missing:
        raise RuntimeError("missing dependencies: " + ", ".join(missing) + "; run 'doctor' for setup instructions")
    executable = args.executable.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"PlaScan executable is missing or not executable: {executable}")
    requested_project = fixture_project(repository_root(), args.fixture) if args.fixture else args.project
    if requested_project and not requested_project.resolve().is_file():
        raise RuntimeError(f"project file does not exist: {requested_project.resolve()}")

    previous = load_state(args.state_file)
    if previous is not None:
        if any(process_matches(record) for record in previous.get("processes", {}).values()):
            raise RuntimeError(f"browser runtime is already active; use '{Path(__file__).name} status'")
        cleanup_project_lock(previous)
        args.state_file.unlink(missing_ok=True)

    display_socket = Path(f"/tmp/.X11-unix/X{args.display}")
    if display_socket.exists():
        raise RuntimeError(f"X display :{args.display} is already in use; choose another --display")
    for host, port, label in (
        ("127.0.0.1", args.vnc_port, "VNC"),
        (args.host, args.novnc_web_port, "noVNC web"),
        (args.host, args.web_port, "debug hub"),
    ):
        if not port_is_available(host, port):
            raise RuntimeError(f"{label} port is already in use: {host}:{port}")

    width, height = args.screen
    runtime_directory = args.runtime_dir.resolve()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    run_directory = runtime_directory / "runs" / run_id
    log_directory = run_directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    profile = run_directory / "profile"
    bridge_socket = run_directory / "bridge.sock"
    ready_file = run_directory / "ready.json"
    token = secrets.token_urlsafe(32)
    source_project_path = requested_project.resolve() if requested_project else None
    project_path = source_project_path
    fixture_mode = ""
    if project_path and args.fixture and not args.copy_project:
        project_path = copy_project_for_agent_fixture(project_path, run_directory)
        fixture_mode = "sparse_resources_sandbox"
    elif project_path and args.copy_project:
        copy_size = project_copy_size(project_path)
        if copy_size > MAXIMUM_DEFAULT_PROJECT_COPY_BYTES and not args.allow_large_project_copy:
            raise RuntimeError(
                f"project copy requires {copy_size / (1024 ** 3):.1f} GiB; "
                "pass --allow-large-project-copy only when enough temporary storage is available"
            )
        project_path = copy_project_for_case(project_path, run_directory)
        fixture_mode = "full_copy" if args.fixture else ""
    environment = os.environ.copy()
    for name, value in geospatial_environment(repository_root()).items():
        environment.setdefault(name, value)
    environment.update(
        {
            "DISPLAY": f":{args.display}",
            "QT_QPA_PLATFORM": "xcb",
            "QT_X11_NO_MITSHM": "1",
            "XDG_CONFIG_HOME": str(profile / "config"),
            "XDG_DATA_HOME": str(profile / "data"),
            "XDG_CACHE_HOME": str(profile / "cache"),
            "PLASCAN_BROWSER_TEST": "1",
            "PLASCAN_BROWSER_DEBUG_SOCKET": str(bridge_socket),
            "PLASCAN_BROWSER_DEBUG_TOKEN": token,
        }
    )
    for directory in (profile / "config", profile / "data", profile / "cache"):
        directory.mkdir(parents=True, exist_ok=True)

    processes: dict[str, dict[str, Any]] = {}
    try:
        processes["xvfb"] = launch_process(
            "xvfb",
            [tools["Xvfb"], f":{args.display}", "-screen", "0", f"{width}x{height}x24", "-nolisten", "tcp", "-noreset"],
            environment,
            log_directory,
            repository_root(),
        )
        wait_for(display_socket.exists, f"X display :{args.display}")
        if tools["window_manager"]:
            processes["window_manager"] = launch_process(
                "window_manager",
                [tools["window_manager"], "--sm-disable"],
                environment,
                log_directory,
                repository_root(),
            )
        processes["vnc"] = launch_process(
            "vnc",
            [
                tools["x11vnc"], "-display", f":{args.display}", "-rfbport", str(args.vnc_port),
                "-localhost", "-forever", "-shared", "-nopw", "-noxdamage", "-repeat",
            ],
            environment,
            log_directory,
            repository_root(),
        )
        wait_for_port("127.0.0.1", args.vnc_port)
        processes["novnc"] = launch_process(
            "novnc",
            [tools["websockify"], "--web", tools["novnc_root"],
             f"{args.host}:{args.novnc_web_port}", f"127.0.0.1:{args.vnc_port}"],
            environment,
            log_directory,
            repository_root(),
        )
        wait_for_port(args.host, args.novnc_web_port)
        app_command = [str(executable)]
        if project_path:
            app_command.append(str(project_path))
        processes["plascan"] = launch_process(
            "plascan", app_command, environment, log_directory, repository_root()
        )
        wait_for(lambda: process_matches(processes["plascan"]), "PlaScan process", args.ready_timeout)
        wait_for(lambda: bridge_is_ready(bridge_socket, token),
                 "PlaScan debug bridge", args.ready_timeout)

        direct_url = novnc_url(args.host, args.novnc_web_port)
        url = browser_url(args.host, args.web_port, token)
        state = {
            "version": STATE_VERSION,
            "run_id": run_id,
            "url": url,
            "novnc_url": direct_url,
            "display": args.display,
            "vnc_port": args.vnc_port,
            "web_port": args.web_port,
            "novnc_web_port": args.novnc_web_port,
            "bridge_socket": str(bridge_socket),
            "token": token,
            "run_directory": str(run_directory),
            "log_directory": str(log_directory),
            "profile_directory": str(profile),
            "project_path": str(project_path) if project_path else "",
            "project_source_path": str(source_project_path) if source_project_path else "",
            "project_is_copy": bool(project_path and (args.copy_project or args.fixture)),
            "project_read_only": fixture_mode == "sparse_resources_sandbox",
            "fixture": args.fixture or "",
            "fixture_mode": fixture_mode,
            "ready_file": str(ready_file),
            "processes": processes,
        }
        save_state(args.state_file, state)
        processes["hub"] = launch_process(
            "hub",
            [sys.executable, str(repository_root() / "scripts/dev/browser_debug_server.py"),
             "--host", args.host, "--port", str(args.web_port),
             "--state-file", str(args.state_file),
             "--web-root", str(repository_root() / "scripts/dev/browser_gui_web"),
             "--bridge-socket", str(bridge_socket),
             "--novnc-url", direct_url],
            environment,
            log_directory,
            repository_root(),
        )
        state["processes"] = processes
        save_state(args.state_file, state)
        wait_for_port(args.host, args.web_port)
        ready_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "url": url,
            "novnc_url": direct_url,
            "bridge_ready": True,
            "log_directory": str(log_directory),
        }
        ready_file.write_text(
            json.dumps(ready_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        ready_file.chmod(0o600)
    except Exception:
        for name in reversed(PROCESS_ORDER):
            if name in processes:
                terminate_process(processes[name])
        args.state_file.unlink(missing_ok=True)
        ready_file.unlink(missing_ok=True)
        raise

    payload = {
        "ok": True,
        "run_id": run_id,
        "url": url,
        "novnc_url": direct_url,
        "bridge_ready": True,
        "logs": str(log_directory),
        "ready_file": str(ready_file),
    }
    print_result(args, payload, [
        "PlaScan browser runtime started",
        f"Run: {run_id}",
        f"URL: {url}",
        f"Logs: {log_directory}",
    ])
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    runtime = root / "build" / "tmp" / "browser-gui"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=runtime)
    parser.add_argument("--state-file", type=Path, default=runtime / "state.json")
    parser.add_argument("--novnc-root", type=Path)
    parser.add_argument("--json", dest="json_output", action="store_true",
                        help="print a machine-readable JSON result")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="check browser runtime dependencies")
    subparsers.add_parser("status", help="show runtime status and browser URL")
    subparsers.add_parser("stop", help="stop only the recorded browser runtime processes")
    inspect = subparsers.add_parser("inspect", help="print a structured PlaScan debug snapshot")
    inspect.add_argument("--timeout", type=float, default=10.0)
    action = subparsers.add_parser("action", help="operate an allow-listed named Qt control")
    action.add_argument("--object-name", required=True)
    action.add_argument(
        "--operation",
        required=True,
        choices=("activate", "focus", "set_text", "set_value", "set_checked",
                 "select_index", "cancel_task"),
    )
    action.add_argument("--value")
    action.add_argument("--timeout", type=float, default=10.0)
    capture = subparsers.add_parser("capture", help="write a failure diagnostics bundle")
    capture.add_argument("--output-dir", type=Path)
    capture.add_argument("--timeout", type=float, default=15.0)
    start = subparsers.add_parser("start", help="start PlaScan in the isolated browser runtime")
    start.add_argument("--executable", type=Path, default=root / "build" / "linux-source-release" / "bin" / "plascan")
    project_source = start.add_mutually_exclusive_group()
    project_source.add_argument("--project", type=Path)
    project_source.add_argument("--fixture", choices=tuple(sorted(FIXTURE_PROJECTS)))
    start.add_argument("--copy-project", action="store_true",
                       help="copy the complete project sidecar into the isolated run directory")
    start.add_argument("--allow-large-project-copy", action="store_true",
                       help="allow --copy-project when the complete project exceeds 5 GiB")
    start.add_argument("--host", type=loopback_host, default="127.0.0.1")
    start.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    start.add_argument("--novnc-web-port", type=int, default=DEFAULT_NOVNC_WEB_PORT)
    start.add_argument("--vnc-port", type=int, default=DEFAULT_VNC_PORT)
    start.add_argument("--display", type=int, default=DEFAULT_DISPLAY)
    start.add_argument("--screen", type=parse_screen, default=parse_screen(DEFAULT_SCREEN))
    start.add_argument("--ready-timeout", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return command_doctor(args)
        if args.command == "status":
            return command_status(args)
        if args.command == "stop":
            return command_stop(args)
        if args.command == "start":
            return command_start(args)
        if args.command == "inspect":
            return command_inspect(args)
        if args.command == "action":
            return command_action(args)
        if args.command == "capture":
            return command_capture(args)
    except (BridgeError, OSError, RuntimeError, ValueError) as error:
        if args.json_output:
            print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False,
                             separators=(",", ":")))
        else:
            print(f"error: {error}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
