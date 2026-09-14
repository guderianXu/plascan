#!/usr/bin/env python3
"""Run PlaScan on a synthetic dataset and emit truth-based JSON/HTML reports."""

from __future__ import annotations

import argparse
import json
import locale
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from executable_resolver import resolve_build_executable
from synthetic_e2e_metrics import evaluate_run, write_html_report


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


QUALITY_RUN_CONFIG: dict[str, dict[str, Any]] = {
    "coarse": {
        "sfm_quality": 2,
        "mvs_quality": "low",
        "mesh_resolution": 64,
        "dem_resolution_m": 0.18,
        "angular_resolution_deg": 2.0,
        "metric_samples": 8_000,
    },
    "medium": {
        "sfm_quality": 2,
        "mvs_quality": "medium",
        "mesh_resolution": 100,
        "dem_resolution_m": 0.10,
        "angular_resolution_deg": 1.0,
        "metric_samples": 20_000,
    },
    "fine": {
        "sfm_quality": 3,
        "mvs_quality": "high",
        "mesh_resolution": 160,
        "dem_resolution_m": 0.06,
        "angular_resolution_deg": 0.5,
        "metric_samples": 50_000,
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_build_dir() -> Path:
    root = repo_root()
    system = platform.system()
    candidates = {
        "Windows": ["windows-source-release", "windows-release"],
        "Linux": ["linux-source-release", "linux-release"],
        "Darwin": ["macos-source-release", "macos-release"],
    }.get(system, [])
    for name in candidates:
        path = root / "build" / name
        if path.is_dir():
            return path
    return root / "build"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "opencl", "metal"], default="cpu")
    parser.add_argument("--mvs-backend", choices=["auto", "cpu", "cuda", "opencl"], default="auto")
    parser.add_argument("--point-cloud-backend", choices=["auto", "cpu", "cuda", "opencl"], default="auto")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--sfm-quality", type=int, choices=range(4))
    parser.add_argument("--mvs-quality", choices=["highest", "high", "medium", "low", "lowest"])
    parser.add_argument("--mesh-resolution", type=int)
    parser.add_argument("--dem-resolution-m", type=float)
    parser.add_argument("--metric-samples", type=int)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--pipeline-report", type=Path)
    parser.add_argument("--small-body-report", type=Path)
    return parser.parse_args(argv)


def ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def stream_process(command: list[str], log_path: Path) -> tuple[int, str]:
    environment = os.environ.copy()
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    environment.setdefault("PYTHONUTF8", "1")
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = decode_native_output(raw_line)
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            lines.append(line)
        return process.wait(), "".join(lines)


def captured_process(command: list[str], log_path: Path) -> tuple[int, str, str]:
    environment = os.environ.copy()
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        env=environment,
    )
    stdout = decode_native_output(completed.stdout)
    stderr = decode_native_output(completed.stderr)
    log_path.write_text(
        "COMMAND\n" + subprocess.list2cmdline(command) + "\n\nSTDOUT\n"
        + stdout + "\nSTDERR\n" + stderr,
        encoding="utf-8",
        newline="\n",
    )
    if stderr:
        print(stderr, end="", file=sys.stderr)
    if stdout:
        print(stdout, end="")
    return completed.returncode, stdout, stderr


def decode_native_output(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False), errors="replace")


def find_pipeline_report(output_dir: Path, output: str) -> Path:
    matches = re.findall(r"^report=(.+)$", output, flags=re.MULTILINE)
    if matches:
        path = Path(matches[-1].strip())
        if path.is_file():
            return path.resolve()
    direct = output_dir / "report.json"
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(output_dir.rglob("report.json"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"pipeline report not found under: {output_dir}")
    return candidates[-1].resolve()


def build_pipeline_command(
    executable: Path,
    dataset_dir: Path,
    pipeline_dir: Path,
    manifest: dict[str, Any],
    args: argparse.Namespace,
    run_config: dict[str, Any],
) -> list[str]:
    kind = manifest["dataset_kind"]
    profile = "orbital_object" if kind == "plascan_synthetic_3d_object" else "aerial_terrain"
    command = [
        str(executable),
        str(dataset_dir / "image_camera.lis"),
        "--output-dir", str(pipeline_dir),
        "--device", args.device,
        "--sfm-matching-algorithm", "auto_sift",
        "--lock-input-camera-poses",
        "--quality", str(args.sfm_quality if args.sfm_quality is not None else run_config["sfm_quality"]),
        "--threads", str(max(1, args.threads)),
        "--feature-max-image-dim", "0",
        "--mvs-quality", args.mvs_quality or run_config["mvs_quality"],
        "--mvs-backend", args.mvs_backend,
        "--point-cloud-backend", args.point_cloud_backend,
        "--mvs-scene-profile", profile,
        "--mvs-depth-filter", "auto",
        "--mvs-mask-dir", str(dataset_dir / "mvs_masks"),
        "--mesh-resolution", str(args.mesh_resolution or run_config["mesh_resolution"]),
        "--skip-texture",
    ]
    if kind == "plascan_synthetic_3d_object":
        command.extend(["--sfm-guided-rematching", "--skip-terrain"])
    else:
        resolution = args.dem_resolution_m or run_config["dem_resolution_m"]
        command.extend(["--dem-resolution", str(resolution)])
    return command


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run_small_body_products(
    executable: Path,
    pipeline_report_path: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    run_config: dict[str, Any],
) -> Path:
    pipeline = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
    model = pipeline.get("model", {})
    mesh_value = model.get("final_model_path") or model.get("model_ply") or model.get("mesh_ply")
    if not mesh_value:
        raise ValueError("pipeline report does not contain a reconstructed mesh")
    mesh_path = Path(mesh_value)
    if not mesh_path.is_absolute():
        mesh_path = pipeline_report_path.parent / mesh_path
    body_dir = output_dir / "small_body_products"
    body_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(executable),
        "--surface", str(mesh_path.resolve()),
        "--output-dir", str(body_dir),
        "--target", "PlaScan Synthetic Object",
        "--body-fixed-frame", "MODEL_LOCAL_BODY_FIXED",
        "--surface-unit", "m",
        "--angular-resolution-deg", str(run_config["angular_resolution_deg"]),
        "--reference-radius-m", str(manifest["object"]["reference_radius_m"]),
        "--manual-center",
        "--center-x", "0", "--center-y", "0", "--center-z", "0",
    ]
    exit_code, stdout, _ = captured_process(command, output_dir / "small_body_pipeline.log")
    if exit_code != 0:
        raise RuntimeError(f"small_body_terrain_cli failed with exit code {exit_code}")
    payload = json.loads(stdout)
    report_path = output_dir / "small_body_report.json"
    write_json(report_path, payload)
    return report_path


def run_pipeline(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path | None]:
    dataset_dir = args.dataset_dir.resolve()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    quality = manifest.get("quality_level", "medium")
    if quality not in QUALITY_RUN_CONFIG:
        raise ValueError(f"unsupported dataset quality level: {quality}")
    run_config = dict(QUALITY_RUN_CONFIG[quality])
    build_dir = (args.build_dir or default_build_dir()).resolve()
    pipeline_executable = resolve_build_executable(build_dir, "reconstruct_pipeline_cli")
    if not pipeline_executable.is_file():
        raise FileNotFoundError(f"reconstruct_pipeline_cli not found: {pipeline_executable}")
    pipeline_dir = output_dir / "pipeline"
    command = build_pipeline_command(
        pipeline_executable,
        dataset_dir,
        pipeline_dir,
        manifest,
        args,
        run_config,
    )
    write_json(output_dir / "run_config.json", {"command": command, "quality_defaults": run_config})
    exit_code, output = stream_process(command, output_dir / "pipeline.log")
    pipeline_report = find_pipeline_report(pipeline_dir, output)
    if exit_code != 0:
        raise RuntimeError(
            f"reconstruct_pipeline_cli failed with exit code {exit_code}; report={pipeline_report}"
        )

    small_body_report = None
    if manifest["dataset_kind"] == "plascan_synthetic_3d_object":
        small_body_executable = resolve_build_executable(build_dir, "small_body_terrain_cli")
        if not small_body_executable.is_file():
            raise FileNotFoundError(f"small_body_terrain_cli not found: {small_body_executable}")
        small_body_report = run_small_body_products(
            small_body_executable,
            pipeline_report,
            output_dir,
            manifest,
            run_config,
        )
    return pipeline_report, small_body_report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not (dataset_dir / "manifest.json").is_file():
        print(f"synthetic dataset manifest not found: {dataset_dir / 'manifest.json'}", file=sys.stderr)
        return 2
    try:
        if args.evaluate_only:
            if args.pipeline_report is None:
                raise ValueError("--evaluate-only requires --pipeline-report")
            output_dir.mkdir(parents=True, exist_ok=True)
            pipeline_report = args.pipeline_report.resolve()
            small_body_report = args.small_body_report.resolve() if args.small_body_report else None
        else:
            ensure_empty_output(output_dir)
            pipeline_report, small_body_report = run_pipeline(args, output_dir)

        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        quality = manifest.get("quality_level", "medium")
        samples = args.metric_samples or QUALITY_RUN_CONFIG[quality]["metric_samples"]
        report = evaluate_run(dataset_dir, pipeline_report, small_body_report, samples)
        json_path = output_dir / "synthetic_e2e_report.json"
        html_path = output_dir / "synthetic_e2e_report.html"
        write_json(json_path, report)
        write_html_report(report, html_path)
        print(f"evaluation_status={report['status']}")
        print(f"json_report={json_path}")
        print(f"html_report={html_path}")
        return 0 if report["status"] == "pass" else 1
    except (FileNotFoundError, RuntimeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"synthetic E2E failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
