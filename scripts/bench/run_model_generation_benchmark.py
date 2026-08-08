#!/usr/bin/env python3
"""Benchmark serial and automatic-worker model generation on fixed depth maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


TIMING_FIELDS = (
    "depth_frame_discovery_elapsed_ms",
    "depth_frame_load_elapsed_ms",
    "depth_tsdf_build_elapsed_ms",
    "mesh_cleanup_elapsed_ms",
    "mesh_simplification_elapsed_ms",
    "mesh_colorization_elapsed_ms",
    "post_integration_elapsed_ms",
    "model_core_elapsed_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-input, no-interpolation model generation with one CPU "
            "worker and with the automatic logical-thread-minus-two budget."
        )
    )
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--depth-map-dir", required=True, type=Path)
    parser.add_argument("--settings-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 2),
    )
    parser.add_argument(
        "--allow-matched-quality-gate-failure",
        action="store_true",
        help=(
            "accept two runs that both reach the same geometry but are rejected "
            "by the final quality gate"
        ),
    )
    return parser.parse_args()


def find_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    result_candidates = [candidate for candidate in candidates if "ok" in candidate]
    if result_candidates:
        return result_candidates[-1]
    if not candidates:
        raise ValueError("mesh CLI did not emit a JSON result")
    return candidates[-1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_case(
    executable: Path,
    depth_map_dir: Path,
    settings_path: Path,
    output_dir: Path,
    label: str,
    repeat_index: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    completed = subprocess.run(
        [
            str(executable),
            "--source-data",
            "depth_maps",
            "--depth-map-dir",
            str(depth_map_dir),
            "--output-dir",
            str(output_dir),
            "--settings-json",
            str(settings_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    wall_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
    payload = find_json_object(completed.stdout + "\n" + completed.stderr)
    result: dict[str, Any] = {
        "label": label,
        "repeat": repeat_index,
        "return_code": completed.returncode,
        "wall_ms": wall_ms,
        "ok": bool(payload.get("ok", False)),
        "error": payload.get("error", ""),
        "discovered_depth_frame_count": payload.get(
            "discovered_depth_frame_count"
        ),
        "loaded_depth_frame_count": payload.get("loaded_depth_frame_count"),
        "depth_frame_load_worker_count": payload.get(
            "depth_frame_load_worker_count"
        ),
        "mesh_colorization_worker_count": payload.get(
            "mesh_colorization_worker_count"
        ),
        "vertex_count": payload.get("vertex_count"),
        "face_count": payload.get("face_count"),
        "final_depth_completeness_gate_passed": payload.get(
            "final_depth_completeness_gate_passed"
        ),
    }
    for field in TIMING_FIELDS:
        result[field] = payload.get(field)
    model_path_value = payload.get("final_model_path") or payload.get("model_ply")
    if model_path_value:
        model_path = Path(model_path_value)
        if model_path.is_file():
            result["model_sha256"] = sha256_file(model_path)
    return result


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


def summarize_case(label: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "label": label,
        "repeat_count": len(runs),
        "all_ok": all(bool(run["ok"]) for run in runs),
        "return_codes": sorted({int(run["return_code"]) for run in runs}),
        "median_wall_ms": median([float(run["wall_ms"]) for run in runs]),
    }
    for field in TIMING_FIELDS:
        summary[f"median_{field}"] = median(
            [float(run[field]) for run in runs if run.get(field) is not None]
        )
    last = runs[-1]
    for field in (
        "discovered_depth_frame_count",
        "loaded_depth_frame_count",
        "depth_frame_load_worker_count",
        "mesh_colorization_worker_count",
        "vertex_count",
        "face_count",
        "final_depth_completeness_gate_passed",
        "model_sha256",
        "error",
    ):
        if field in last:
            summary[field] = last[field]
    return summary


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    serial = report["summary"]["serial"]
    parallel = report["summary"]["parallel"]
    lines = [
        "# Model generation benchmark",
        "",
        "| Case | Workers | Wall (ms) | Frame load (ms) | TSDF (ms) | Color (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in (serial, parallel):
        lines.append(
            "| {label} | {workers} | {wall:.3f} | {load} | {tsdf} | {color} |".format(
                label=case["label"],
                workers=case.get("depth_frame_load_worker_count", ""),
                wall=float(case["median_wall_ms"]),
                load=case.get("median_depth_frame_load_elapsed_ms", ""),
                tsdf=case.get("median_depth_tsdf_build_elapsed_ms", ""),
                color=case.get("median_mesh_colorization_elapsed_ms", ""),
            )
        )
    lines.extend(
        [
            "",
            f"Wall speedup: **{report['wall_speedup']:.3f}x**",
            f"Frame-load speedup: **{report['frame_load_speedup']:.3f}x**",
            f"Geometry equivalent: **{report['geometry_equivalent']}**",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.parallel_workers < 1:
        raise ValueError("--parallel-workers must be at least 1")
    for required_path in (args.exe, args.depth_map_dir, args.settings_json):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    root_settings = json.loads(args.settings_json.read_text(encoding="utf-8"))
    settings = root_settings.get("generate_model", root_settings)
    if settings.get("interpolation") != "disabled":
        raise ValueError("benchmark requires interpolation=disabled")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_settings_paths: dict[str, Path] = {}
    for label, worker_count in (("serial", 1), ("parallel", args.parallel_workers)):
        case_root = json.loads(json.dumps(root_settings))
        case_settings = case_root.get("generate_model", case_root)
        case_settings["threads"] = worker_count
        settings_path = args.output_dir / f"settings_{label}.json"
        settings_path.write_text(
            json.dumps(case_root, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        case_settings_paths[label] = settings_path

    runs: list[dict[str, Any]] = []
    for repeat_index in range(1, args.repeat + 1):
        for label in ("serial", "parallel"):
            run_output = args.output_dir / f"{label}_{repeat_index}"
            run = run_case(
                args.exe,
                args.depth_map_dir,
                case_settings_paths[label],
                run_output,
                label,
                repeat_index,
            )
            runs.append(run)
            print(
                f"{label} repeat={repeat_index}: wall={run['wall_ms']:.3f} ms, "
                f"load={run.get('depth_frame_load_elapsed_ms')} ms, "
                f"return={run['return_code']}"
            )

    serial_runs = [run for run in runs if run["label"] == "serial"]
    parallel_runs = [run for run in runs if run["label"] == "parallel"]
    serial = summarize_case("serial", serial_runs)
    parallel = summarize_case("parallel", parallel_runs)
    geometry_keys = ("vertex_count", "face_count", "model_sha256")
    geometry_equivalent = all(
        serial.get(key) == parallel.get(key)
        for key in geometry_keys
        if key in serial or key in parallel
    )
    wall_speedup = float(serial["median_wall_ms"]) / float(
        parallel["median_wall_ms"]
    )
    frame_load_speedup = float(
        serial["median_depth_frame_load_elapsed_ms"]
    ) / float(parallel["median_depth_frame_load_elapsed_ms"])
    report = {
        "schema_version": 1,
        "depth_map_dir": str(args.depth_map_dir.resolve()),
        "settings_json": str(args.settings_json.resolve()),
        "parallel_workers": args.parallel_workers,
        "runs": runs,
        "summary": {"serial": serial, "parallel": parallel},
        "wall_speedup": wall_speedup,
        "frame_load_speedup": frame_load_speedup,
        "geometry_equivalent": geometry_equivalent,
    }
    (args.output_dir / "benchmark_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_markdown(args.output_dir / "benchmark_results.md", report)

    both_ok = bool(serial["all_ok"] and parallel["all_ok"])
    matched_quality_failure = (
        args.allow_matched_quality_gate_failure
        and not both_ok
        and geometry_equivalent
        and serial.get("final_depth_completeness_gate_passed") is False
        and parallel.get("final_depth_completeness_gate_passed") is False
    )
    return 0 if geometry_equivalent and (both_ok or matched_quality_failure) else 1


if __name__ == "__main__":
    raise SystemExit(main())
