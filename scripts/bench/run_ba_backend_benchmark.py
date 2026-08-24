from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path
from typing import Iterable


CASES: dict[str, tuple[int, int, int]] = {
    "small": (20, 600, 4),
    "medium": (80, 3000, 8),
    "large": (140, 9000, 10),
}

DEFAULT_FIELDS = [
    "case",
    "repeat",
    "backend",
    "requested",
    "used",
    "gpu",
    "fallback",
    "solver",
    "observations",
    "tracks",
    "optimized",
    "valid_ratio",
    "rms_before",
    "rms_after",
    "quality_rejected",
    "backend_reason",
    "quality_message",
    "plamatrix_initial_cost",
    "plamatrix_final_cost",
    "plamatrix_accepted_steps",
    "plamatrix_rejected_steps",
    "plamatrix_rejected_initial_tracks",
    "plamatrix_linear_solver",
    "plamatrix_device",
    "plamatrix_linear_iterations",
    "plamatrix_schur_pattern_builds",
    "plamatrix_schur_pattern_reuses",
    "plamatrix_schur_assembly_on_device",
    "plamatrix_schur_assembly_seconds",
    "plamatrix_linear_solve_seconds",
    "native_pcg_iterations",
    "native_linear_residual",
    "native_active_observations",
    "native_upload_seconds",
    "native_kernel_seconds",
    "native_download_seconds",
    "native_host_cost_seconds",
    "native_device_select_seconds",
    "native_staging_seconds",
    "native_release_seconds",
    "setup_seconds",
    "solve_seconds",
    "total_seconds",
    "seconds",
]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_metric_line(line: str) -> dict[str, str] | None:
    parts = [part.strip() for part in line.split(",")]
    if not parts or parts[0] == "dataset":
        return None
    row = {"backend": parts[0]}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            row[key] = value
    return row


def parse_float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def median(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return statistics.median(values)


def run_one(
    exe: Path,
    case_name: str,
    camera_count: int,
    track_count: int,
    views_per_track: int,
    iterations: int,
    threads: int,
    refine_pose: bool,
    repeat_index: int,
    wanted_backends: set[str],
) -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            str(exe),
            str(camera_count),
            str(track_count),
            str(views_per_track),
            str(iterations),
            str(threads),
            "1" if refine_pose else "0",
            ",".join(sorted(wanted_backends)),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        row = parse_metric_line(line)
        if not row or row["backend"] not in wanted_backends:
            continue
        row["case"] = case_name
        row["repeat"] = str(repeat_index)
        rows.append(row)
    print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    return rows


def build_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["case"], row["backend"]), []).append(row)

    cases: list[dict[str, object]] = []
    for (case_name, backend), group_rows in sorted(grouped.items()):
        total_values = [value for row in group_rows if (value := parse_float(row, "total_seconds")) is not None]
        wall_values = [value for row in group_rows if (value := parse_float(row, "seconds")) is not None]
        rms_values = [value for row in group_rows if (value := parse_float(row, "rms_after")) is not None]
        valid_values = [value for row in group_rows if (value := parse_float(row, "valid_ratio")) is not None]
        last = group_rows[-1]
        cases.append(
            {
                "case": case_name,
                "backend": backend,
                "repeat_count": len(group_rows),
                "median_total_seconds": median(total_values),
                "median_wall_seconds": median(wall_values),
                "median_rms_after": median(rms_values),
                "median_valid_ratio": median(valid_values),
                "last_used_backend": last.get("used", ""),
                "last_gpu": last.get("gpu", ""),
                "last_fallback": last.get("fallback", ""),
                "last_solver": last.get("solver", ""),
                "last_quality_rejected": last.get("quality_rejected", ""),
                "last_backend_reason": last.get("backend_reason", ""),
            }
        )

    return {"case_count": len(cases), "cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PlaScan BA backend benchmark.")
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--cases", default="medium", help="逗号分隔: small,medium,large")
    parser.add_argument(
        "--backends",
        default=(
            "legacy_cpu,plamatrix_cpu,plamatrix_cuda,plamatrix_opencl,"
            "auto"
        ),
    )
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--iterations", default=8, type=int)
    parser.add_argument("--threads", default=32, type=int)
    parser.add_argument(
        "--refine-pose",
        action="store_true",
        help="同时优化相机位姿。benchmark 会通过公共 BA 校验补足 gauge 锚定。",
    )
    args = parser.parse_args()

    case_names = split_csv(args.cases)
    unknown_cases = [case for case in case_names if case not in CASES]
    if unknown_cases:
        raise SystemExit(f"未知 case: {', '.join(unknown_cases)}")
    wanted_backends = set(split_csv(args.backends))

    all_rows: list[dict[str, str]] = []
    for case_name in case_names:
        camera_count, track_count, views_per_track = CASES[case_name]
        for repeat_index in range(1, max(1, args.repeat) + 1):
            all_rows.extend(
                run_one(
                    exe=args.exe,
                    case_name=case_name,
                    camera_count=camera_count,
                    track_count=track_count,
                    views_per_track=views_per_track,
                    iterations=max(1, args.iterations),
                    threads=max(1, args.threads),
                    refine_pose=args.refine_pose,
                    repeat_index=repeat_index,
                    wanted_backends=wanted_backends,
                )
            )

    fieldnames = [field for field in DEFAULT_FIELDS if any(field in row for row in all_rows)]
    extras = sorted({key for row in all_rows for key in row if key not in fieldnames})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames + extras)
        writer.writeheader()
        writer.writerows(all_rows)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(build_summary(all_rows), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
