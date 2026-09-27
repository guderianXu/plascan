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
    "cameras",
    "tracks",
    "views",
    "repetition",
    "phase",
    "backend",
    "available",
    "requested",
    "used",
    "status",
    "usable",
    "gpu",
    "fallback",
    "selection_reason",
    "backend_message",
    "valid_ratio",
    "rms_before",
    "rms_after",
    "quality_rejected",
    "initial_cost",
    "final_cost",
    "linear_solver",
    "device",
    "linear_iterations",
    "schur_pattern_builds",
    "schur_pattern_reuses",
    "schur_on_device",
    "mixed_precision_used",
    "assembly_seconds",
    "linear_solve_seconds",
    "back_substitution_seconds",
    "solve_seconds",
    "total_seconds",
    "wall_seconds",
]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_metric_line(line: str) -> dict[str, str] | None:
    parts = [part.strip() for part in line.split(",")]
    if not parts or parts[0] != "run":
        return None
    row: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            row[key] = value
    return row if row.get("backend") else None


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
    repetitions: int,
    wanted_backends: set[str],
    device_index: int,
    mixed_precision: bool,
) -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            str(exe),
            str(camera_count),
            str(track_count),
            str(views_per_track),
            str(iterations),
            str(threads),
            str(repetitions),
            ",".join(sorted(wanted_backends)),
            str(device_index),
            "1" if mixed_precision else "0",
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
        row["cameras"] = str(camera_count)
        row["tracks"] = str(track_count)
        row["views"] = str(views_per_track)
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
        wall_values = [value for row in group_rows if (value := parse_float(row, "wall_seconds")) is not None]
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
                "last_status": last.get("status", ""),
                "last_solver": last.get("linear_solver", ""),
                "last_quality_rejected": last.get("quality_rejected", ""),
                "last_backend_reason": last.get("selection_reason", ""),
            }
        )

    return {"case_count": len(cases), "cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PlaBundle backend benchmark.")
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--cases", default="medium", help="逗号分隔: small,medium,large")
    parser.add_argument(
        "--backends",
        default="plamatrix_cpu,plamatrix_cuda,plamatrix_vulkan,plamatrix_opencl,auto",
    )
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--iterations", default=8, type=int)
    parser.add_argument("--threads", default=32, type=int)
    parser.add_argument("--device", default=0, type=int, help="PlaMatrix CUDA/OpenCL 设备索引")
    parser.add_argument("--mixed-precision", action="store_true", help="请求受保护的 FP32 PCG 初值")
    args = parser.parse_args()

    case_names = split_csv(args.cases)
    unknown_cases = [case for case in case_names if case not in CASES]
    if unknown_cases:
        raise SystemExit(f"未知 case: {', '.join(unknown_cases)}")
    wanted_backends = set(split_csv(args.backends))
    if not wanted_backends:
        raise SystemExit("至少需要一个后端")

    all_rows: list[dict[str, str]] = []
    for case_name in case_names:
        camera_count, track_count, views_per_track = CASES[case_name]
        all_rows.extend(
            run_one(
                exe=args.exe,
                case_name=case_name,
                camera_count=camera_count,
                track_count=track_count,
                views_per_track=views_per_track,
                iterations=max(1, args.iterations),
                threads=max(0, args.threads),
                repetitions=max(1, args.repeat),
                wanted_backends=wanted_backends,
                device_index=max(0, args.device),
                mixed_precision=args.mixed_precision,
            )
        )

    if not all_rows:
        raise SystemExit("benchmark 未输出任何 run 记录，请确认 --exe 指向 plabundle_benchmark")

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
