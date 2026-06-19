#!/usr/bin/env python3
"""Compare ordinary BA and LiDAR-constrained BA summary JSON files."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


MAX_REPROJECTION_RMS_REGRESSION_PX = 0.25
MAX_REPROJECTION_RMS_REGRESSION_PERCENT = 10.0
MAX_OPTIMIZED_TRACK_DROP_PERCENT = 5.0
QUALITY_GATE_EPSILON = 1e-9


def as_float(obj: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = obj.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(obj: dict[str, Any], key: str, default: int = 0) -> int:
    value = obj.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def percent_reduction(before: float, after: float) -> float:
    if before <= 0.0:
        return 0.0
    return (before - after) * 100.0 / before


def percent_increase(before: float, after: float) -> float:
    if before <= 0.0:
        return 0.0 if after <= before else 100.0
    return max(0.0, (after - before) * 100.0 / before)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def valid_points_by_index(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    points: dict[int, dict[str, Any]] = {}
    for item in summary.get("points") or []:
        if not isinstance(item, dict) or not item.get("valid", False):
            continue
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        points[index] = item
    return points


def common_track_metrics(baseline: dict[str, Any], lidar: dict[str, Any]) -> dict[str, Any]:
    baseline_points = valid_points_by_index(baseline)
    lidar_points = valid_points_by_index(lidar)
    common_indices = sorted(set(baseline_points) & set(lidar_points))
    baseline_rms = [finite_float(baseline_points[index].get("rms_after")) for index in common_indices]
    lidar_rms = [finite_float(lidar_points[index].get("rms_after")) for index in common_indices]
    baseline_mean = mean(baseline_rms)
    lidar_mean = mean(lidar_rms)
    return {
        "baseline_valid_count": len(baseline_points),
        "lidar_valid_count": len(lidar_points),
        "common_valid_count": len(common_indices),
        "baseline_mean_rms_after_px": baseline_mean,
        "lidar_mean_rms_after_px": lidar_mean,
        "delta_mean_rms_after_px": lidar_mean - baseline_mean,
    }


def camera_center(camera_record: dict[str, Any]) -> list[float] | None:
    camera = camera_record.get("camera") if isinstance(camera_record, dict) else None
    center = camera.get("C") if isinstance(camera, dict) else None
    if not isinstance(center, list) or len(center) < 3:
        return None
    return [finite_float(center[index]) for index in range(3)]


def camera_rotation(camera_record: dict[str, Any]) -> list[float] | None:
    camera = camera_record.get("camera") if isinstance(camera_record, dict) else None
    rotation = camera.get("R") if isinstance(camera, dict) else None
    if not isinstance(rotation, list) or len(rotation) < 9:
        return None
    return [finite_float(rotation[index]) for index in range(9)]


def camera_records_by_image(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for item in summary.get("refined_cameras") or []:
        if not isinstance(item, dict):
            continue
        image_path = str(item.get("image_path", ""))
        if image_path:
            records[image_path] = item
    return records


def euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def mat_mul(left: list[float], right: list[float]) -> list[float]:
    return [
        sum(left[row * 3 + mid] * right[mid * 3 + col] for mid in range(3))
        for row in range(3)
        for col in range(3)
    ]


def mat_transpose(matrix: list[float]) -> list[float]:
    return [matrix[col * 3 + row] for row in range(3) for col in range(3)]


def rotation_delta_degrees(baseline_rotation: list[float], lidar_rotation: list[float]) -> float:
    delta = mat_mul(lidar_rotation, mat_transpose(baseline_rotation))
    trace = delta[0] + delta[4] + delta[8]
    cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(cosine))


def camera_pose_delta_metrics(baseline: dict[str, Any], lidar: dict[str, Any]) -> dict[str, Any]:
    baseline_cameras = camera_records_by_image(baseline)
    lidar_cameras = camera_records_by_image(lidar)
    common_images = sorted(set(baseline_cameras) & set(lidar_cameras))
    center_shifts: list[float] = []
    rotation_shifts: list[float] = []
    for image_path in common_images:
        baseline_center = camera_center(baseline_cameras[image_path])
        lidar_center = camera_center(lidar_cameras[image_path])
        if baseline_center is not None and lidar_center is not None:
            center_shifts.append(euclidean_distance(baseline_center, lidar_center))

        baseline_rotation = camera_rotation(baseline_cameras[image_path])
        lidar_rotation = camera_rotation(lidar_cameras[image_path])
        if baseline_rotation is not None and lidar_rotation is not None:
            rotation_shifts.append(rotation_delta_degrees(baseline_rotation, lidar_rotation))

    return {
        "common_camera_count": len(common_images),
        "mean_center_shift_m": mean(center_shifts),
        "max_center_shift_m": max(center_shifts) if center_shifts else 0.0,
        "mean_rotation_shift_deg": mean(rotation_shifts),
        "max_rotation_shift_deg": max(rotation_shifts) if rotation_shifts else 0.0,
    }


def build_quality_gate(
    baseline: dict[str, Any],
    lidar: dict[str, Any],
    laser_summary: dict[str, Any],
    baseline_after: float,
    lidar_after: float,
    laser_reduction: float,
) -> dict[str, Any]:
    associated_tracks = as_int(laser_summary, "associated_tracks")
    laser_constraint_count = as_int(laser_summary, "laser_constraint_count")
    baseline_optimized = as_int(baseline, "optimized_count")
    lidar_optimized = as_int(lidar, "optimized_count")

    reprojection_regression_px = max(0.0, lidar_after - baseline_after)
    reprojection_regression_percent = percent_increase(baseline_after, lidar_after)
    optimized_track_drop = max(0, baseline_optimized - lidar_optimized)
    optimized_track_drop_percent = (
        optimized_track_drop * 100.0 / baseline_optimized
        if baseline_optimized > 0
        else 0.0
    )

    failure_codes: list[str] = []
    failure_reasons: list[str] = []
    if laser_constraint_count <= 0 or associated_tracks <= 0:
        failure_codes.append("no_laser_constraints")
        failure_reasons.append("LiDAR BA did not keep any active laser constraints.")
    if laser_reduction <= 0.0:
        failure_codes.append("laser_rms_not_reduced")
        failure_reasons.append("LiDAR point-to-plane RMS did not decrease.")
    if (
        reprojection_regression_px > MAX_REPROJECTION_RMS_REGRESSION_PX + QUALITY_GATE_EPSILON
        or reprojection_regression_percent > MAX_REPROJECTION_RMS_REGRESSION_PERCENT + QUALITY_GATE_EPSILON
    ):
        failure_codes.append("reprojection_rms_regressed")
        failure_reasons.append("LiDAR BA increased reprojection RMS beyond the acceptance threshold.")
    if optimized_track_drop_percent > MAX_OPTIMIZED_TRACK_DROP_PERCENT + QUALITY_GATE_EPSILON:
        failure_codes.append("optimized_tracks_dropped")
        failure_reasons.append("LiDAR BA optimized materially fewer tracks than the baseline run.")

    return {
        "passed": not failure_codes,
        "status": "pass" if not failure_codes else "fail",
        "failure_codes": failure_codes,
        "failure_reasons": failure_reasons,
        "thresholds": {
            "max_reprojection_rms_regression_px": MAX_REPROJECTION_RMS_REGRESSION_PX,
            "max_reprojection_rms_regression_percent": MAX_REPROJECTION_RMS_REGRESSION_PERCENT,
            "max_optimized_track_drop_percent": MAX_OPTIMIZED_TRACK_DROP_PERCENT,
        },
        "metrics": {
            "reprojection_rms_regression_px": reprojection_regression_px,
            "reprojection_rms_regression_percent": reprojection_regression_percent,
            "optimized_track_drop": optimized_track_drop,
            "optimized_track_drop_percent": optimized_track_drop_percent,
        },
    }


def compare_summaries(baseline: dict[str, Any], lidar: dict[str, Any]) -> dict[str, Any]:
    laser_summary = lidar.get("laser_constraints_summary") or {}
    baseline_after = as_float(baseline, "mean_rms_after")
    lidar_after = as_float(lidar, "mean_rms_after")
    laser_before = as_float(laser_summary, "laser_rms_before_m")
    laser_after = as_float(laser_summary, "laser_rms_after_m")
    laser_reduction = laser_before - laser_after
    associated_tracks = as_int(laser_summary, "associated_tracks")
    laser_constraint_count = as_int(laser_summary, "laser_constraint_count")
    quality_gate = build_quality_gate(
        baseline,
        lidar,
        laser_summary,
        baseline_after,
        lidar_after,
        laser_reduction,
    )

    if laser_constraint_count <= 0:
        verdict = "LiDAR BA did not report active laser constraints; inspect input association and options."
    elif laser_reduction > 0.0:
        verdict = "LiDAR BA reduced point-to-plane residuals; inspect reprojection RMS and pose drift before accepting."
    else:
        verdict = "LiDAR BA did not reduce point-to-plane residuals; tune association distance, weights, or normals."

    return {
        "baseline": {
            "track_count": as_int(baseline, "track_count"),
            "optimized_count": as_int(baseline, "optimized_count"),
            "mean_rms_before_px": as_float(baseline, "mean_rms_before"),
            "mean_rms_after_px": baseline_after,
        },
        "lidar": {
            "track_count": as_int(lidar, "track_count"),
            "optimized_count": as_int(lidar, "optimized_count"),
            "mean_rms_before_px": as_float(lidar, "mean_rms_before"),
            "mean_rms_after_px": lidar_after,
            "associated_tracks": associated_tracks,
            "laser_constraint_count": laser_constraint_count,
            "laser_rms_before_m": laser_before,
            "laser_rms_after_m": laser_after,
            "laser_median_before_m": as_float(laser_summary, "laser_median_before_m"),
            "laser_median_after_m": as_float(laser_summary, "laser_median_after_m"),
        },
        "deltas": {
            "reprojection_rms_after_px": lidar_after - baseline_after,
            "optimized_count_delta": as_int(lidar, "optimized_count") - as_int(baseline, "optimized_count"),
            "laser_rms_reduction_m": laser_reduction,
            "laser_rms_reduction_percent": percent_reduction(laser_before, laser_after),
        },
        "common_tracks": common_track_metrics(baseline, lidar),
        "camera_pose_delta": camera_pose_delta_metrics(baseline, lidar),
        "quality_gate": quality_gate,
        "verdict": verdict,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_markdown_report(comparison: dict[str, Any], output_path: Path) -> None:
    baseline = comparison["baseline"]
    lidar = comparison["lidar"]
    deltas = comparison["deltas"]
    common_tracks = comparison["common_tracks"]
    camera_delta = comparison["camera_pose_delta"]
    quality_gate = comparison["quality_gate"]
    lines = [
        "# LiDAR BA A/B Comparison",
        "",
        f"Quality gate: {quality_gate['status'].upper()}",
        "",
        "",
        "| Metric | Baseline BA | LiDAR BA | Delta |",
        "|---|---:|---:|---:|",
        (
            f"| Mean reprojection RMS after (px) | {baseline['mean_rms_after_px']:.6f} | "
            f"{lidar['mean_rms_after_px']:.6f} | {deltas['reprojection_rms_after_px']:.6f} |"
        ),
        (
            f"| Optimized tracks | {baseline['optimized_count']} | "
            f"{lidar['optimized_count']} | {deltas['optimized_count_delta']} |"
        ),
        (
            f"| LiDAR RMS (m) |  | {lidar['laser_rms_before_m']:.6f} -> "
            f"{lidar['laser_rms_after_m']:.6f} | -{deltas['laser_rms_reduction_m']:.6f} |"
        ),
        (
            f"| LiDAR median (m) |  | {lidar['laser_median_before_m']:.6f} -> "
            f"{lidar['laser_median_after_m']:.6f} |  |"
        ),
        (
            f"| Associated tracks |  | {lidar['associated_tracks']} / "
            f"{lidar['laser_constraint_count']} constraints |  |"
        ),
        (
            f"| Common valid track RMS after (px) | {common_tracks['baseline_mean_rms_after_px']:.6f} | "
            f"{common_tracks['lidar_mean_rms_after_px']:.6f} | "
            f"{common_tracks['delta_mean_rms_after_px']:.6f} |"
        ),
        (
            f"| Valid tracks | {common_tracks['baseline_valid_count']} | "
            f"{common_tracks['lidar_valid_count']} | "
            f"common {common_tracks['common_valid_count']} |"
        ),
        (
            f"| Camera center shift (m) |  | mean {camera_delta['mean_center_shift_m']:.6f}, "
            f"max {camera_delta['max_center_shift_m']:.6f} |  |"
        ),
        (
            f"| Camera rotation shift (deg) |  | mean {camera_delta['mean_rotation_shift_deg']:.6f}, "
            f"max {camera_delta['max_rotation_shift_deg']:.6f} |  |"
        ),
        "",
        f"Verdict: {comparison['verdict']}",
        "",
    ]
    if quality_gate["failure_reasons"]:
        lines.extend(["Quality gate failures:", ""])
        lines.extend(f"- {reason}" for reason in quality_gate["failure_reasons"])
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline BA and LiDAR-constrained BA JSON summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-json", type=Path, required=True, help="ordinary BA summary JSON")
    parser.add_argument("--lidar-json", type=Path, required=True, help="LiDAR-constrained BA summary JSON")
    parser.add_argument("--output-json", type=Path, required=True, help="comparison JSON output")
    parser.add_argument("--output-md", type=Path, help="optional Markdown report output")
    parser.add_argument(
        "--fail-on-quality-gate",
        action="store_true",
        help="return a non-zero exit code when the LiDAR BA quality gate fails",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        comparison = compare_summaries(load_json(args.baseline_json), load_json(args.lidar_json))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"failed to compare BA summaries: {exc}", file=sys.stderr)
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown_report(comparison, args.output_md)
    print(f"wrote: {args.output_json}")
    if args.output_md:
        print(f"wrote: {args.output_md}")
    if args.fail_on_quality_gate and not comparison["quality_gate"]["passed"]:
        print("quality gate failed: LiDAR BA result should not be accepted automatically", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
