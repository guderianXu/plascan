#!/usr/bin/env python3
"""Prepare a temporary PlaScan .plascan project for MUN-FRL LiDAR BA tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class PrepareProjectResult:
    project_path: Path
    summary_path: Path
    image_count: int
    match_count: int


@dataclass
class RigidTransform:
    parent_frame: str
    child_frame: str
    rotation: list[float]
    translation: list[float]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_yaml_scalar(text: str, key: str, default: float | None = None) -> float:
    match = re.search(rf"^{re.escape(key)}:\s*([^\r\n#]+)", text, flags=re.MULTILINE)
    if not match:
        if default is not None:
            return default
        raise ValueError(f"camera info missing scalar: {key}")
    return float(match.group(1).strip())


def parse_yaml_list(text: str, key: str) -> list[float]:
    match = re.search(rf"^{re.escape(key)}:\s*\[([^\]]*)\]", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"camera info missing list: {key}")
    values = [part.strip() for part in match.group(1).split(",")]
    return [float(value) for value in values if value]


def parse_camera_info(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    k = parse_yaml_list(text, "K")
    d = parse_yaml_list(text, "D")
    if len(k) < 9:
        raise ValueError(f"K must contain 9 values: {path}")
    while len(d) < 5:
        d.append(0.0)

    return {
        "width": int(parse_yaml_scalar(text, "width", 0.0)),
        "height": int(parse_yaml_scalar(text, "height", 0.0)),
        "fu": k[0],
        "fv": k[4],
        "cu": k[2],
        "cv": k[5],
        "k1": d[0],
        "k2": d[1],
        "p1": d[2],
        "p2": d[3],
        "k3": d[4],
    }


def read_trajectory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append(
                    {
                        "stamp_ns": int(row["header_stamp_ns"]),
                        "frame_id": row.get("frame_id", ""),
                        "child_frame_id": row.get("child_frame_id", ""),
                        "px": float(row["px"]),
                        "py": float(row["py"]),
                        "pz": float(row["pz"]),
                        "qx": float(row["qx"]),
                        "qy": float(row["qy"]),
                        "qz": float(row["qz"]),
                        "qw": float(row["qw"]),
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid trajectory row in {path}: {row}") from exc
    if not rows:
        raise ValueError(f"trajectory is empty: {path}")
    rows.sort(key=lambda item: item["stamp_ns"])
    return rows


def read_tf_static(path: Path) -> list[RigidTransform]:
    transforms: list[RigidTransform] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                parent_frame = row["parent_frame_id"].strip()
                child_frame = row["child_frame_id"].strip()
                if not parent_frame or not child_frame:
                    continue
                transforms.append(
                    RigidTransform(
                        parent_frame=parent_frame,
                        child_frame=child_frame,
                        rotation=quaternion_to_rotation(
                            float(row["qx"]),
                            float(row["qy"]),
                            float(row["qz"]),
                            float(row["qw"]),
                        ),
                        translation=[float(row["tx"]), float(row["ty"]), float(row["tz"])],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid tf_static row in {path}: {row}") from exc
    if not transforms:
        raise ValueError(f"tf_static is empty: {path}")
    return transforms


def nearest_pose(rows: list[dict[str, Any]], stamp_ns: int) -> dict[str, Any]:
    return min(rows, key=lambda item: abs(item["stamp_ns"] - stamp_ns))


def quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> list[float]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    return [
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy - qz * qw),
        2.0 * (qx * qz + qy * qw),
        2.0 * (qx * qy + qz * qw),
        1.0 - 2.0 * (qx * qx + qz * qz),
        2.0 * (qy * qz - qx * qw),
        2.0 * (qx * qz - qy * qw),
        2.0 * (qy * qz + qx * qw),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ]


def mat_mul(left: list[float], right: list[float]) -> list[float]:
    return [
        sum(left[row * 3 + mid] * right[mid * 3 + col] for mid in range(3))
        for row in range(3)
        for col in range(3)
    ]


def mat_vec_mul(matrix: list[float], vector: list[float]) -> list[float]:
    return [
        sum(matrix[row * 3 + col] * vector[col] for col in range(3))
        for row in range(3)
    ]


def mat_transpose(matrix: list[float]) -> list[float]:
    return [matrix[col * 3 + row] for row in range(3) for col in range(3)]


def transform_from_pose_row(pose: dict[str, Any]) -> RigidTransform:
    return RigidTransform(
        parent_frame=str(pose.get("frame_id", "")),
        child_frame=str(pose.get("child_frame_id", "")),
        rotation=quaternion_to_rotation(pose["qx"], pose["qy"], pose["qz"], pose["qw"]),
        translation=[pose["px"], pose["py"], pose["pz"]],
    )


def compose_transform(parent_to_child: RigidTransform, child_to_grandchild: RigidTransform) -> RigidTransform:
    rotated_translation = mat_vec_mul(parent_to_child.rotation, child_to_grandchild.translation)
    return RigidTransform(
        parent_frame=parent_to_child.parent_frame,
        child_frame=child_to_grandchild.child_frame,
        rotation=mat_mul(parent_to_child.rotation, child_to_grandchild.rotation),
        translation=[
            parent_to_child.translation[index] + rotated_translation[index]
            for index in range(3)
        ],
    )


def invert_transform(transform: RigidTransform) -> RigidTransform:
    rotation_inverse = mat_transpose(transform.rotation)
    neg_translation = [-value for value in transform.translation]
    return RigidTransform(
        parent_frame=transform.child_frame,
        child_frame=transform.parent_frame,
        rotation=rotation_inverse,
        translation=mat_vec_mul(rotation_inverse, neg_translation),
    )


def find_body_to_camera_transform(
    transforms: list[RigidTransform],
    camera_frame: str,
    body_frame: str,
) -> RigidTransform:
    for transform in transforms:
        if transform.parent_frame == body_frame and transform.child_frame == camera_frame:
            return transform
    for transform in transforms:
        if transform.parent_frame == camera_frame and transform.child_frame == body_frame:
            return invert_transform(transform)
    available = ", ".join(f"{item.parent_frame}->{item.child_frame}" for item in transforms)
    raise ValueError(
        f"tf_static missing transform between body frame '{body_frame}' and camera frame "
        f"'{camera_frame}'. Available: {available}"
    )


def normalize_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def image_key(path: str | Path) -> tuple[str, str]:
    info = Path(path)
    return info.name.lower(), info.stem.lower()


def resolve_image_token(token: str, images: list[dict[str, Any]]) -> dict[str, Any] | None:
    token_name, token_stem = image_key(token)
    token_path = Path(token)
    token_resolved = str(token_path.resolve()).lower() if token_path.is_absolute() else ""
    for image in images:
        image_path = Path(image["path"])
        name, stem = image_key(image_path)
        if token_resolved and str(image_path.resolve()).lower() == token_resolved:
            return image
        if token_name == name or token_stem == stem:
            return image
    return None


def camera_json(
    camera_info: dict[str, Any],
    pose: dict[str, Any],
    camera_transform: RigidTransform | None = None,
    pose_source: str = "mun_frl_camera_info_plus_odometry",
    tf_static_camera_frame: str = "",
    tf_static_body_frame: str = "",
) -> dict[str, Any]:
    if camera_transform is None:
        camera_transform = transform_from_pose_row(pose)
    return {
        "model": "tsai",
        "source": pose_source,
        "pose_source": pose_source,
        "width": camera_info["width"],
        "height": camera_info["height"],
        "fu": camera_info["fu"],
        "fv": camera_info["fv"],
        "cu": camera_info["cu"],
        "cv": camera_info["cv"],
        "pitch": 1.0,
        "intrinsics_unit": "pixel",
        "camera_center_unit": "m",
        "u_direction": 1,
        "v_direction": 1,
        "depth_axis_flipped": False,
        "k1": camera_info["k1"],
        "k2": camera_info["k2"],
        "k3": camera_info["k3"],
        "p1": camera_info["p1"],
        "p2": camera_info["p2"],
        "C": camera_transform.translation,
        "R": camera_transform.rotation,
        "pose_stamp_ns": pose["stamp_ns"],
        "odometry_parent_frame": pose.get("frame_id", ""),
        "odometry_child_frame": pose.get("child_frame_id", ""),
        "camera_frame": camera_transform.child_frame,
        "world_frame": camera_transform.parent_frame,
        "tf_static_camera_frame": tf_static_camera_frame,
        "tf_static_body_frame": tf_static_body_frame,
    }


def build_images(
    plan: dict[str, Any],
    camera_info: dict[str, Any],
    trajectory: list[dict[str, Any]],
    body_to_camera: RigidTransform | None = None,
    camera_frame: str = "",
    body_frame: str = "",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, item in enumerate(plan.get("images", [])):
        image_path = Path(item["path"]).resolve()
        stamp_ns = int(item.get("stamp_ns", item.get("header_stamp_ns", 0)))
        pose = nearest_pose(trajectory, stamp_ns)
        world_to_body = transform_from_pose_row(pose)
        world_to_camera = world_to_body
        pose_source = "mun_frl_camera_info_plus_odometry"
        if body_to_camera is not None:
            world_to_camera = compose_transform(world_to_body, body_to_camera)
            pose_source = "mun_frl_camera_info_odometry_tf_static"
        image = {
            "index": index,
            "path": str(image_path),
            "name": image_path.name,
            "relative_path": item.get("relative_path", image_path.name),
            "stamp_ns": stamp_ns,
            "nearest_pose_stamp_ns": pose["stamp_ns"],
            "nearest_pose_dt_ms": (pose["stamp_ns"] - stamp_ns) / 1_000_000.0,
            "camera": camera_json(
                camera_info,
                pose,
                camera_transform=world_to_camera,
                pose_source=pose_source,
                tf_static_camera_frame=camera_frame,
                tf_static_body_frame=body_frame,
            ),
        }
        output.append(image)
    if len(output) < 2:
        raise ValueError("benchmark plan must contain at least 2 images")
    return output


def sidecar_tokens(sidecar: dict[str, Any], sidecar_path: Path) -> tuple[str, str]:
    left = sidecar.get("image0_path") or sidecar.get("image0_name")
    right = sidecar.get("image1_path") or sidecar.get("image1_name")
    if left and right:
        return str(left), str(right)

    stem = sidecar_path.name
    if stem.endswith(".match.json"):
        stem = stem[: -len(".match.json")]
    if "__" not in stem:
        return "", ""
    left_stem, right_stem = stem.split("__", 1)
    return left_stem, right_stem


def build_ipmatch_results(matches_dir: Path, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for sidecar_path in sorted(matches_dir.glob("*.match.json")):
        sidecar = read_json(sidecar_path)
        if int(sidecar.get("feature_format_version", 0)) < 2:
            continue
        left_token, right_token = sidecar_tokens(sidecar, sidecar_path)
        left = resolve_image_token(left_token, images)
        right = resolve_image_token(right_token, images)
        if left is None or right is None or left["path"] == right["path"]:
            continue

        match_file = sidecar.get("match_file")
        if not match_file:
            match_file = str(sidecar_path)[: -len(".json")]
        record = {
            "image0": left["path"],
            "image1": right["path"],
            "output": normalize_path(match_file),
            "num_matches": int(sidecar.get("num_matches", len(sidecar.get("matched_indices0", [])))),
            "settings": {
                "image_files": [left["path"], right["path"]],
                "sidecar_json": normalize_path(sidecar_path),
                "feature_format_version": 2,
                "match_algorithm": sidecar.get("match_algorithm", "bf"),
                "feature_algorithm": sidecar.get("feature_algorithm", "sift"),
            },
        }
        results.append(record)
    if not results:
        raise ValueError(f"no BA-ready match sidecars found in {matches_dir}")
    return results


def write_plascan(project_path: Path, project_files: dict[str, Any], project_results: dict[str, Any]) -> None:
    manifest = {
        "type": "PlaScanProject",
        "format_version": 2,
        "created_by": "prepare_mun_frl_lidar_ba_project.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    project_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(project_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr("project_files.json", json.dumps(project_files, ensure_ascii=False, indent=2))
        archive.writestr("project_results.json", json.dumps(project_results, ensure_ascii=False, indent=2))
        archive.writestr("project_config.json", json.dumps({}, ensure_ascii=False, indent=2))


def prepare_project(
    benchmark_plan: Path,
    camera_info: Path,
    trajectory: Path,
    matches_dir: Path,
    output_dir: Path,
    project_name: str = "mun_frl_lidar_ba",
    tf_static: Path | None = None,
    camera_frame: str = "camera",
    body_frame: str = "imu_link",
) -> PrepareProjectResult:
    benchmark_plan = Path(benchmark_plan)
    camera_info = Path(camera_info)
    trajectory = Path(trajectory)
    matches_dir = Path(matches_dir)
    output_dir = Path(output_dir)
    tf_static = Path(tf_static) if tf_static else None

    plan = read_json(benchmark_plan)
    parsed_camera_info = parse_camera_info(camera_info)
    trajectory_rows = read_trajectory(trajectory)
    body_to_camera: RigidTransform | None = None
    if tf_static is not None:
        body_to_camera = find_body_to_camera_transform(
            read_tf_static(tf_static),
            camera_frame=camera_frame,
            body_frame=body_frame,
        )
    images = build_images(
        plan,
        parsed_camera_info,
        trajectory_rows,
        body_to_camera=body_to_camera,
        camera_frame=camera_frame if body_to_camera is not None else "",
        body_frame=body_frame if body_to_camera is not None else "",
    )
    ipmatch_results = build_ipmatch_results(matches_dir, images)

    project_files = {
        "project_name": project_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_benchmark_plan": normalize_path(benchmark_plan),
        "images": images,
    }
    project_results = {
        "ipmatch_results": ipmatch_results,
        "lidar_ba_input": {
            "laser_constraint_cloud_path": plan.get("laser_constraint_cloud_path", ""),
            "matches_dir": normalize_path(matches_dir),
            "camera_info": normalize_path(camera_info),
            "trajectory": normalize_path(trajectory),
            "tf_static": normalize_path(tf_static) if tf_static is not None else "",
            "tf_static_camera_frame": camera_frame if body_to_camera is not None else "",
            "tf_static_body_frame": body_frame if body_to_camera is not None else "",
        },
    }

    project_path = output_dir / f"{project_name}.plascan"
    write_plascan(project_path, project_files, project_results)

    summary = {
        "project_path": normalize_path(project_path),
        "image_count": len(images),
        "match_count": len(ipmatch_results),
        "laser_constraint_cloud_path": plan.get("laser_constraint_cloud_path", ""),
        "tf_static": normalize_path(tf_static) if tf_static is not None else "",
        "tf_static_camera_frame": camera_frame if body_to_camera is not None else "",
        "tf_static_body_frame": body_frame if body_to_camera is not None else "",
        "first_image": images[0]["path"],
        "last_image": images[-1]["path"],
    }
    summary_path = output_dir / "project_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return PrepareProjectResult(
        project_path=project_path,
        summary_path=summary_path,
        image_count=len(images),
        match_count=len(ipmatch_results),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-plan", required=True, type=Path)
    parser.add_argument("--camera-info", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--matches-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-name", default="mun_frl_lidar_ba")
    parser.add_argument("--tf-static", type=Path, help="MUN-FRL tf_static_unique.csv used to convert body pose to camera pose")
    parser.add_argument("--camera-frame", default="camera", help="Camera frame name in tf_static")
    parser.add_argument("--body-frame", default="imu_link", help="Body/IMU frame name in tf_static")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare_project(
        benchmark_plan=args.benchmark_plan,
        camera_info=args.camera_info,
        trajectory=args.trajectory,
        matches_dir=args.matches_dir,
        output_dir=args.output_dir,
        project_name=args.project_name,
        tf_static=args.tf_static,
        camera_frame=args.camera_frame,
        body_frame=args.body_frame,
    )
    print(f"project_path={result.project_path}")
    print(f"image_count={result.image_count}")
    print(f"match_count={result.match_count}")
    print(f"summary_path={result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
