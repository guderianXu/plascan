#!/usr/bin/env python3
"""Run a PlaScan full CLI integration test from an image/camera list.

List format: each non-empty line contains an image path and its matching camera
path, separated by whitespace or one comma.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shlex
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUFFIX = {
    "superpoint": ".sp",
    "sift": ".sift",
    "orb": ".orb",
    "akaze": ".akz",
    "surf": ".surf",
    "disk": ".dsk",
    "aliked": ".alk",
}

MODEL_CANDIDATES = {
    "superpoint": {
        "cpu": ["superpoint_extractor_cpu.torchscript", "superpoint_extractor_cpu.pt", "superpoint_extractor.torchscript", "superpoint_extractor.pt"],
        "cuda": ["superpoint_extractor_cuda.torchscript", "superpoint_extractor_cuda.pt", "superpoint_extractor.torchscript", "superpoint_extractor.pt"],
    },
    "disk": {
        "cpu": ["disk_extractor_cpu_8192.torchscript", "disk_extractor_cpu_8192.pt", "disk_extractor_cpu_1200.torchscript", "disk_extractor_cpu_1200.pt", "disk_extractor.torchscript", "disk_extractor.pt"],
        "cuda": ["disk_extractor_cuda_8192.torchscript", "disk_extractor_cuda_8192.pt", "disk_extractor_cuda_1200.torchscript", "disk_extractor_cuda_1200.pt", "disk_extractor.torchscript", "disk_extractor.pt"],
    },
    "aliked": {
        "cpu": ["aliked_extractor_cpu_480.torchscript", "aliked_extractor_cpu_480.pt", "aliked_extractor.torchscript", "aliked_extractor.pt"],
        "cuda": ["aliked_extractor_cuda_480.torchscript", "aliked_extractor_cuda_480.pt", "aliked_extractor.torchscript", "aliked_extractor.pt"],
    },
    "superglue": {
        "cpu": ["superglue_outdoor_cpu.pt", "superglue_outdoor.pt", "superglue_indoor_cpu.pt"],
        "cuda": ["superglue_outdoor_cuda.pt", "superglue_outdoor.pt", "superglue_indoor_cuda.pt"],
    },
    "lightglue": {
        "cpu": ["lightglue_matcher_cpu.torchscript", "lightglue_matcher.torchscript"],
        "cuda": ["lightglue_matcher_cuda.torchscript", "lightglue_matcher.torchscript"],
    },
    "lightglue_disk": {
        "cpu": ["lightglue_disk_cpu.torchscript"],
        "cuda": ["lightglue_disk_cuda.torchscript"],
    },
    "lightglue_aliked": {
        "cpu": ["lightglue_aliked_cpu.torchscript"],
        "cuda": ["lightglue_aliked_cuda.torchscript"],
    },
    "lightglue_sift": {
        "cpu": ["lightglue_sift_cpu.torchscript"],
        "cuda": ["lightglue_sift_cuda.torchscript"],
    },
}

DEFAULT_MAX_KEYPOINTS = {
    "disk": 8192,
    "aliked": 480,
    "superpoint": 4096,
}


@dataclass
class InputItem:
    image: Path
    camera: Path
    work_image: Path | None = None
    work_camera: Path | None = None


@dataclass
class CameraGeometry:
    focal_px: float
    center: tuple[float, float, float]
    rotation: tuple[float, float, float, float, float, float, float, float, float]
    depth_flipped: bool = False


@dataclass(frozen=True)
class LocalFrame:
    origin: tuple[float, float, float]
    axes: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PlaScan full CLI reconstruction pipeline")
    parser.add_argument("list_file", type=Path, help="image/camera list file")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--algorithms", default="all",
                        help="comma list or all; default runs all CLI feature algorithms")
    parser.add_argument("--pair-mode", choices=["all", "adjacent"], default="all")
    parser.add_argument("--max-image-dim", type=int, default=0)
    parser.add_argument("--max-keypoints", type=int, default=0,
                        help="0 uses per-algorithm defaults; DISK defaults to 8192")
    parser.add_argument("--dense-max-disp", type=int, default=0,
                        help="dense disparity upper bound in pixels; 0 estimates from camera geometry")
    parser.add_argument("--dense-algorithm", default="opencv_sgbm")
    parser.add_argument("--dense-cost", default="census")
    parser.add_argument("--dense-pair-mode", choices=["best", "adjacent", "all"], default="adjacent",
                        help="image pairs used for dense reconstruction; matching still uses --pair-mode")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dem-resolution", type=float, default=0.0)
    parser.add_argument("--skip-learned-matchers", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_list(path: Path) -> list[InputItem]:
    items: list[InputItem] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            parts = [part.strip() for part in line.split(",", 1)]
        else:
            parts = shlex.split(line)
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno}: expected '<image> <camera>'")
        image = Path(parts[0]).expanduser().resolve()
        camera = Path(parts[1]).expanduser().resolve()
        if not image.exists():
            raise FileNotFoundError(f"image not found: {image}")
        if not camera.exists():
            raise FileNotFoundError(f"camera not found: {camera}")
        items.append(InputItem(image=image, camera=camera))
    if len(items) < 2:
        raise ValueError("at least two image/camera rows are required")
    return items


_NUMBER_PATTERN = re.compile(r"[+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?")


def parse_numbers(text: str) -> list[float]:
    return [float(value) for value in _NUMBER_PATTERN.findall(text)]


def starts_with_key(text_lower: str, key_lower: str) -> bool:
    if not text_lower.startswith(key_lower):
        return False
    if len(text_lower) == len(key_lower):
        return True
    return text_lower[len(key_lower)].isspace() or text_lower[len(key_lower)] in "=:"


def load_camera_geometry(path: Path) -> CameraGeometry:
    fu = 0.0
    fv = 0.0
    pitch = 1.0
    center = (0.0, 0.0, 0.0)
    rotation = (1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0)
    depth_flipped = False

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = raw.strip()
        if not text:
            continue
        text_lower = text.lower()
        values = parse_numbers(text)
        if starts_with_key(text_lower, "fu") and values:
            fu = values[0]
        elif starts_with_key(text_lower, "fv") and values:
            fv = values[0]
        elif starts_with_key(text_lower, "pitch") and values:
            pitch = values[0]
        elif starts_with_key(text_lower, "c") and len(values) >= 3:
            center = (values[0], values[1], values[2])
        elif starts_with_key(text_lower, "r") and len(values) >= 9:
            rotation = tuple(values[:9])  # type: ignore[assignment]
        elif starts_with_key(text_lower, "w_direction") and values:
            wz = values[2] if len(values) >= 3 else values[0]
            depth_flipped = wz < 0.0

    if pitch <= 0.0 or fu <= 0.0 or fv <= 0.0:
        raise ValueError(f"invalid camera intrinsics in {path}")
    return CameraGeometry(
        focal_px=max(fu, fv) / pitch,
        center=center,
        rotation=rotation,
        depth_flipped=depth_flipped,
    )


def forward_depth_to_origin(camera: CameraGeometry) -> float | None:
    r = camera.rotation
    c = camera.center
    x = -c[0]
    y = -c[1]
    z = -c[2]
    camera_z = r[2] * x + r[5] * y + r[8] * z
    depth = -camera_z if camera.depth_flipped else camera_z
    if not math.isfinite(depth) or depth <= 0.0:
        return None
    return depth


def round_up_to_multiple(value: float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def estimate_dense_max_disp_for_pair(left_camera: Path,
                                     right_camera: Path,
                                     max_image_dim: int) -> int:
    left = load_camera_geometry(left_camera)
    right = load_camera_geometry(right_camera)
    baseline = math.dist(left.center, right.center)
    if baseline <= 0.0 or not math.isfinite(baseline):
        return 128

    depths = [
        depth for depth in (forward_depth_to_origin(left), forward_depth_to_origin(right))
        if depth is not None and depth > baseline * 0.25
    ]
    if depths:
        typical_depth = sorted(depths)[len(depths) // 2]
    else:
        typical_depth = baseline * 8.0

    typical_disp = max(left.focal_px, right.focal_px) * baseline / max(typical_depth, 1.0)
    estimated = round_up_to_multiple(max(128.0, typical_disp * 2.0 + 32.0), 16)
    if max_image_dim > 0:
        cap = max(16, ((max_image_dim - 1) // 16) * 16)
        estimated = min(estimated, cap)
    return max(16, estimated)


def resolve_dense_max_disp(args: argparse.Namespace,
                           left_item: InputItem,
                           right_item: InputItem) -> tuple[int, bool]:
    explicit = int(getattr(args, "dense_max_disp", 0) or 0)
    if explicit > 0:
        return explicit, False
    if left_item.work_camera is None or right_item.work_camera is None:
        raise RuntimeError("working camera paths are not prepared")
    return estimate_dense_max_disp_for_pair(
        left_item.work_camera,
        right_item.work_camera,
        int(getattr(args, "max_image_dim", 0) or 0),
    ), True


def command_path(build_dir: Path, name: str) -> Path:
    candidates = [
        build_dir / "bin" / name,
        build_dir / "src" / "core" / "terrain" / name,
        build_dir / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"required executable not found under {build_dir}: {name}")


def find_model(root: Path, kind: str, device: str) -> Path | None:
    models_dir = root / "resources" / "models"
    device_candidates = MODEL_CANDIDATES.get(kind, {}).get(device, [])
    fallback = MODEL_CANDIDATES.get(kind, {}).get("cpu", [])
    for name in [*device_candidates, *fallback]:
        candidate = models_dir / name
        if candidate.exists():
            return candidate
    return None


def max_keypoints_for_algorithm(args: argparse.Namespace, algo: str) -> int:
    explicit = int(getattr(args, "max_keypoints", 0) or 0)
    if explicit > 0:
        return explicit
    return DEFAULT_MAX_KEYPOINTS.get(algo, 4096)


def lightglue_model_kind_for_algorithm(algo: str) -> str:
    if algo == "disk":
        return "lightglue_disk"
    if algo == "aliked":
        return "lightglue_aliked"
    if algo == "sift":
        return "lightglue_sift"
    return "lightglue"


def run_command(cmd: list[str], log_path: Path) -> CommandResult:
    start = time.monotonic()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    seconds = time.monotonic() - start
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        log.write(f"[exit={proc.returncode} seconds={seconds:.3f}]\n")
        if proc.stdout:
            log.write("[stdout]\n" + proc.stdout + "\n")
        if proc.stderr:
            log.write("[stderr]\n" + proc.stderr + "\n")
    return CommandResult(proc.returncode, proc.stdout, proc.stderr, seconds)


def require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV Python package is required for image resizing") from exc
    return cv2


def prepare_work_inputs(items: list[InputItem], output_dir: Path, max_dim: int) -> None:
    cv2 = require_cv2()
    image_dir = output_dir / "working_images"
    camera_dir = output_dir / "working_cameras"
    image_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(items, 1):
        image = cv2.imread(str(item.image), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"failed to read image: {item.image}")
        height, width = image.shape[:2]
        scale = 1.0
        if max_dim > 0 and max(width, height) > max_dim:
            scale = max_dim / float(max(width, height))
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        work_image = image_dir / f"{index:03d}{item.image.suffix.lower()}"
        if not cv2.imwrite(str(work_image), image):
            raise RuntimeError(f"failed to write working image: {work_image}")

        work_camera = camera_dir / f"{index:03d}{item.camera.suffix.lower()}"
        camera_text = item.camera.read_text()
        if scale != 1.0:
            def replace_pitch(match: re.Match[str]) -> str:
                return f"{match.group(1)}{float(match.group(2)) / scale:.12g}"

            camera_text, count = re.subn(
                r"^(pitch\s*=\s*)([0-9.eE+-]+)",
                replace_pitch,
                camera_text,
                flags=re.MULTILINE,
            )
            if count != 1:
                raise RuntimeError(f"pitch line not found once in {item.camera}")
        work_camera.write_text(camera_text)
        item.work_image = work_image
        item.work_camera = work_camera


def read_feature_count(path: Path) -> int:
    data = path.read_bytes()
    if len(data) < 16:
        return -1
    offset = 4
    _version = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    name_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + name_len
    if offset + 4 > len(data):
        return -1
    return struct.unpack_from("<I", data, offset)[0]


def read_match_count(path: Path) -> int:
    data = path.read_bytes()
    if len(data) < 4:
        return -1
    return struct.unpack_from(">i", data, 0)[0]


def read_ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return -1


def format_intensity_token(value: str) -> str:
    try:
        return str(max(0, min(255, int(round(float(value))))))
    except ValueError:
        return "128"


def row_has_intensity(row: str) -> bool:
    return len(row.split()) >= 5


def read_ascii_ply_vertex_rows(path: Path) -> list[str]:
    rows: list[str] = []
    vertex_count = -1
    vertex_props: list[str] = []
    current_element = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("format ") and "ascii" not in line:
                raise RuntimeError(f"only ASCII PLY can be merged: {path}")
            if line.startswith("element "):
                parts = line.split()
                if len(parts) >= 3:
                    current_element = parts[1]
                    if current_element == "vertex":
                        vertex_count = int(parts[2])
            elif line.startswith("property ") and current_element == "vertex":
                parts = line.split()
                if len(parts) >= 3 and parts[1] != "list":
                    vertex_props.append(parts[-1])
            if line == "end_header":
                break

        def token_for(parts: list[str], name: str, default_index: int, default_value: str) -> str:
            if name in vertex_props:
                prop_index = vertex_props.index(name)
                if prop_index < len(parts):
                    return parts[prop_index]
            if vertex_props:
                return default_value
            if default_index < len(parts):
                return parts[default_index]
            return default_value

        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            x = token_for(parts, "x", 0, "0")
            y = token_for(parts, "y", 1, "0")
            z = token_for(parts, "z", 2, "0")
            error = token_for(parts, "error", 3, "0")
            row_parts = [x, y, z, error]

            intensity = ""
            for prop_name in ("intensity", "gray", "grey", "luminance"):
                if prop_name in vertex_props:
                    prop_index = vertex_props.index(prop_name)
                    if prop_index < len(parts):
                        intensity = format_intensity_token(parts[prop_index])
                        break
            if not intensity and all(prop in vertex_props for prop in ("red", "green", "blue")):
                try:
                    red = float(parts[vertex_props.index("red")])
                    green = float(parts[vertex_props.index("green")])
                    blue = float(parts[vertex_props.index("blue")])
                    intensity = format_intensity_token(str(0.299 * red + 0.587 * green + 0.114 * blue))
                except (ValueError, IndexError):
                    intensity = "128"
            if intensity:
                row_parts.append(intensity)

            rows.append(" ".join(row_parts))
            if vertex_count >= 0 and len(rows) >= vertex_count:
                break
    return rows


def parse_ascii_ply_vertex_row(row: str) -> tuple[float, float, float, float] | None:
    parts = row.split()
    if len(parts) < 3:
        return None
    try:
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2])
        error = float(parts[3]) if len(parts) > 3 else 0.0
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in (x, y, z, error)):
        return None
    return x, y, z, error


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percent / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def point_cloud_quality_from_rows(rows: list[str],
                                  min_point_count: int = 3,
                                  max_p99_to_median_ratio: float = 20.0) -> dict[str, Any]:
    points = [parsed for row in rows if (parsed := parse_ascii_ply_vertex_row(row)) is not None]
    count = len(points)
    if count == 0:
        return {
            "passed": False,
            "reason": "empty_point_cloud",
            "point_count": 0,
        }

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    center = (percentile(xs, 50.0), percentile(ys, 50.0), percentile(zs, 50.0))
    distances = [
        math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2 + (point[2] - center[2]) ** 2)
        for point in points
    ]
    distance_p50 = percentile(distances, 50.0)
    distance_p95 = percentile(distances, 95.0)
    distance_p99 = percentile(distances, 99.0)
    distance_max = max(distances)
    denominator = max(distance_p50, 1e-9)
    p99_to_median_ratio = distance_p99 / denominator

    passed = count >= min_point_count and p99_to_median_ratio <= max_p99_to_median_ratio
    reason = "ok"
    if count < min_point_count:
        reason = "too_few_points"
    elif not passed:
        reason = "too_many_spatial_outliers"

    return {
        "passed": passed,
        "reason": reason,
        "point_count": count,
        "center": center,
        "distance_p50": distance_p50,
        "distance_p95": distance_p95,
        "distance_p99": distance_p99,
        "distance_max": distance_max,
        "p99_to_median_ratio": p99_to_median_ratio,
        "max_p99_to_median_ratio": max_p99_to_median_ratio,
    }


def filter_ascii_ply_vertex_rows(rows: list[str],
                                 max_distance_to_median_ratio: float = 12.0) -> tuple[list[str], dict[str, Any]]:
    parsed_rows: list[tuple[str, tuple[float, float, float, float]]] = []
    for row in rows:
        parsed = parse_ascii_ply_vertex_row(row)
        if parsed is not None:
            parsed_rows.append((row, parsed))

    quality_before = point_cloud_quality_from_rows([row for row, _ in parsed_rows])
    if not parsed_rows:
        return [], {
            "raw_point_count": len(rows),
            "point_count": 0,
            "removed_outlier_count": len(rows),
            "quality_before_filter": quality_before,
            "quality": quality_before,
            "filter_threshold": 0.0,
        }

    center = quality_before.get("center", (0.0, 0.0, 0.0))
    distance_p50 = float(quality_before.get("distance_p50", 0.0) or 0.0)
    distance_p95 = float(quality_before.get("distance_p95", 0.0) or 0.0)
    base_distance = max(distance_p50, min(distance_p95, distance_p50 * 4.0) if distance_p50 > 0 else 1.0, 1.0)
    threshold = base_distance * max_distance_to_median_ratio

    filtered: list[str] = []
    for row, point in parsed_rows:
        distance = math.sqrt(
            (point[0] - center[0]) ** 2
            + (point[1] - center[1]) ** 2
            + (point[2] - center[2]) ** 2
        )
        if distance <= threshold:
            filtered.append(row)

    quality_after = point_cloud_quality_from_rows(filtered)
    return filtered, {
        "raw_point_count": len(parsed_rows),
        "point_count": len(filtered),
        "removed_outlier_count": len(parsed_rows) - len(filtered),
        "quality_before_filter": quality_before,
        "quality": quality_after,
        "filter_threshold": threshold,
    }


def write_merged_ascii_ply(paths: list[Path], output_path: Path) -> int:
    rows: list[str] = []
    for path in paths:
        rows.extend(read_ascii_ply_vertex_rows(path))

    write_ascii_ply_rows(rows, output_path)
    return len(rows)


def write_ascii_ply_rows(rows: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_intensity = any(row_has_intensity(row) for row in rows)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(rows)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property float error\n")
        if has_intensity:
            handle.write("property uchar intensity\n")
        handle.write("end_header\n")
        for row in rows:
            parts = row.split()
            if has_intensity and len(parts) == 4:
                parts.append("128")
            handle.write(" ".join(parts[:5] if has_intensity else parts[:4]))
            handle.write("\n")


def format_float(value: float) -> str:
    text = f"{value:.12g}"
    return "0" if text == "-0" else text


def dot3(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def transform_point_to_local(point: tuple[float, float, float],
                             frame: LocalFrame) -> tuple[float, float, float]:
    delta = (
        point[0] - frame.origin[0],
        point[1] - frame.origin[1],
        point[2] - frame.origin[2],
    )
    return (
        dot3(frame.axes[0], delta),
        dot3(frame.axes[1], delta),
        dot3(frame.axes[2], delta),
    )


def transform_rotation_to_local(rotation: tuple[float, float, float,
                                               float, float, float,
                                               float, float, float],
                                frame: LocalFrame) -> tuple[float, float, float,
                                                             float, float, float,
                                                             float, float, float]:
    matrix = (
        rotation[0:3],
        rotation[3:6],
        rotation[6:9],
    )
    values: list[float] = []
    for axis in frame.axes:
        for column in range(3):
            values.append(
                axis[0] * matrix[0][column]
                + axis[1] * matrix[1][column]
                + axis[2] * matrix[2][column]
            )
    return tuple(values)  # type: ignore[return-value]


def compute_point_cloud_local_frame(rows: list[str]) -> LocalFrame | None:
    points = [parsed[:3] for row in rows if (parsed := parse_ascii_ply_vertex_row(row)) is not None]
    if len(points) < 3:
        return None
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] < 3:
        return None
    origin_array = np.median(array, axis=0)
    centered = array - origin_array
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]

    first = eigenvectors[:, order[0]]
    second = eigenvectors[:, order[1]]

    def orient(vector: Any) -> Any:
        index = int(np.argmax(np.abs(vector)))
        return -vector if vector[index] < 0.0 else vector

    first = orient(first)
    second = second - first * float(np.dot(first, second))
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-12:
        return None
    second = orient(second / second_norm)
    third = np.cross(first, second)
    third_norm = float(np.linalg.norm(third))
    if third_norm < 1e-12:
        return None
    third = third / third_norm

    return LocalFrame(
        origin=tuple(float(value) for value in origin_array),  # type: ignore[arg-type]
        axes=(
            tuple(float(value) for value in first),   # type: ignore[arg-type]
            tuple(float(value) for value in second),  # type: ignore[arg-type]
            tuple(float(value) for value in third),   # type: ignore[arg-type]
        ),
    )


def rewrite_tsai_pose(text: str,
                      center: tuple[float, float, float],
                      rotation: tuple[float, float, float,
                                      float, float, float,
                                      float, float, float]) -> str:
    center_line = "C = " + " ".join(format_float(value) for value in center)
    rotation_line = "R = " + " ".join(format_float(value) for value in rotation)
    text, center_count = re.subn(r"^C\s*=.*$", center_line, text, count=1, flags=re.MULTILINE)
    text, rotation_count = re.subn(r"^R\s*=.*$", rotation_line, text, count=1, flags=re.MULTILINE)
    if center_count != 1 or rotation_count != 1:
        raise ValueError("camera file must contain exactly one C and one R pose line")
    return text


def prepare_terrain_local_frame(point_cloud_path: Path,
                                items: list[InputItem],
                                frame_dir: Path,
                                frame: LocalFrame | None = None) -> dict[str, Any]:
    rows = read_ascii_ply_vertex_rows(point_cloud_path)
    local_frame = frame if frame is not None else compute_point_cloud_local_frame(rows)
    if local_frame is None:
        return {
            "enabled": False,
            "reason": "local_frame_unavailable",
            "point_cloud": str(point_cloud_path),
            "items": items,
        }

    frame_dir.mkdir(parents=True, exist_ok=True)
    local_cloud = frame_dir / "dense_local_frame.ply"
    local_rows: list[str] = []
    for row in rows:
        parsed = parse_ascii_ply_vertex_row(row)
        if parsed is None:
            continue
        parts = row.split()
        local = transform_point_to_local((parsed[0], parsed[1], parsed[2]), local_frame)
        local_row = [
            format_float(local[0]),
            format_float(local[1]),
            format_float(local[2]),
            format_float(parsed[3]),
        ]
        if len(parts) >= 5:
            local_row.append(format_intensity_token(parts[4]))
        local_rows.append(" ".join(local_row))
    write_ascii_ply_rows(local_rows, local_cloud)

    camera_dir = frame_dir / "cameras"
    camera_dir.mkdir(parents=True, exist_ok=True)
    local_items: list[InputItem] = []
    for index, item in enumerate(items, 1):
        if item.work_image is None or item.work_camera is None:
            raise RuntimeError("working image/camera paths are not prepared")
        camera = load_camera_geometry(item.work_camera)
        local_center = transform_point_to_local(camera.center, local_frame)
        local_rotation = transform_rotation_to_local(camera.rotation, local_frame)
        local_camera = camera_dir / f"{index:03d}{item.work_camera.suffix.lower()}"
        local_camera.write_text(
            rewrite_tsai_pose(item.work_camera.read_text(encoding="utf-8", errors="replace"),
                              local_center,
                              local_rotation),
            encoding="utf-8",
        )
        local_items.append(InputItem(
            image=item.image,
            camera=item.camera,
            work_image=item.work_image,
            work_camera=local_camera,
        ))

    return {
        "enabled": True,
        "reason": "ok",
        "point_cloud": str(local_cloud),
        "items": local_items,
        "origin": local_frame.origin,
        "axes": local_frame.axes,
        "input_point_cloud": str(point_cloud_path),
    }


def write_quality_filtered_ascii_ply(paths: list[Path], output_path: Path) -> dict[str, Any]:
    rows: list[str] = []
    for path in paths:
        rows.extend(read_ascii_ply_vertex_rows(path))

    filtered_rows, result = filter_ascii_ply_vertex_rows(rows)
    write_ascii_ply_rows(filtered_rows, output_path)
    result["path"] = str(output_path)
    return result


def evaluate_dom_mask_quality(mask: list[list[int]] | Any,
                              max_large_components: int = 2,
                              min_largest_component_ratio: float = 0.8) -> dict[str, Any]:
    rows = [[bool(value) for value in row] for row in mask]
    height = len(rows)
    width = len(rows[0]) if height else 0
    if height == 0 or width == 0:
        return {
            "passed": False,
            "reason": "empty_dom_mask",
            "width": width,
            "height": height,
            "nonblack_pixel_count": 0,
            "nonblack_ratio": 0.0,
            "component_count": 0,
            "large_component_count": 0,
            "largest_component_ratio": 0.0,
        }

    visited = [[False] * width for _ in range(height)]
    areas: list[int] = []
    nonblack = sum(1 for row in rows for value in row if value)
    for y in range(height):
        for x in range(width):
            if not rows[y][x] or visited[y][x]:
                continue
            stack = [(x, y)]
            visited[y][x] = True
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if nx == cx and ny == cy:
                            continue
                        if rows[ny][nx] and not visited[ny][nx]:
                            visited[ny][nx] = True
                            stack.append((nx, ny))
            areas.append(area)

    largest = max(areas) if areas else 0
    large_threshold = max(1, int(math.ceil(nonblack * 0.01)))
    large_components = [area for area in areas if area >= large_threshold]
    largest_ratio = (largest / nonblack) if nonblack else 0.0
    nonblack_ratio = nonblack / float(width * height)
    passed = (
        nonblack > 0
        and len(large_components) <= max_large_components
        and largest_ratio >= min_largest_component_ratio
    )
    reason = "ok"
    if nonblack == 0:
        reason = "empty_dom"
    elif len(large_components) > max_large_components:
        reason = "fragmented_dom"
    elif largest_ratio < min_largest_component_ratio:
        reason = "dominant_component_too_small"

    return {
        "passed": passed,
        "reason": reason,
        "width": width,
        "height": height,
        "nonblack_pixel_count": nonblack,
        "nonblack_ratio": nonblack_ratio,
        "component_count": len(areas),
        "large_component_count": len(large_components),
        "largest_component_ratio": largest_ratio,
        "large_component_threshold": large_threshold,
        "max_large_components": max_large_components,
        "min_largest_component_ratio": min_largest_component_ratio,
    }


def evaluate_dom_quality(path: Path) -> dict[str, Any]:
    cv2 = require_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return {"passed": False, "reason": "dom_not_readable", "path": str(path)}
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    mask = (gray > 5).tolist()
    quality = evaluate_dom_mask_quality(mask)
    quality["path"] = str(path)
    return quality


def evaluate_dem_quality(path: Path,
                         max_p01_p99_range: float = 50000.0) -> dict[str, Any]:
    cv2 = require_cv2()
    dem = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if dem is None:
        return {"passed": False, "reason": "dem_not_readable", "path": str(path)}

    values: list[float] = []
    for raw in dem.reshape(-1).tolist():
        value = float(raw)
        if math.isfinite(value) and value > -9998.5:
            values.append(value)
    if not values:
        return {
            "passed": False,
            "reason": "dem_has_no_valid_pixels",
            "path": str(path),
            "valid_pixel_count": 0,
        }

    z_p01 = percentile(values, 1.0)
    z_p50 = percentile(values, 50.0)
    z_p99 = percentile(values, 99.0)
    z_range = z_p99 - z_p01
    passed = z_range <= max_p01_p99_range
    return {
        "passed": passed,
        "reason": "ok" if passed else "dem_height_range_too_large",
        "path": str(path),
        "width": int(dem.shape[1]),
        "height": int(dem.shape[0]),
        "valid_pixel_count": len(values),
        "valid_pixel_ratio": len(values) / float(dem.size),
        "z_p01": z_p01,
        "z_p50": z_p50,
        "z_p99": z_p99,
        "z_p01_p99_range": z_range,
        "max_p01_p99_range": max_p01_p99_range,
    }


def algorithm_list(value: str) -> list[str]:
    if value == "all":
        return ["superpoint", "sift", "orb", "akaze", "surf", "disk", "aliked"]
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def pair_indices(count: int, mode: str) -> list[tuple[int, int]]:
    if mode == "adjacent":
        return [(index, index + 1) for index in range(count - 1)]
    return list(itertools.combinations(range(count), 2))


def dense_pair_indices(count: int, mode: str, best_pair: list[int]) -> list[tuple[int, int]]:
    if mode == "best":
        return [(int(best_pair[0]), int(best_pair[1]))]
    if mode == "adjacent":
        return pair_indices(count, "adjacent")
    return pair_indices(count, "all")


def extract_features(args: argparse.Namespace,
                     items: list[InputItem],
                     tools: dict[str, Path],
                     output_dir: Path,
                     log_path: Path) -> dict[str, Any]:
    root = repo_root()
    features: dict[str, Any] = {}
    for algo in algorithm_list(args.algorithms):
        if algo not in SUFFIX:
            features[algo] = {"status": "skipped", "reason": f"unknown algorithm: {algo}"}
            continue
        model = find_model(root, algo, args.device) if algo in {"superpoint", "disk", "aliked"} else None
        if algo in {"superpoint", "disk", "aliked"} and model is None:
            features[algo] = {"status": "skipped", "reason": "model not found"}
            continue

        algo_dir = output_dir / "features" / algo
        algo_dir.mkdir(parents=True, exist_ok=True)
        records = []
        failures = []
        for index, item in enumerate(items):
            out_path = algo_dir / f"{item.work_image.stem}{SUFFIX[algo]}"
            cmd = [
                str(tools["feature_extract_cli"]),
                "-a", algo,
                "-i", str(item.work_image),
                "-o", str(out_path),
                "-n", str(max_keypoints_for_algorithm(args, algo)),
                "--max-dim", str(args.max_image_dim),
            ]
            if model:
                cmd.extend(["-m", str(model)])
            cmd.append("--cuda" if args.device == "cuda" else "--no-cuda")
            result = run_command(cmd, log_path)
            if result.returncode == 0 and out_path.exists():
                records.append({
                    "image_index": index,
                    "path": str(out_path),
                    "keypoints": read_feature_count(out_path),
                })
            else:
                failures.append({
                    "image_index": index,
                    "returncode": result.returncode,
                    "message": (result.stderr or result.stdout).strip().splitlines()[-3:],
                })
        features[algo] = {
            "status": "ok" if records else "failed",
            "model": str(model) if model else "",
            "records": records,
            "failures": failures,
        }
    return features


def match_features(args: argparse.Namespace,
                   items: list[InputItem],
                   tools: dict[str, Path],
                   features: dict[str, Any],
                   output_dir: Path,
                   log_path: Path) -> list[dict[str, Any]]:
    root = repo_root()
    match_rows: list[dict[str, Any]] = []
    pairs = pair_indices(len(items), args.pair_mode)
    for algo, data in features.items():
        records = {row["image_index"]: Path(row["path"]) for row in data.get("records", [])}
        if len(records) < 2:
            continue
        matcher_specs: list[tuple[str, Path | None]] = [("bf", None)]
        if not args.skip_learned_matchers:
            lg_model = find_model(root, lightglue_model_kind_for_algorithm(algo), args.device)
            if lg_model is not None:
                matcher_specs.append(("lightglue", lg_model))
            if algo == "superpoint":
                sg_model = find_model(root, "superglue", args.device)
                if sg_model is not None:
                    matcher_specs.append(("superglue", sg_model))

        for matcher, model in matcher_specs:
            for left, right in pairs:
                if left not in records or right not in records:
                    continue
                out_dir = output_dir / "matches" / f"{algo}_{matcher}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{left + 1:03d}_{right + 1:03d}.match"
                cmd = [
                    str(tools["feature_match_cli"]),
                    "-a", matcher,
                    "--sp1", str(records[left]),
                    "--sp2", str(records[right]),
                    "-o", str(out_path),
                    "--cuda" if args.device == "cuda" else "--no-cuda",
                ]
                if model:
                    cmd.extend(["-m", str(model)])
                result = run_command(cmd, log_path)
                count = read_match_count(out_path) if result.returncode == 0 and out_path.exists() else 0
                match_rows.append({
                    "feature_algorithm": algo,
                    "matcher": matcher,
                    "pair": [left, right],
                    "match_count": count,
                    "path": str(out_path) if out_path.exists() else "",
                    "status": "ok" if count > 0 else "failed",
                    "returncode": result.returncode,
                })
    return match_rows


def run_dense_attempt(args: argparse.Namespace,
                      items: list[InputItem],
                      tools: dict[str, Path],
                      output_dir: Path,
                      log_path: Path,
                      left: int,
                      right: int) -> dict[str, Any]:
    prefix = output_dir / "dense" / f"{left + 1:03d}_{right + 1:03d}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rect_prefix = str(prefix) + "_rect"
    disp_path = prefix.with_suffix(".disp.tif")
    ply_path = prefix.with_suffix(".ply")
    dense_max_disp, dense_max_disp_auto = resolve_dense_max_disp(args, items[left], items[right])

    rect = run_command([
        str(tools["rectify_cli"]),
        "-L", str(items[left].work_image),
        "-R", str(items[right].work_image),
        "--camL", str(items[left].work_camera),
        "--camR", str(items[right].work_camera),
        "-o", rect_prefix,
        "-V",
    ], log_path)
    if rect.returncode != 0:
        return {
            "pair": [left, right],
            "status": "rectify_failed",
            "points": 0,
            "dense_max_disp": dense_max_disp,
            "dense_max_disp_auto": dense_max_disp_auto,
        }

    dense = run_command([
        str(tools["dense_match_cli"]),
        "-L", rect_prefix + "_L.tif",
        "-R", rect_prefix + "_R.tif",
        "-o", str(disp_path),
        "-a", args.dense_algorithm,
        "-f", args.dense_cost,
        "--subpixel", "none",
        "--min-disp", "0",
        "--max-disp", str(dense_max_disp),
        "--kernel-w", "7",
        "--kernel-h", "7",
        "--threads", str(args.threads),
        "--median-filter", "3",
        "--cuda" if args.device == "cuda" else "--no-cuda",
        "-V",
    ], log_path)
    if dense.returncode != 0 or not disp_path.exists():
        return {
            "pair": [left, right],
            "status": "dense_failed",
            "points": 0,
            "dense_max_disp": dense_max_disp,
            "dense_max_disp_auto": dense_max_disp_auto,
        }

    tri = run_command([
        str(tools["triangulate_cli"]),
        "-d", str(disp_path),
        "--rect", rect_prefix + ".xml",
        "--camL", str(items[left].work_camera),
        "--camR", str(items[right].work_camera),
        "-o", str(ply_path),
        "--intensity-image", rect_prefix + "_L.tif",
        "--max-error", "100",
        "--threads", str(args.threads),
        "-V",
    ], log_path)
    points = read_ply_vertex_count(ply_path) if tri.returncode == 0 and ply_path.exists() else 0
    return {
        "pair": [left, right],
        "status": "ok" if points > 0 else "triangulation_empty",
        "points": points,
        "rect_prefix": rect_prefix,
        "disparity": str(disp_path),
        "point_cloud": str(ply_path) if ply_path.exists() else "",
        "dense_max_disp": dense_max_disp,
        "dense_max_disp_auto": dense_max_disp_auto,
    }


def run_terrain(args: argparse.Namespace,
                items: list[InputItem],
                tools: dict[str, Path],
                dense_result: dict[str, Any],
                output_dir: Path,
                log_path: Path) -> dict[str, Any]:
    terrain_dir = output_dir / "terrain"
    terrain_dir.mkdir(parents=True, exist_ok=True)
    terrain_frame = prepare_terrain_local_frame(
        Path(dense_result["point_cloud"]),
        items,
        output_dir / "terrain_local_frame",
    )
    terrain_items = terrain_frame.get("items", items)
    terrain_point_cloud = str(terrain_frame.get("point_cloud", dense_result["point_cloud"]))

    image_camera_list = output_dir / "terrain_image_camera.lis"
    with image_camera_list.open("w", encoding="utf-8") as handle:
        for item in terrain_items:
            if item.work_image is None or item.work_camera is None:
                raise RuntimeError("working image/camera paths are not prepared")
            handle.write(f"{item.work_image} {item.work_camera}\n")

    cmd = [
        str(tools["terrain_dem_dom_tool"]),
        "--list",
        terrain_point_cloud,
        str(image_camera_list),
        str(terrain_dir),
        str(args.dem_resolution),
    ]
    result = run_command(cmd, log_path)
    products = terrain_dir / "products"
    terrain_payload: dict[str, Any] = {}
    try:
        terrain_payload = json.loads(result.stdout) if result.stdout.strip() else {}
    except json.JSONDecodeError:
        terrain_payload = {}
    dom_payload = terrain_payload.get("dom", {}) if isinstance(terrain_payload, dict) else {}
    dem_payload = terrain_payload.get("dem", {}) if isinstance(terrain_payload, dict) else {}
    camera_projected = bool(dom_payload.get("camera_projected"))
    filled_pixel_count = int(dom_payload.get("filled_pixel_count", 0) or 0)
    dem_exists = (products / "dem.tif").exists()
    dom_exists = (products / "dom.png").exists()
    dom_quality = evaluate_dom_quality(products / "dom.png") if dom_exists else {
        "passed": False,
        "reason": "dom_missing",
    }
    dem_quality = evaluate_dem_quality(products / "dem.tif") if dem_exists else {
        "passed": False,
        "reason": "dem_missing",
    }
    mesh_ply = str(dem_payload.get("mesh_ply", "") or "")
    quality_ok = bool(dom_quality.get("passed")) and bool(dem_quality.get("passed"))
    return {
        "status": "ok" if result.returncode == 0
        and dem_exists
        and dom_exists
        and camera_projected
        and filled_pixel_count > 0
        and quality_ok else "failed",
        "returncode": result.returncode,
        "image_camera_list": str(image_camera_list),
        "dem_tif": str(products / "dem.tif"),
        "dom_png": str(products / "dom.png"),
        "mesh_ply": mesh_ply,
        "dem_exists": dem_exists,
        "dom_exists": dom_exists,
        "mesh_exists": Path(mesh_ply).exists() if mesh_ply else False,
        "camera_projected": camera_projected,
        "filled_pixel_count": filled_pixel_count,
        "quality_ok": quality_ok,
        "dom_quality": dom_quality,
        "dem_quality": dem_quality,
        "terrain_frame": {
            key: value
            for key, value in terrain_frame.items()
            if key != "items"
        },
        "tool_result": terrain_payload,
    }


def write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    best = report.get("best_match") or {}
    dense = report.get("dense_result") or {}
    terrain = report.get("terrain_result") or {}
    mesh_path = terrain.get("mesh_ply") or terrain.get("mesh_obj") or ""
    dom_path = terrain.get("dom_png") or terrain.get("dom_tif") or ""
    lines = [
        "# PlaScan Full Pipeline",
        "",
        f"- status: {report.get('status')}",
        f"- input_count: {len(report.get('inputs', []))}",
        f"- best: {best.get('feature_algorithm')} + {best.get('matcher')} "
        f"pair={best.get('pair')} matches={best.get('match_count')}",
        f"- dense_points: {dense.get('points', 0)}",
        f"- dense_source_pairs: {dense.get('source_pair_count', 0)}",
        f"- point_cloud: {dense.get('point_cloud', '')}",
        f"- mesh: {mesh_path} exists={terrain.get('mesh_exists')}",
        f"- dem: {terrain.get('dem_tif', '')} exists={terrain.get('dem_exists')}",
        f"- dom: {dom_path} exists={terrain.get('dom_exists')} "
        f"camera_projected={terrain.get('camera_projected')}",
        f"- dom_filled_pixels: {terrain.get('filled_pixel_count', 0)}",
        "",
        "## Match Leaderboard",
    ]
    rows = sorted(report.get("matches", []), key=lambda row: row.get("match_count", 0), reverse=True)
    for row in rows[:20]:
        lines.append(
            f"- {row['feature_algorithm']} + {row['matcher']} "
            f"pair={row['pair']} matches={row['match_count']} status={row['status']}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = repo_root()
    build_dir = (root / args.build_dir).resolve() if not args.build_dir.is_absolute() else args.build_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or build_dir / "测试用临时文件" / f"full_pipeline_{timestamp}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "commands.log"

    tools = {
        "feature_extract_cli": command_path(build_dir, "feature_extract_cli"),
        "feature_match_cli": command_path(build_dir, "feature_match_cli"),
        "rectify_cli": command_path(build_dir, "rectify_cli"),
        "dense_match_cli": command_path(build_dir, "dense_match_cli"),
        "triangulate_cli": command_path(build_dir, "triangulate_cli"),
        "terrain_dem_dom_tool": command_path(build_dir, "terrain_dem_dom_tool"),
    }

    items = parse_list(args.list_file)
    prepare_work_inputs(items, output_dir, args.max_image_dim)
    features = extract_features(args, items, tools, output_dir, log_path)
    matches = match_features(args, items, tools, features, output_dir, log_path)
    valid_matches = [row for row in matches if row.get("match_count", 0) > 0]
    if not valid_matches:
        report = {"status": "failed", "reason": "no valid matches", "features": features, "matches": matches}
        write_reports(output_dir, report)
        print(f"FAILED: no valid matches; report={output_dir / 'report.json'}")
        return 2

    best = max(valid_matches, key=lambda row: row["match_count"])
    dense_pairs = dense_pair_indices(len(items), args.dense_pair_mode, best["pair"])
    attempts: list[dict[str, Any]] = []
    selected_dense_results: list[dict[str, Any]] = []
    terrain_result: dict[str, Any] = {"status": "skipped"}
    for left, right in dense_pairs:
        pair_attempts = [
            run_dense_attempt(args, items, tools, output_dir, log_path, left, right),
            run_dense_attempt(args, items, tools, output_dir, log_path, right, left),
        ]
        attempts.extend(pair_attempts)
        selected = max(pair_attempts, key=lambda row: row.get("points", 0))
        if selected.get("points", 0) > 0 and selected.get("point_cloud"):
            selected_dense_results.append(selected)

    if selected_dense_results:
        merged_cloud = output_dir / "dense" / "merged_dense.ply"
        merge_result = write_quality_filtered_ascii_ply(
            [Path(row["point_cloud"]) for row in selected_dense_results],
            merged_cloud,
        )
        merged_points = int(merge_result.get("point_count", 0) or 0)
        cloud_quality_ok = bool(merge_result.get("quality", {}).get("passed"))
        dense_result = {
            "status": "ok" if merged_points > 0 and cloud_quality_ok else "quality_failed",
            "points": merged_points,
            "raw_points": int(merge_result.get("raw_point_count", 0) or 0),
            "removed_outlier_count": int(merge_result.get("removed_outlier_count", 0) or 0),
            "point_cloud": str(merged_cloud),
            "quality": merge_result.get("quality", {}),
            "quality_before_filter": merge_result.get("quality_before_filter", {}),
            "filter_threshold": merge_result.get("filter_threshold", 0.0),
            "source_pair_count": len(selected_dense_results),
            "source_results": selected_dense_results,
        }
    else:
        dense_result = {"status": "failed", "points": 0, "point_cloud": "", "source_pair_count": 0}

    if dense_result.get("status") == "ok" and dense_result.get("points", 0) > 0:
        terrain_result = run_terrain(args, items, tools, dense_result, output_dir, log_path)

    status = "ok" if dense_result.get("points", 0) > 0 and terrain_result.get("status") == "ok" else "failed"
    report = {
        "status": status,
        "list_file": str(args.list_file.resolve()),
        "output_dir": str(output_dir),
        "inputs": [
            {
                "image": str(item.image),
                "camera": str(item.camera),
                "working_image": str(item.work_image),
                "working_camera": str(item.work_camera),
            }
            for item in items
        ],
        "features": features,
        "matches": matches,
        "best_match": best,
        "dense_pair_mode": args.dense_pair_mode,
        "dense_attempts": attempts,
        "dense_result": dense_result,
        "terrain_result": terrain_result,
    }
    write_reports(output_dir, report)
    print(f"status={status}")
    print(f"output_dir={output_dir}")
    print(f"best={best['feature_algorithm']}+{best['matcher']} pair={best['pair']} matches={best['match_count']}")
    print(f"dense_points={dense_result.get('points', 0)}")
    print(f"report={output_dir / 'report.json'}")
    return 0 if status == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
