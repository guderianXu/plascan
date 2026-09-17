#!/usr/bin/env python3
"""Generate a deterministic aerial terrain dataset with independent DEM/DOM truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


GENERATOR_VERSION = "1.0"
DATASET_KIND = "plascan_synthetic_terrain"

QUALITY_PRESETS: dict[str, dict[str, int | float]] = {
    "coarse": {
        "view_count": 12,
        "image_width": 512,
        "image_height": 384,
        "focal_length_px": 555.0,
        "dem_height": 192,
        "dem_width": 256,
        "mesh_height": 72,
        "mesh_width": 96,
        "texture_height": 384,
        "texture_width": 512,
        "detail_layers": 2,
        "sensor_noise_std": 0.0,
    },
    "medium": {
        "view_count": 20,
        "image_width": 640,
        "image_height": 480,
        "focal_length_px": 690.0,
        "dem_height": 384,
        "dem_width": 512,
        "mesh_height": 144,
        "mesh_width": 192,
        "texture_height": 768,
        "texture_width": 1024,
        "detail_layers": 4,
        "sensor_noise_std": 0.35,
    },
    "fine": {
        "view_count": 30,
        "image_width": 960,
        "image_height": 736,
        "focal_length_px": 1035.0,
        "dem_height": 768,
        "dem_width": 1024,
        "mesh_height": 288,
        "mesh_width": 384,
        "texture_height": 1536,
        "texture_width": 2048,
        "detail_layers": 6,
        "sensor_noise_std": 0.7,
    },
}


@dataclass(frozen=True)
class DatasetConfig:
    output_dir: Path
    quality_level: str = "medium"
    view_count: int = 20
    image_width: int = 640
    image_height: int = 480
    focal_length_px: float = 690.0
    terrain_width_m: float = 24.0
    terrain_height_m: float = 18.0
    camera_height_m: float = 19.0
    dem_height: int = 384
    dem_width: int = 512
    mesh_height: int = 144
    mesh_width: int = 192
    texture_height: int = 768
    texture_width: int = 1024
    detail_layers: int = 4
    sensor_noise_std: float = 0.35
    seed: int = 20260914


@dataclass(frozen=True)
class Camera:
    camera_id: str
    center: np.ndarray
    target: np.ndarray
    camera_to_world: np.ndarray


def validate_config(config: DatasetConfig) -> None:
    if config.quality_level not in (*QUALITY_PRESETS, "custom"):
        raise ValueError(f"unknown quality level: {config.quality_level}")
    if config.view_count < 4:
        raise ValueError("view count must be at least 4")
    if config.image_width < 64 or config.image_height < 48:
        raise ValueError("image dimensions must be at least 64x48")
    if config.focal_length_px <= 0.0:
        raise ValueError("focal length must be positive")
    if config.terrain_width_m <= 0.0 or config.terrain_height_m <= 0.0:
        raise ValueError("terrain dimensions must be positive")
    if config.camera_height_m <= 8.0:
        raise ValueError("camera height must exceed the terrain relief")
    if min(config.dem_height, config.dem_width) < 16:
        raise ValueError("DEM dimensions are too small")
    if min(config.mesh_height, config.mesh_width) < 8:
        raise ValueError("mesh dimensions are too small")
    if min(config.texture_height, config.texture_width) < 32:
        raise ValueError("texture dimensions are too small")
    if config.detail_layers < 1:
        raise ValueError("detail layers must be positive")
    if config.sensor_noise_std < 0.0:
        raise ValueError("sensor noise must not be negative")


def ensure_empty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def normalize(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(lengths, 1.0e-12)


def terrain_elevation(x: np.ndarray, y: np.ndarray, detail_layers: int) -> np.ndarray:
    elevation = (
        0.72 * np.sin(0.23 * x + 0.31 * y)
        + 0.46 * np.cos(0.47 * x - 0.16 * y)
        + 0.28 * np.sin(0.82 * x + 0.57 * y)
    )
    ridge = 1.05 * np.exp(-((y - 0.20 * x - 1.2) / 1.55) ** 2)
    channel = -0.68 * np.exp(-((x + 0.32 * y + 2.1) / 0.78) ** 2)
    crater = -0.92 * np.exp(-((x - 4.1) ** 2 + (y + 2.4) ** 2) / 2.2)
    crater += 0.34 * np.exp(-(((x - 4.1) ** 2 + (y + 2.4) ** 2) - 3.1) ** 2 / 2.8)
    elevation = elevation + ridge + channel + crater
    for layer in range(1, detail_layers + 1):
        frequency = 0.72 * (1.55**layer)
        amplitude = 0.19 * (0.56**layer)
        elevation += amplitude * np.sin(frequency * x + 0.61 * layer) * np.cos(
            0.83 * frequency * y - 0.37 * layer
        )
    return elevation.astype(np.float64, copy=False)


def terrain_grid(config: DatasetConfig, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-0.5 * config.terrain_width_m, 0.5 * config.terrain_width_m, width)
    y = np.linspace(0.5 * config.terrain_height_m, -0.5 * config.terrain_height_m, height)
    xx, yy = np.meshgrid(x, y)
    return xx, yy, terrain_elevation(xx, yy, config.detail_layers)


def build_dom(config: DatasetConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    height, width = config.texture_height, config.texture_width
    noise = np.zeros((height, width), dtype=np.float32)
    weight_sum = 0.0
    for layer in range(config.detail_layers):
        layer_noise = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
        sigma = max(0.6, width / (75.0 * (1.9**layer)))
        weight = 0.61**layer
        noise += weight * cv2.GaussianBlur(layer_noise, (0, 0), sigma)
        weight_sum += weight
    noise /= weight_sum
    noise -= float(noise.min())
    noise /= max(float(noise.max()), 1.0e-6)

    xx, yy, elevation = terrain_grid(config, height, width)
    normalized_height = (elevation - elevation.min()) / max(float(np.ptp(elevation)), 1.0e-6)
    dom = np.empty((height, width, 3), dtype=np.float32)
    dom[..., 0] = 58.0 + 58.0 * noise + 18.0 * normalized_height
    dom[..., 1] = 69.0 + 72.0 * noise + 25.0 * normalized_height
    dom[..., 2] = 76.0 + 78.0 * noise + 31.0 * normalized_height

    image = np.clip(dom, 0.0, 255.0).astype(np.uint8)
    for _ in range(max(30, config.detail_layers * 24)):
        center = np.array(
            [int(rng.integers(0, width)), int(rng.integers(0, height))]
        )
        radius = int(rng.integers(max(1, width // 500), max(3, width // 110)))
        gray = int(rng.integers(45, 175))
        tint = int(rng.integers(-10, 11))
        color = (gray - tint, gray, gray + tint)
        angles = np.linspace(0.0, 2.0 * math.pi, int(rng.integers(6, 11)), endpoint=False)
        radii = radius * rng.uniform(0.58, 1.25, size=len(angles))
        points = np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None]
        points = np.rint(points + center).astype(np.int32)
        cv2.fillPoly(image, [points], color, cv2.LINE_AA)
    for _ in range(max(8, config.detail_layers * 5)):
        points = np.column_stack(
            (
                np.linspace(rng.integers(0, width), rng.integers(0, width), 5),
                np.linspace(rng.integers(0, height), rng.integers(0, height), 5)
                + rng.normal(0.0, height / 45.0, 5),
            )
        ).astype(np.int32)
        gray = int(rng.integers(45, 105))
        cv2.polylines(image, [points], False, (gray, gray + 4, gray + 9), 1, cv2.LINE_AA)
    return image


def look_at_camera_to_world(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = normalize(target - center)
    right = normalize(np.cross(forward, np.array([0.0, 0.0, 1.0])))
    down = normalize(np.cross(forward, right))
    return np.column_stack((right, down, forward))


def camera_grid(config: DatasetConfig) -> list[Camera]:
    columns = math.ceil(math.sqrt(config.view_count * 4.0 / 3.0))
    rows = math.ceil(config.view_count / columns)
    x_values = np.linspace(-4.8, 4.8, columns)
    y_values = np.linspace(-3.6, 3.6, rows)
    cameras: list[Camera] = []
    for index in range(config.view_count):
        row, column = divmod(index, columns)
        x = float(x_values[column])
        y = float(y_values[row])
        center = np.array([x, y, config.camera_height_m + 0.35 * math.sin(index)], dtype=np.float64)
        target = np.array([0.28 * x, 0.28 * y, 0.0], dtype=np.float64)
        cameras.append(Camera(f"view_{index:02d}", center, target, look_at_camera_to_world(center, target)))
    return cameras


def sample_dom(config: DatasetConfig, dom: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    map_x = ((x / config.terrain_width_m) + 0.5) * (dom.shape[1] - 1)
    map_y = (0.5 - y / config.terrain_height_m) * (dom.shape[0] - 1)
    return cv2.remap(
        dom,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def render_view(config: DatasetConfig, camera: Camera, dom: np.ndarray, camera_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = config.image_height, config.image_width
    principal_x = (width - 1) * 0.5
    principal_y = (height - 1) * 0.5
    pixel_y, pixel_x = np.mgrid[0:height, 0:width]
    local_rays = np.stack(
        (
            (pixel_x - principal_x) / config.focal_length_px,
            (pixel_y - principal_y) / config.focal_length_px,
            np.ones((height, width), dtype=np.float64),
        ),
        axis=-1,
    )
    world_rays = local_rays @ camera.camera_to_world.T
    ray_z = world_rays[..., 2]
    depth = np.maximum(0.0, -camera.center[2] / np.minimum(ray_z, -1.0e-6))
    for _ in range(9):
        x = camera.center[0] + depth * world_rays[..., 0]
        y = camera.center[1] + depth * world_rays[..., 1]
        elevation = terrain_elevation(x, y, config.detail_layers)
        depth = (elevation - camera.center[2]) / ray_z

    x = camera.center[0] + depth * world_rays[..., 0]
    y = camera.center[1] + depth * world_rays[..., 1]
    elevation = terrain_elevation(x, y, config.detail_layers)
    valid = (
        (depth > 0.0)
        & (np.abs(x) <= 0.5 * config.terrain_width_m)
        & (np.abs(y) <= 0.5 * config.terrain_height_m)
        & (np.abs(camera.center[2] + depth * ray_z - elevation) < 0.01)
    )

    epsilon = 0.025
    dz_dx = (
        terrain_elevation(x + epsilon, y, config.detail_layers)
        - terrain_elevation(x - epsilon, y, config.detail_layers)
    ) / (2.0 * epsilon)
    dz_dy = (
        terrain_elevation(x, y + epsilon, config.detail_layers)
        - terrain_elevation(x, y - epsilon, config.detail_layers)
    ) / (2.0 * epsilon)
    normals = normalize(np.stack((-dz_dx, -dz_dy, np.ones_like(dz_dx)), axis=-1))
    sun = normalize(np.array([0.45, -0.28, 0.85], dtype=np.float64))
    illumination = 0.58 + 0.42 * np.maximum(0.0, normals @ sun)
    colors = sample_dom(config, dom, x, y).astype(np.float64)
    rng = np.random.default_rng(config.seed + 1543 * (camera_index + 1))
    noise = rng.normal(0.0, config.sensor_noise_std, colors.shape)
    shaded = np.clip(colors * illumination[..., None] + noise, 0.0, 255.0).astype(np.uint8)

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., :] = (18, 21, 25)
    image[valid] = shaded[valid]
    depth_output = depth.astype(np.float32)
    depth_output[~valid] = np.nan
    mask = np.where(valid, 255, 0).astype(np.uint8)
    return image, depth_output, mask


def format_number(value: float) -> str:
    return format(float(value), ".17g")


def write_tsai(path: Path, config: DatasetConfig, camera: Camera) -> None:
    principal_x = (config.image_width - 1) * 0.5
    principal_y = (config.image_height - 1) * 0.5
    rotation = " ".join(format_number(value) for value in camera.camera_to_world.flat)
    center = " ".join(format_number(value) for value in camera.center)
    lines = [
        "VERSION_4", "PINHOLE",
        f"fu = {format_number(config.focal_length_px)}",
        f"fv = {format_number(config.focal_length_px)}",
        f"cu = {format_number(principal_x)}", f"cv = {format_number(principal_y)}",
        "u_direction = 1 0 0", "v_direction = 0 1 0", "w_direction = 0 0 1",
        f"C = {center}", f"R = {rotation}", "pitch = 1", "BrownConrady",
        "xp = 0", "yp = 0", "k1 = 0", "k2 = 0", "k3 = 0", "p1 = 0", "p2 = 0", "phi = 0", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_depth_preview(path: Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth)
    preview = np.zeros(depth.shape, dtype=np.uint16)
    if np.any(valid):
        minimum = float(np.min(depth[valid]))
        scale = 65534.0 / max(float(np.ptp(depth[valid])), 1.0e-6)
        preview[valid] = np.clip(1.0 + (depth[valid] - minimum) * scale, 1, 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), preview):
        raise OSError(f"failed to write depth preview: {path}")


def write_world_file(path: Path, config: DatasetConfig, height: int, width: int) -> None:
    pixel_x = config.terrain_width_m / width
    pixel_y = config.terrain_height_m / height
    lines = [pixel_x, 0.0, 0.0, -pixel_y, -0.5 * config.terrain_width_m + 0.5 * pixel_x, 0.5 * config.terrain_height_m - 0.5 * pixel_y]
    path.write_text("\n".join(format_number(value) for value in lines) + "\n", encoding="utf-8", newline="\n")


def write_mesh(path: Path, config: DatasetConfig, dom: np.ndarray) -> None:
    xx, yy, zz = terrain_grid(config, config.mesh_height, config.mesh_width)
    colors = sample_dom(config, dom, xx, yy)
    vertex_count = config.mesh_height * config.mesh_width
    face_count = 2 * (config.mesh_height - 1) * (config.mesh_width - 1)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {vertex_count}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write(f"element face {face_count}\nproperty list uchar int vertex_indices\nend_header\n")
        for x, y, z, color in zip(xx.flat, yy.flat, zz.flat, colors.reshape(-1, 3), strict=True):
            blue, green, red = (int(value) for value in color)
            stream.write(f"{x:.9g} {y:.9g} {z:.9g} {red} {green} {blue}\n")
        for row in range(config.mesh_height - 1):
            for column in range(config.mesh_width - 1):
                upper_left = row * config.mesh_width + column
                upper_right = upper_left + 1
                lower_left = upper_left + config.mesh_width
                lower_right = lower_left + 1
                stream.write(f"3 {upper_left} {lower_left} {upper_right}\n")
                stream.write(f"3 {upper_right} {lower_left} {lower_right}\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generated_file_records(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(output_dir).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(item for item in output_dir.rglob("*") if item.is_file())
        if path.name != "manifest.json"
    ]


def generate_dataset(config: DatasetConfig) -> dict[str, Any]:
    validate_config(config)
    output_dir = config.output_dir.resolve()
    ensure_empty_output(output_dir)
    images_dir = output_dir / "images"
    cameras_dir = output_dir / "cameras"
    truth_dir = output_dir / "ground_truth"
    depth_dir = truth_dir / "depth"
    mask_dir = truth_dir / "masks"
    mvs_masks_dir = output_dir / "mvs_masks"
    for directory in (
        images_dir,
        cameras_dir,
        truth_dir,
        depth_dir,
        mask_dir,
        mvs_masks_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    dom = build_dom(config)
    xx, yy, dem = terrain_grid(config, config.dem_height, config.dem_width)
    np.save(truth_dir / "dem.npy", dem.astype(np.float32), allow_pickle=False)
    if not cv2.imwrite(str(truth_dir / "dem.tif"), dem.astype(np.float32)):
        raise OSError("failed to write DEM TIFF")
    dem_preview = np.clip((dem - dem.min()) / max(float(np.ptp(dem)), 1.0e-6) * 65535.0, 0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(truth_dir / "dem_preview.png"), dem_preview):
        raise OSError("failed to write DEM preview")
    dem_color = cv2.applyColorMap(
        np.clip(dem_preview / 257, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    if not cv2.imwrite(str(truth_dir / "dem_color.png"), dem_color):
        raise OSError("failed to write color DEM preview")
    if not cv2.imwrite(str(truth_dir / "dom.png"), dom) or not cv2.imwrite(str(truth_dir / "dom.tif"), dom):
        raise OSError("failed to write DOM")
    write_world_file(truth_dir / "dem.tfw", config, config.dem_height, config.dem_width)
    write_world_file(truth_dir / "dom.tfw", config, config.texture_height, config.texture_width)
    local_wkt = 'LOCAL_CS["PlaScan synthetic terrain",LOCAL_DATUM["Synthetic",0],UNIT["metre",1]]\n'
    (truth_dir / "dem.prj").write_text(local_wkt, encoding="utf-8", newline="\n")
    (truth_dir / "dom.prj").write_text(local_wkt, encoding="utf-8", newline="\n")
    write_mesh(truth_dir / "terrain.ply", config, dom)

    camera_records: list[dict[str, Any]] = []
    list_lines: list[str] = []
    coverage_values: list[float] = []
    for camera_index, camera in enumerate(camera_grid(config)):
        image, depth, mask = render_view(config, camera, dom, camera_index)
        image_relative = Path("images") / f"{camera.camera_id}.png"
        camera_relative = Path("cameras") / f"{camera.camera_id}.tsai"
        depth_relative = Path("ground_truth/depth") / f"{camera.camera_id}.npy"
        preview_relative = Path("ground_truth/depth") / f"{camera.camera_id}_preview.png"
        mask_relative = Path("ground_truth/masks") / f"{camera.camera_id}.png"
        mvs_mask_relative = Path("mvs_masks") / f"{camera.camera_id}_mask.png"
        if not cv2.imwrite(str(output_dir / image_relative), image):
            raise OSError(f"failed to write image: {image_relative}")
        np.save(output_dir / depth_relative, depth, allow_pickle=False)
        write_depth_preview(output_dir / preview_relative, depth)
        if not cv2.imwrite(str(output_dir / mask_relative), mask):
            raise OSError(f"failed to write mask: {mask_relative}")
        if not cv2.imwrite(str(output_dir / mvs_mask_relative), 255 - mask):
            raise OSError(f"failed to write MVS exclusion mask: {mvs_mask_relative}")
        write_tsai(output_dir / camera_relative, config, camera)
        coverage = float(np.count_nonzero(mask)) / float(mask.size)
        coverage_values.append(coverage)
        list_lines.append(f"{image_relative.as_posix()} {camera_relative.as_posix()}")
        camera_records.append({
            "id": camera.camera_id, "image": image_relative.as_posix(), "camera": camera_relative.as_posix(),
            "depth_camera_z": depth_relative.as_posix(), "valid_mask": mask_relative.as_posix(),
            "mvs_exclusion_mask": mvs_mask_relative.as_posix(),
            "terrain_coverage": coverage, "center": camera.center.tolist(), "target": camera.target.tolist(),
            "camera_to_world": camera.camera_to_world.tolist(),
        })

    (output_dir / "image_camera.lis").write_text("\n".join(list_lines) + "\n", encoding="utf-8", newline="\n")
    manifest: dict[str, Any] = {
        "schema_version": 1, "dataset_kind": DATASET_KIND, "generator_version": GENERATOR_VERSION,
        "quality_level": config.quality_level, "seed": config.seed,
        "coordinate_system": {"name": "LOCAL_CS[synthetic terrain, metre]", "axis_order": ["x", "y", "z"], "unit": "metre"},
        "image": {"width": config.image_width, "height": config.image_height, "focal_length_px": config.focal_length_px,
                  "principal_point_px": [(config.image_width - 1) * 0.5, (config.image_height - 1) * 0.5],
                  "sensor_noise_std": config.sensor_noise_std},
        "terrain": {"width_m": config.terrain_width_m, "height_m": config.terrain_height_m,
                    "minimum_elevation_m": float(dem.min()), "maximum_elevation_m": float(dem.max()),
                    "detail_layers": config.detail_layers},
        "cameras": camera_records,
        "ground_truth": {"dem_npy": "ground_truth/dem.npy", "dem_path": "ground_truth/dem.tif",
                         "dom": "ground_truth/dom.tif", "mesh": "ground_truth/terrain.ply",
                         "dem_shape": [config.dem_height, config.dem_width],
                         "dem_pixel_size_m": [config.terrain_width_m / config.dem_width, config.terrain_height_m / config.dem_height],
                         "per_view_depth_definition": "positive camera z"},
        "summary": {"view_count": config.view_count, "minimum_terrain_coverage": min(coverage_values),
                    "maximum_terrain_coverage": max(coverage_values), "mean_terrain_coverage": sum(coverage_values) / len(coverage_values),
                    "list_file": "image_camera.lis"},
    }
    manifest["files"] = generated_file_records(output_dir)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality", choices=sorted(QUALITY_PRESETS), default="medium")
    parser.add_argument("--views", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--focal-length-px", type=float)
    parser.add_argument("--terrain-width-m", type=float, default=24.0)
    parser.add_argument("--terrain-height-m", type=float, default=18.0)
    parser.add_argument("--camera-height-m", type=float, default=19.0)
    parser.add_argument("--dem-height", type=int)
    parser.add_argument("--dem-width", type=int)
    parser.add_argument("--mesh-height", type=int)
    parser.add_argument("--mesh-width", type=int)
    parser.add_argument("--texture-height", type=int)
    parser.add_argument("--texture-width", type=int)
    parser.add_argument("--detail-layers", type=int)
    parser.add_argument("--sensor-noise-std", type=float)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> DatasetConfig:
    values = dict(QUALITY_PRESETS[args.quality])
    overrides = {"view_count": args.views, "image_width": args.width, "image_height": args.height,
                 "focal_length_px": args.focal_length_px, "dem_height": args.dem_height, "dem_width": args.dem_width,
                 "mesh_height": args.mesh_height, "mesh_width": args.mesh_width, "texture_height": args.texture_height,
                 "texture_width": args.texture_width, "detail_layers": args.detail_layers,
                 "sensor_noise_std": args.sensor_noise_std}
    values.update({key: value for key, value in overrides.items() if value is not None})
    return DatasetConfig(output_dir=args.output_dir, quality_level=args.quality,
                         terrain_width_m=args.terrain_width_m, terrain_height_m=args.terrain_height_m,
                         camera_height_m=args.camera_height_m, seed=args.seed, **values)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    try:
        manifest = generate_dataset(config)
    except (OSError, ValueError) as error:
        print(f"failed to generate synthetic terrain dataset: {error}")
        return 1
    print(f"dataset={config.output_dir.resolve()}")
    print(f"quality={config.quality_level}")
    print(f"views={manifest['summary']['view_count']}")
    print(f"mean_terrain_coverage={manifest['summary']['mean_terrain_coverage']:.6f}")
    print(f"manifest={(config.output_dir / 'manifest.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
