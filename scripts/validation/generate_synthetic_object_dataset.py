#!/usr/bin/env python3
"""Generate a deterministic multi-view 3D object dataset with independent truth."""

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


GENERATOR_VERSION = "1.1"
DATASET_KIND = "plascan_synthetic_3d_object"


QUALITY_PRESETS: dict[str, dict[str, int | float]] = {
    "coarse": {
        "view_count": 20,
        "image_width": 640,
        "image_height": 480,
        "focal_length_px": 620.0,
        "latitude_segments": 24,
        "longitude_segments": 48,
        "radial_map_height": 128,
        "radial_map_width": 256,
        "texture_height": 512,
        "texture_width": 1024,
        "texture_detail_layers": 2,
        "sensor_noise_std": 0.0,
    },
    "medium": {
        "view_count": 28,
        "image_width": 800,
        "image_height": 608,
        "focal_length_px": 775.0,
        "latitude_segments": 32,
        "longitude_segments": 64,
        "radial_map_height": 256,
        "radial_map_width": 512,
        "texture_height": 768,
        "texture_width": 1536,
        "texture_detail_layers": 3,
        "sensor_noise_std": 0.35,
    },
    "fine": {
        "view_count": 40,
        "image_width": 960,
        "image_height": 736,
        "focal_length_px": 930.0,
        "latitude_segments": 64,
        "longitude_segments": 128,
        "radial_map_height": 512,
        "radial_map_width": 1024,
        "texture_height": 1536,
        "texture_width": 3072,
        "texture_detail_layers": 5,
        "sensor_noise_std": 0.7,
    },
}


@dataclass(frozen=True)
class DatasetConfig:
    output_dir: Path
    quality_level: str = "medium"
    view_count: int = 28
    image_width: int = 800
    image_height: int = 608
    focal_length_px: float = 775.0
    reference_radius_m: float = 3.0
    camera_distance_m: float = 10.0
    latitude_segments: int = 32
    longitude_segments: int = 64
    radial_map_height: int = 256
    radial_map_width: int = 512
    texture_height: int = 768
    texture_width: int = 1536
    texture_detail_layers: int = 3
    sensor_noise_std: float = 0.35
    seed: int = 20260914


@dataclass(frozen=True)
class ObjectMesh:
    vertices: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    colors_bgr: np.ndarray


@dataclass(frozen=True)
class Camera:
    camera_id: str
    center: np.ndarray
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
    if config.reference_radius_m <= 0.0:
        raise ValueError("reference radius must be positive")
    if config.camera_distance_m <= 1.8 * config.reference_radius_m:
        raise ValueError("camera distance must exceed 1.8 times the object radius")
    if config.latitude_segments < 8 or config.longitude_segments < 16:
        raise ValueError("object mesh resolution is too small")
    if config.radial_map_height < 16 or config.radial_map_width < 32:
        raise ValueError("radial truth map resolution is too small")
    if config.texture_height < 32 or config.texture_width < 64:
        raise ValueError("texture resolution is too small")
    if config.texture_detail_layers < 1:
        raise ValueError("texture detail layers must be positive")
    if config.sensor_noise_std < 0.0:
        raise ValueError("sensor noise must not be negative")


def ensure_empty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def normalize(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(lengths, 1.0e-12)


def object_radius(directions: np.ndarray, reference_radius: float) -> np.ndarray:
    """Return an analytic, non-convex radial surface for unit directions."""
    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]
    deformation = (
        0.105 * (x * x - y * y)
        + 0.065 * np.sin(3.0 * np.arctan2(y, x) + 0.7) * (1.0 - z * z)
        + 0.045 * np.sin(5.0 * z + 1.4 * x)
        + 0.035 * np.cos(4.0 * y - 2.0 * z)
    )

    crater_centers = normalize(
        np.array(
            [
                [0.82, 0.31, 0.47],
                [-0.48, 0.79, 0.38],
                [0.24, -0.91, -0.34],
                [-0.71, -0.42, 0.56],
            ],
            dtype=np.float64,
        )
    )
    crater_depths = (0.115, 0.085, 0.075, 0.065)
    crater_widths = (0.035, 0.024, 0.020, 0.018)
    for center, depth, width in zip(
        crater_centers, crater_depths, crater_widths, strict=True
    ):
        angular_measure = 1.0 - np.clip(directions @ center, -1.0, 1.0)
        bowl = np.exp(-angular_measure / width)
        rim = np.exp(-((angular_measure - 2.3 * width) ** 2) / (0.42 * width) ** 2)
        deformation += -depth * bowl + 0.026 * rim
    return reference_radius * np.maximum(0.72, 1.0 + deformation)


def build_texture(height: int, width: int, seed: int, detail_layers: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros((height, width), dtype=np.float32)
    total_weight = 0.0
    for layer in range(detail_layers):
        layer_noise = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
        sigma = max(0.65, width / (90.0 * (2**layer)))
        weight = 0.58**layer
        noise += weight * cv2.GaussianBlur(layer_noise, (0, 0), sigma)
        total_weight += weight
    noise /= total_weight
    noise -= float(noise.min())
    noise /= max(float(noise.max()), 1.0e-6)

    # A fine, band-limited mineral grain gives SIFT stable local structure at
    # neighboring viewpoints.  Broad albedo fields alone look plausible but do
    # not provide enough shared tracks for production MVS neighbor selection.
    grain = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), max(0.8, width / 900.0))
    grain -= float(grain.min())
    grain /= max(float(grain.max()), 1.0e-6)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    bands = (
        0.5
        + 0.22 * np.sin(xx * 0.061 + yy * 0.023)
        + 0.14 * np.cos(xx * 0.019 - yy * 0.089)
    )
    texture = np.empty((height, width, 3), dtype=np.float32)
    texture[..., 0] = 48.0 + 82.0 * noise + 26.0 * bands + 24.0 * (grain - 0.5)
    texture[..., 1] = 58.0 + 96.0 * noise + 34.0 * bands + 27.0 * (grain - 0.5)
    texture[..., 2] = 66.0 + 112.0 * noise + 42.0 * bands + 30.0 * (grain - 0.5)
    texture = np.clip(texture, 0.0, 255.0).astype(np.uint8)

    marker_count = min(2400, max(256, height * width // 500))
    for index in range(marker_count):
        center = tuple(
            int(value)
            for value in (
                rng.integers(0, width),
                rng.integers(0, height),
            )
        )
        radius = int(rng.integers(max(1, width // 520), max(4, width // 105)))
        gray = int(rng.integers(35, 185))
        warm_tint = int(rng.integers(-13, 14))
        color = (
            int(np.clip(gray - warm_tint, 0, 255)),
            gray,
            int(np.clip(gray + warm_tint, 0, 255)),
        )
        thickness = -1 if index % 6 else max(1, width // 900)
        cv2.circle(texture, center, radius, color, thickness, cv2.LINE_AA)
    for _ in range(max(80, detail_layers * 40)):
        start = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        end = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        gray = int(rng.integers(32, 145))
        color = (gray, gray + int(rng.integers(0, 9)), gray + int(rng.integers(3, 15)))
        cv2.line(texture, start, end, color, max(1, width // 1100), cv2.LINE_AA)
    return texture


def apply_sensor_model(
    image: np.ndarray,
    mask: np.ndarray,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    if noise_std <= 0.0:
        return image
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.sqrt(
        ((xx - (width - 1) * 0.5) / max(width * 0.5, 1.0)) ** 2
        + ((yy - (height - 1) * 0.5) / max(height * 0.5, 1.0)) ** 2
    )
    vignette = np.clip(1.0 - 0.08 * radius * radius, 0.86, 1.0)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, image.shape).astype(np.float32)
    result = image.astype(np.float32) * vignette[..., None] + noise
    result[mask == 0] = image[mask == 0]
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def directions_to_texture_maps(
    directions: np.ndarray, texture: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    unit = normalize(directions)
    longitude = np.arctan2(unit[..., 1], unit[..., 0])
    latitude = np.arcsin(np.clip(unit[..., 2], -1.0, 1.0))
    map_x = ((longitude / (2.0 * np.pi) + 0.5) * (texture.shape[1] - 1)).astype(
        np.float32
    )
    map_y = ((0.5 - latitude / np.pi) * (texture.shape[0] - 1)).astype(
        np.float32
    )
    return map_x, map_y


def sample_texture(texture: np.ndarray, directions: np.ndarray) -> np.ndarray:
    map_x, map_y = directions_to_texture_maps(directions, texture)
    return cv2.remap(
        texture,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def build_object_mesh(config: DatasetConfig, texture: np.ndarray) -> ObjectMesh:
    directions: list[list[float]] = [[0.0, 0.0, 1.0]]
    for latitude_index in range(1, config.latitude_segments):
        latitude = math.pi * 0.5 - math.pi * latitude_index / config.latitude_segments
        cos_latitude = math.cos(latitude)
        for longitude_index in range(config.longitude_segments):
            longitude = 2.0 * math.pi * longitude_index / config.longitude_segments
            directions.append(
                [
                    cos_latitude * math.cos(longitude),
                    cos_latitude * math.sin(longitude),
                    math.sin(latitude),
                ]
            )
    south_index = len(directions)
    directions.append([0.0, 0.0, -1.0])
    unit_directions = np.asarray(directions, dtype=np.float64)
    radii = object_radius(unit_directions, config.reference_radius_m)
    vertices = unit_directions * radii[:, None]

    faces: list[list[int]] = []
    first_ring = 1
    for longitude_index in range(config.longitude_segments):
        current = first_ring + longitude_index
        following = first_ring + (longitude_index + 1) % config.longitude_segments
        faces.append([0, current, following])
    for ring_index in range(config.latitude_segments - 2):
        upper = first_ring + ring_index * config.longitude_segments
        lower = upper + config.longitude_segments
        for longitude_index in range(config.longitude_segments):
            upper_left = upper + longitude_index
            upper_right = upper + (longitude_index + 1) % config.longitude_segments
            lower_left = lower + longitude_index
            lower_right = lower + (longitude_index + 1) % config.longitude_segments
            faces.append([upper_left, lower_left, upper_right])
            faces.append([upper_right, lower_left, lower_right])
    last_ring = first_ring + (config.latitude_segments - 2) * config.longitude_segments
    for longitude_index in range(config.longitude_segments):
        current = last_ring + longitude_index
        following = last_ring + (longitude_index + 1) % config.longitude_segments
        faces.append([current, south_index, following])

    face_array = np.asarray(faces, dtype=np.int32)
    for index, face in enumerate(face_array):
        a, b, c = vertices[face]
        if float(np.dot(np.cross(b - a, c - a), a + b + c)) < 0.0:
            face_array[index, 1], face_array[index, 2] = (
                face_array[index, 2],
                face_array[index, 1],
            )

    normals = np.zeros_like(vertices)
    for face in face_array:
        a, b, c = vertices[face]
        face_normal = np.cross(b - a, c - a)
        normals[face] += face_normal
    normals = normalize(normals)
    colors = sample_texture(texture, unit_directions.reshape(-1, 1, 3)).reshape(-1, 3)
    return ObjectMesh(vertices, normals, face_array, colors)


def look_at_camera_to_world(center: np.ndarray) -> np.ndarray:
    forward = -center
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1.0e-8:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    return np.column_stack((right, down, forward))


def camera_orbit(config: DatasetConfig) -> list[Camera]:
    cameras: list[Camera] = []
    for index in range(config.view_count):
        # Start away from the equirectangular texture seam so the first frame,
        # which is a boundary case for sequential pair planning, has robust
        # forward and wraparound neighbors.
        azimuth = 2.0 * math.pi * (index / config.view_count + 0.25)
        # Keep consecutive frames spatially adjacent so both feature matching and
        # MVS neighbor selection see a continuous orbit while still covering the
        # object's northern and southern hemispheres.
        elevation = math.radians(18.0) * math.sin(azimuth)
        center = config.camera_distance_m * np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float64,
        )
        cameras.append(
            Camera(
                camera_id=f"view_{index:02d}",
                center=center,
                camera_to_world=look_at_camera_to_world(center),
            )
        )
    return cameras


def render_view(
    config: DatasetConfig,
    mesh: ObjectMesh,
    camera: Camera,
    texture: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = config.image_height, config.image_width
    principal_x = (width - 1) * 0.5
    principal_y = (height - 1) * 0.5
    camera_vertices = (mesh.vertices - camera.center) @ camera.camera_to_world
    camera_normals = mesh.normals @ camera.camera_to_world
    screen = np.empty((len(mesh.vertices), 2), dtype=np.float64)
    screen[:, 0] = config.focal_length_px * camera_vertices[:, 0] / camera_vertices[:, 2] + principal_x
    screen[:, 1] = config.focal_length_px * camera_vertices[:, 1] / camera_vertices[:, 2] + principal_y

    depth = np.full((height, width), np.inf, dtype=np.float64)
    world_positions = np.full((height, width, 3), np.nan, dtype=np.float64)
    world_normals = np.full((height, width, 3), np.nan, dtype=np.float64)

    for face in mesh.faces:
        camera_triangle = camera_vertices[face]
        if np.any(camera_triangle[:, 2] <= 1.0e-6):
            continue
        face_normal = np.cross(
            camera_triangle[1] - camera_triangle[0],
            camera_triangle[2] - camera_triangle[0],
        )
        if float(np.dot(face_normal, -np.mean(camera_triangle, axis=0))) <= 0.0:
            continue

        triangle = screen[face]
        min_x = max(0, int(math.floor(float(np.min(triangle[:, 0])))))
        max_x = min(width - 1, int(math.ceil(float(np.max(triangle[:, 0])))))
        min_y = max(0, int(math.floor(float(np.min(triangle[:, 1])))))
        max_y = min(height - 1, int(math.ceil(float(np.max(triangle[:, 1])))))
        if min_x > max_x or min_y > max_y:
            continue

        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denominator)) < 1.0e-12:
            continue
        pixel_y, pixel_x = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        sample_x = pixel_x + 0.5
        sample_y = pixel_y + 0.5
        weight0 = ((y1 - y2) * (sample_x - x2) + (x2 - x1) * (sample_y - y2)) / denominator
        weight1 = ((y2 - y0) * (sample_x - x2) + (x0 - x2) * (sample_y - y2)) / denominator
        weight2 = 1.0 - weight0 - weight1
        inside = (weight0 >= -1.0e-7) & (weight1 >= -1.0e-7) & (weight2 >= -1.0e-7)
        if not np.any(inside):
            continue

        reciprocal_z = 1.0 / camera_triangle[:, 2]
        interpolated_reciprocal_z = (
            weight0 * reciprocal_z[0]
            + weight1 * reciprocal_z[1]
            + weight2 * reciprocal_z[2]
        )
        candidate_depth = 1.0 / interpolated_reciprocal_z
        target_depth = depth[min_y : max_y + 1, min_x : max_x + 1]
        update = inside & (candidate_depth < target_depth)
        if not np.any(update):
            continue

        weights = np.stack((weight0, weight1, weight2), axis=-1)
        perspective_weights = weights * reciprocal_z
        perspective_weights /= np.sum(perspective_weights, axis=-1, keepdims=True)
        positions = perspective_weights @ mesh.vertices[face]
        normals = normalize(perspective_weights @ mesh.normals[face])
        target_depth[update] = candidate_depth[update]
        target_positions = world_positions[min_y : max_y + 1, min_x : max_x + 1]
        target_normals = world_normals[min_y : max_y + 1, min_x : max_x + 1]
        target_positions[update] = positions[update]
        target_normals[update] = normals[update]

    valid = np.isfinite(depth)
    image = np.empty((height, width, 3), dtype=np.uint8)
    vertical_gradient = np.linspace(0.0, 1.0, height, dtype=np.float64)[:, None]
    image[..., 0] = np.clip(13.0 + 7.0 * vertical_gradient, 0, 255).astype(np.uint8)
    image[..., 1] = np.clip(16.0 + 8.0 * vertical_gradient, 0, 255).astype(np.uint8)
    image[..., 2] = np.clip(21.0 + 10.0 * vertical_gradient, 0, 255).astype(np.uint8)
    if np.any(valid):
        colors = sample_texture(texture, world_positions)
        sun = normalize(np.array([0.55, -0.31, 0.77], dtype=np.float64))
        illumination = 0.50 + 0.50 * np.maximum(
            0.0, np.nan_to_num(world_normals, nan=0.0) @ sun
        )
        shaded = np.clip(colors.astype(np.float64) * illumination[..., None], 0, 255).astype(np.uint8)
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
        "VERSION_4",
        "PINHOLE",
        f"fu = {format_number(config.focal_length_px)}",
        f"fv = {format_number(config.focal_length_px)}",
        f"cu = {format_number(principal_x)}",
        f"cv = {format_number(principal_y)}",
        "u_direction = 1 0 0",
        "v_direction = 0 1 0",
        "w_direction = 0 0 1",
        f"C = {center}",
        f"R = {rotation}",
        "pitch = 1",
        "BrownConrady",
        "xp = 0",
        "yp = 0",
        "k1 = 0",
        "k2 = 0",
        "k3 = 0",
        "p1 = 0",
        "p2 = 0",
        "phi = 0",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_depth_preview(path: Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth)
    preview = np.zeros(depth.shape, dtype=np.uint16)
    if np.any(valid):
        minimum = float(np.min(depth[valid]))
        maximum = float(np.max(depth[valid]))
        scale = 65535.0 / max(maximum - minimum, 1.0e-6)
        preview[valid] = np.clip((depth[valid] - minimum) * scale, 1, 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), preview):
        raise OSError(f"failed to write depth preview: {path}")


def write_truth_mesh(path: Path, mesh: ObjectMesh) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(mesh.vertices)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property float nx\nproperty float ny\nproperty float nz\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write(f"element face {len(mesh.faces)}\n")
        stream.write("property list uchar int vertex_indices\nend_header\n")
        for vertex, normal, color in zip(
            mesh.vertices, mesh.normals, mesh.colors_bgr, strict=True
        ):
            blue, green, red = (int(value) for value in color)
            stream.write(
                f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} "
                f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g} "
                f"{red} {green} {blue}\n"
            )
        for face in mesh.faces:
            stream.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def radial_truth(
    config: DatasetConfig, texture: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    latitudes = np.linspace(
        math.pi * 0.5,
        -math.pi * 0.5,
        config.radial_map_height,
        dtype=np.float64,
    )
    longitudes = np.linspace(
        -math.pi,
        math.pi,
        config.radial_map_width,
        endpoint=False,
        dtype=np.float64,
    )
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    directions = np.stack(
        (
            np.cos(latitude_grid) * np.cos(longitude_grid),
            np.cos(latitude_grid) * np.sin(longitude_grid),
            np.sin(latitude_grid),
        ),
        axis=-1,
    )
    radius = object_radius(directions, config.reference_radius_m).astype(np.float32)
    dom = sample_texture(texture, directions)
    return radius, dom


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generated_file_records(output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def generate_dataset(config: DatasetConfig) -> dict[str, Any]:
    validate_config(config)
    output_dir = config.output_dir.resolve()
    ensure_empty_output(output_dir)
    images_dir = output_dir / "images"
    cameras_dir = output_dir / "cameras"
    truth_dir = output_dir / "ground_truth"
    depths_dir = truth_dir / "depth"
    masks_dir = truth_dir / "masks"
    mvs_masks_dir = output_dir / "mvs_masks"
    for directory in (
        images_dir,
        cameras_dir,
        truth_dir,
        depths_dir,
        masks_dir,
        mvs_masks_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    texture = build_texture(
        config.texture_height,
        config.texture_width,
        config.seed,
        config.texture_detail_layers,
    )
    if not cv2.imwrite(str(truth_dir / "radial_albedo.png"), texture):
        raise OSError("failed to write radial albedo texture")
    mesh = build_object_mesh(config, texture)
    write_truth_mesh(truth_dir / "object.ply", mesh)
    radius_map, radial_dom = radial_truth(config, texture)
    np.save(truth_dir / "radial_radius.npy", radius_map, allow_pickle=False)
    np.save(
        truth_dir / "radial_elevation.npy",
        radius_map - config.reference_radius_m,
        allow_pickle=False,
    )
    if not cv2.imwrite(str(truth_dir / "radial_dom.png"), radial_dom):
        raise OSError("failed to write radial DOM")

    list_lines: list[str] = []
    camera_records: list[dict[str, Any]] = []
    coverage_values: list[float] = []
    for camera_index, camera in enumerate(camera_orbit(config)):
        image, depth, mask = render_view(config, mesh, camera, texture)
        image = apply_sensor_model(
            image,
            mask,
            config.sensor_noise_std,
            config.seed + 1009 * (camera_index + 1),
        )
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
        camera_records.append(
            {
                "id": camera.camera_id,
                "image": image_relative.as_posix(),
                "camera": camera_relative.as_posix(),
                "depth_camera_z": depth_relative.as_posix(),
                "valid_mask": mask_relative.as_posix(),
                "mvs_exclusion_mask": mvs_mask_relative.as_posix(),
                "silhouette_coverage": coverage,
                "center": camera.center.tolist(),
                "camera_to_world": camera.camera_to_world.tolist(),
            }
        )

    (output_dir / "image_camera.lis").write_text(
        "\n".join(list_lines) + "\n", encoding="utf-8", newline="\n"
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_kind": DATASET_KIND,
        "generator_version": GENERATOR_VERSION,
        "quality_level": config.quality_level,
        "seed": config.seed,
        "coordinate_system": {
            "name": "LOCAL_CS[synthetic object, metre]",
            "axis_order": ["x", "y", "z"],
            "unit": "metre",
            "object_center": [0.0, 0.0, 0.0],
        },
        "image": {
            "width": config.image_width,
            "height": config.image_height,
            "focal_length_px": config.focal_length_px,
            "principal_point_px": [
                (config.image_width - 1) * 0.5,
                (config.image_height - 1) * 0.5,
            ],
            "sensor_noise_std": config.sensor_noise_std,
        },
        "object": {
            "surface": "analytic_irregular_radial_object_v1",
            "reference_radius_m": config.reference_radius_m,
            "mesh_vertices": int(len(mesh.vertices)),
            "mesh_faces": int(len(mesh.faces)),
            "texture_detail_layers": config.texture_detail_layers,
        },
        "cameras": camera_records,
        "ground_truth": {
            "mesh": "ground_truth/object.ply",
            "radial_radius": "ground_truth/radial_radius.npy",
            "radial_elevation": "ground_truth/radial_elevation.npy",
            "radial_dom": "ground_truth/radial_dom.png",
            "radial_map_shape": [config.radial_map_height, config.radial_map_width],
            "per_view_depth_definition": "positive camera z",
        },
        "summary": {
            "view_count": config.view_count,
            "minimum_silhouette_coverage": min(coverage_values),
            "maximum_silhouette_coverage": max(coverage_values),
            "mean_silhouette_coverage": sum(coverage_values) / len(coverage_values),
            "list_file": "image_camera.lis",
        },
    }
    manifest["files"] = generated_file_records(output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality", choices=sorted(QUALITY_PRESETS), default="medium")
    parser.add_argument("--views", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--focal-length-px", type=float)
    parser.add_argument("--reference-radius-m", type=float, default=3.0)
    parser.add_argument("--camera-distance-m", type=float, default=10.0)
    parser.add_argument("--latitude-segments", type=int)
    parser.add_argument("--longitude-segments", type=int)
    parser.add_argument("--radial-map-height", type=int)
    parser.add_argument("--radial-map-width", type=int)
    parser.add_argument("--texture-height", type=int)
    parser.add_argument("--texture-width", type=int)
    parser.add_argument("--texture-detail-layers", type=int)
    parser.add_argument("--sensor-noise-std", type=float)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> DatasetConfig:
    values = dict(QUALITY_PRESETS[args.quality])
    overrides = {
        "view_count": args.views,
        "image_width": args.width,
        "image_height": args.height,
        "focal_length_px": args.focal_length_px,
        "latitude_segments": args.latitude_segments,
        "longitude_segments": args.longitude_segments,
        "radial_map_height": args.radial_map_height,
        "radial_map_width": args.radial_map_width,
        "texture_height": args.texture_height,
        "texture_width": args.texture_width,
        "texture_detail_layers": args.texture_detail_layers,
        "sensor_noise_std": args.sensor_noise_std,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    return DatasetConfig(
        output_dir=args.output_dir,
        quality_level=args.quality,
        reference_radius_m=args.reference_radius_m,
        camera_distance_m=args.camera_distance_m,
        seed=args.seed,
        **values,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    try:
        manifest = generate_dataset(config)
    except (OSError, ValueError) as error:
        print(f"failed to generate synthetic 3D object dataset: {error}")
        return 1
    print(f"dataset={config.output_dir.resolve()}")
    print(f"quality={config.quality_level}")
    print(f"views={manifest['summary']['view_count']}")
    print(
        "mean_silhouette_coverage="
        f"{manifest['summary']['mean_silhouette_coverage']:.6f}"
    )
    print(f"manifest={(config.output_dir / 'manifest.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
