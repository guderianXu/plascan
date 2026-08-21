"""Deterministic perspective-correct CPU rasterizer used by the evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import TexturedMesh
from .sampling import sample_texture_bilinear


@dataclass(frozen=True)
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    rotation_world_to_camera: np.ndarray
    camera_center: np.ndarray

    @classmethod
    def from_json(cls, value: dict[str, object]) -> "PinholeCamera":
        required = ("fx", "fy", "cx", "cy", "rotation_world_to_camera", "camera_center")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"Camera model is missing fields: {missing}")
        distortion_fields = ("k1", "k2", "k3", "p1", "p2", "radial_k1", "radial_k2", "radial_k3")
        nonzero_distortion = {
            key: float(value[key])
            for key in distortion_fields
            if key in value and abs(float(value[key])) > 1.0e-15
        }
        if nonzero_distortion:
            raise ValueError(
                "Held-out rasterization requires a prepared zero-distortion image/camera; "
                f"received {nonzero_distortion}"
            )
        rotation = np.asarray(value["rotation_world_to_camera"], dtype=np.float64)
        center = np.asarray(value["camera_center"], dtype=np.float64)
        if rotation.size != 9 or center.size != 3:
            raise ValueError("Camera rotation must have 9 values and camera_center must have 3")
        rotation = rotation.reshape(3, 3)
        center = center.reshape(3)
        scalars = np.asarray([value["fx"], value["fy"], value["cx"], value["cy"]], dtype=np.float64)
        if not np.all(np.isfinite(scalars)) or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(center)):
            raise ValueError("Camera model contains non-finite values")
        if scalars[0] <= 0.0 or scalars[1] <= 0.0:
            raise ValueError("Camera focal lengths must be positive")
        orthogonality_error = np.max(np.abs(rotation @ rotation.T - np.eye(3)))
        if orthogonality_error > 1.0e-5 or np.linalg.det(rotation) < 0.0:
            raise ValueError(f"Camera rotation is invalid (orthogonality error {orthogonality_error:.3g})")
        return cls(*map(float, scalars), rotation, center)

    def scaled(self, scale_x: float, scale_y: float) -> "PinholeCamera":
        if scale_x <= 0.0 or scale_y <= 0.0:
            raise ValueError("Camera scale must be positive")
        return PinholeCamera(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=(self.cx + 0.5) * scale_x - 0.5,
            cy=(self.cy + 0.5) * scale_y - 0.5,
            rotation_world_to_camera=self.rotation_world_to_camera,
            camera_center=self.camera_center,
        )


@dataclass(frozen=True)
class RenderResult:
    linear_rgb: np.ndarray
    depth: np.ndarray
    face_ids: np.ndarray
    rendered_face_count: int
    backface_count: int
    near_clipped_face_count: int


def _clip_triangle_to_near_plane(
    camera_triangle: np.ndarray,
    triangle_uv: np.ndarray,
    near_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip a camera-space triangle while carrying its affine UV attributes."""
    clipped_vertices: list[np.ndarray] = []
    clipped_uvs: list[np.ndarray] = []
    previous_vertex = camera_triangle[-1]
    previous_uv = triangle_uv[-1]
    previous_inside = bool(previous_vertex[2] >= near_depth)

    for current_vertex, current_uv in zip(camera_triangle, triangle_uv, strict=True):
        current_inside = bool(current_vertex[2] >= near_depth)
        if current_inside != previous_inside:
            fraction = (near_depth - previous_vertex[2]) / (
                current_vertex[2] - previous_vertex[2]
            )
            intersection = previous_vertex + fraction * (current_vertex - previous_vertex)
            intersection = intersection.copy()
            intersection[2] = near_depth
            clipped_vertices.append(intersection)
            clipped_uvs.append(previous_uv + fraction * (current_uv - previous_uv))
        if current_inside:
            clipped_vertices.append(current_vertex.copy())
            clipped_uvs.append(current_uv.copy())
        previous_vertex = current_vertex
        previous_uv = current_uv
        previous_inside = current_inside

    if not clipped_vertices:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.asarray(clipped_vertices, dtype=np.float64), np.asarray(clipped_uvs, dtype=np.float64)


def render_textured_mesh(
    mesh: TexturedMesh,
    camera: PinholeCamera,
    width: int,
    height: int,
    *,
    near_depth: float = 1.0e-6,
    cull_backfaces: bool = False,
) -> RenderResult:
    if width <= 0 or height <= 0:
        raise ValueError("Render dimensions must be positive")
    if near_depth <= 0.0:
        raise ValueError("near_depth must be positive")

    camera_vertices = (camera.rotation_world_to_camera @ (mesh.vertices - camera.camera_center).T).T
    output = np.zeros((height, width, 3), dtype=np.float32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float64)
    face_ids = np.full((height, width), -1, dtype=np.int32)
    rendered_faces = 0
    backfaces = 0
    near_clipped = 0

    for face_index, vertex_indices in enumerate(mesh.faces):
        camera_triangle = camera_vertices[vertex_indices]
        texture_indices = mesh.face_texcoords[face_index]
        triangle_uv = mesh.texcoords[texture_indices]
        if np.any(camera_triangle[:, 2] < near_depth):
            near_clipped += 1
        clipped_vertices, clipped_uvs = _clip_triangle_to_near_plane(
            camera_triangle,
            triangle_uv,
            near_depth,
        )
        if len(clipped_vertices) < 3:
            continue

        normal = np.cross(camera_triangle[1] - camera_triangle[0], camera_triangle[2] - camera_triangle[0])
        centroid = np.mean(camera_triangle, axis=0)
        is_backface = float(np.dot(normal, centroid)) >= 0.0
        backfaces += int(is_backface)
        if cull_backfaces and is_backface:
            continue

        material = mesh.materials[int(mesh.face_materials[face_index])]
        face_rendered = False
        for offset in range(1, len(clipped_vertices) - 1):
            fragment_indices = [0, offset, offset + 1]
            fragment_vertices = clipped_vertices[fragment_indices]
            fragment_uvs = clipped_uvs[fragment_indices]
            face_depths = fragment_vertices[:, 2]
            triangle = np.empty((3, 2), dtype=np.float64)
            triangle[:, 0] = camera.fx * fragment_vertices[:, 0] / face_depths + camera.cx
            triangle[:, 1] = camera.fy * fragment_vertices[:, 1] / face_depths + camera.cy
            if not np.all(np.isfinite(triangle)):
                continue
            if (
                np.max(triangle[:, 0]) < 0.0
                or np.max(triangle[:, 1]) < 0.0
                or np.min(triangle[:, 0]) > width - 1.0
                or np.min(triangle[:, 1]) > height - 1.0
            ):
                continue
            minimum_x = max(0, int(np.ceil(np.min(triangle[:, 0]) - 0.5)))
            maximum_x = min(width - 1, int(np.floor(np.max(triangle[:, 0]) + 0.5)))
            minimum_y = max(0, int(np.ceil(np.min(triangle[:, 1]) - 0.5)))
            maximum_y = min(height - 1, int(np.floor(np.max(triangle[:, 1]) + 0.5)))
            if minimum_x > maximum_x or minimum_y > maximum_y:
                continue

            x0, y0 = triangle[0]
            x1, y1 = triangle[1]
            x2, y2 = triangle[2]
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if not np.isfinite(denominator) or abs(denominator) <= 1.0e-12:
                continue
            grid_y, grid_x = np.mgrid[minimum_y : maximum_y + 1, minimum_x : maximum_x + 1]
            lambda0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
            lambda1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
            lambda2 = 1.0 - lambda0 - lambda1
            inside = (lambda0 >= -1.0e-9) & (lambda1 >= -1.0e-9) & (lambda2 >= -1.0e-9)
            if not np.any(inside):
                continue

            inverse_depth = (
                lambda0 / face_depths[0] + lambda1 / face_depths[1] + lambda2 / face_depths[2]
            )
            candidate_depth = np.divide(
                1.0,
                inverse_depth,
                out=np.full_like(inverse_depth, np.inf),
                where=inverse_depth > 0.0,
            )
            depth_region = depth_buffer[minimum_y : maximum_y + 1, minimum_x : maximum_x + 1]
            update = inside & (candidate_depth < depth_region)
            if not np.any(update):
                continue

            numerator_u = (
                lambda0 * fragment_uvs[0, 0] / face_depths[0]
                + lambda1 * fragment_uvs[1, 0] / face_depths[1]
                + lambda2 * fragment_uvs[2, 0] / face_depths[2]
            )
            numerator_v = (
                lambda0 * fragment_uvs[0, 1] / face_depths[0]
                + lambda1 * fragment_uvs[1, 1] / face_depths[1]
                + lambda2 * fragment_uvs[2, 1] / face_depths[2]
            )
            uv = np.stack(
                [numerator_u[update] / inverse_depth[update], numerator_v[update] / inverse_depth[update]],
                axis=1,
            )
            colors = sample_texture_bilinear(material.texture_linear_rgb, uv)
            region_rgb = output[minimum_y : maximum_y + 1, minimum_x : maximum_x + 1]
            region_faces = face_ids[minimum_y : maximum_y + 1, minimum_x : maximum_x + 1]
            depth_region[update] = candidate_depth[update]
            region_rgb[update] = colors
            region_faces[update] = face_index
            face_rendered = True

        rendered_faces += int(face_rendered)

    return RenderResult(output, depth_buffer, face_ids, rendered_faces, backfaces, near_clipped)
