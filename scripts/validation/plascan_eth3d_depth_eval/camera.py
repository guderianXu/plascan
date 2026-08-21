"""Exact COLMAP PINHOLE and THIN_PRISM_FISHEYE projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera_io import (
    ColmapCamera,
    colmap_pixels_to_raster_coordinates,
    raster_indices_to_colmap_pixels,
)

# COLMAP parameter order: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1.


class ProjectionError(ValueError):
    """Raised when a camera projection cannot satisfy its accuracy contract."""


@dataclass(frozen=True)
class ProjectionResult:
    pixels: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class InverseProjectionResult:
    rays: np.ndarray
    converged: np.ndarray
    reprojection_error_pixels: np.ndarray
    iterations: np.ndarray


def project_pinhole(camera: ColmapCamera, rays: np.ndarray) -> ProjectionResult:
    if camera.model != "PINHOLE":
        raise ValueError(f"Expected PINHOLE camera, got {camera.model}")
    rays = _point_array(rays, 3, "rays")
    z = rays[..., 2]
    valid = np.all(np.isfinite(rays), axis=-1) & (z > np.finfo(np.float64).eps)
    safe_z = np.where(valid, z, 1.0)
    pixels = np.stack(
        (
            camera.fx * rays[..., 0] / safe_z + camera.cx,
            camera.fy * rays[..., 1] / safe_z + camera.cy,
        ),
        axis=-1,
    )
    pixels = np.where(valid[..., None], pixels, np.nan)
    return ProjectionResult(pixels=pixels, valid=valid)


def unproject_pinhole(camera: ColmapCamera, pixels: np.ndarray) -> np.ndarray:
    if camera.model != "PINHOLE":
        raise ValueError(f"Expected PINHOLE camera, got {camera.model}")
    pixels = _point_array(pixels, 2, "pixels")
    rays = np.stack(
        (
            (pixels[..., 0] - camera.cx) / camera.fx,
            (pixels[..., 1] - camera.cy) / camera.fy,
            np.ones(pixels.shape[:-1], dtype=np.float64),
        ),
        axis=-1,
    )
    rays[~np.all(np.isfinite(pixels), axis=-1)] = np.nan
    return rays


def project_thin_prism_fisheye(
    camera: ColmapCamera,
    rays: np.ndarray,
) -> ProjectionResult:
    """Project camera rays using COLMAP's THIN_PRISM_FISHEYE equations."""

    if camera.model != "THIN_PRISM_FISHEYE":
        raise ValueError(f"Expected THIN_PRISM_FISHEYE camera, got {camera.model}")
    rays = _point_array(rays, 3, "rays")
    z = rays[..., 2]
    valid = np.all(np.isfinite(rays), axis=-1) & (z > np.finfo(np.float64).eps)
    safe_z = np.where(valid, z, 1.0)
    normal_u = rays[..., 0] / safe_z
    normal_v = rays[..., 1] / safe_z
    fisheye_u, fisheye_v = _fisheye_from_normal(normal_u, normal_v)
    distorted_u, distorted_v = _distort_fisheye(camera, fisheye_u, fisheye_v)
    pixels = np.stack(
        (
            camera.fx * distorted_u + camera.cx,
            camera.fy * distorted_v + camera.cy,
        ),
        axis=-1,
    )
    valid &= np.all(np.isfinite(pixels), axis=-1)
    pixels = np.where(valid[..., None], pixels, np.nan)
    return ProjectionResult(pixels=pixels, valid=valid)


def unproject_thin_prism_fisheye(
    camera: ColmapCamera,
    pixels: np.ndarray,
    *,
    maximum_iterations: int = 100,
    step_tolerance: float = 1.0e-13,
    maximum_reprojection_error: float = 1.0e-6,
    require_all: bool = True,
) -> InverseProjectionResult:
    """Invert THIN_PRISM_FISHEYE with safeguarded Newton iterations.

    A sample is accepted only after the Newton step converges *and* projecting
    the recovered ray through the forward model returns to the input pixel
    within ``maximum_reprojection_error`` pixels.
    """

    if camera.model != "THIN_PRISM_FISHEYE":
        raise ValueError(f"Expected THIN_PRISM_FISHEYE camera, got {camera.model}")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    if not np.isfinite(step_tolerance) or step_tolerance <= 0.0:
        raise ValueError("step_tolerance must be finite and positive")
    if (
        not np.isfinite(maximum_reprojection_error)
        or maximum_reprojection_error <= 0.0
    ):
        raise ValueError("maximum_reprojection_error must be finite and positive")

    pixels = _point_array(pixels, 2, "pixels")
    original_shape = pixels.shape[:-1]
    flat_pixels = pixels.reshape(-1, 2)
    target_u = (flat_pixels[:, 0] - camera.cx) / camera.fx
    target_v = (flat_pixels[:, 1] - camera.cy) / camera.fy
    fisheye_u = target_u.copy()
    fisheye_v = target_v.copy()
    finite = np.all(np.isfinite(flat_pixels), axis=1)
    active = finite.copy()
    converged = np.zeros(flat_pixels.shape[0], dtype=bool)
    iterations = np.zeros(flat_pixels.shape[0], dtype=np.int32)

    for iteration in range(1, maximum_iterations + 1):
        if not np.any(active):
            break
        indices = np.flatnonzero(active)
        u = fisheye_u[indices]
        v = fisheye_v[indices]
        distorted_u, distorted_v, jacobian = _distort_fisheye_with_jacobian(
            camera, u, v
        )
        residual_u = distorted_u - target_u[indices]
        residual_v = distorted_v - target_v[indices]
        determinant = (
            jacobian[:, 0, 0] * jacobian[:, 1, 1]
            - jacobian[:, 0, 1] * jacobian[:, 1, 0]
        )
        solvable = np.isfinite(determinant) & (np.abs(determinant) > 1.0e-18)
        step_u = np.full(u.shape, np.nan, dtype=np.float64)
        step_v = np.full(v.shape, np.nan, dtype=np.float64)
        step_u[solvable] = (
            jacobian[solvable, 1, 1] * residual_u[solvable]
            - jacobian[solvable, 0, 1] * residual_v[solvable]
        ) / determinant[solvable]
        step_v[solvable] = (
            -jacobian[solvable, 1, 0] * residual_u[solvable]
            + jacobian[solvable, 0, 0] * residual_v[solvable]
        ) / determinant[solvable]

        step_norm = np.hypot(step_u, step_v)
        trust_radius = np.maximum(0.1 * np.hypot(u, v), 0.1)
        clipped = step_norm > trust_radius
        step_scale = np.ones(step_norm.shape, dtype=np.float64)
        step_scale[clipped] = trust_radius[clipped] / step_norm[clipped]
        step_u *= step_scale
        step_v *= step_scale
        next_u = u - step_u
        next_v = v - step_v
        usable = solvable & np.isfinite(next_u) & np.isfinite(next_v)
        fisheye_u[indices[usable]] = next_u[usable]
        fisheye_v[indices[usable]] = next_v[usable]
        iterations[indices] = iteration

        next_distorted_u, next_distorted_v = _distort_fisheye(
            camera,
            next_u,
            next_v,
        )
        residual_pixels = np.hypot(
            camera.fx * (next_distorted_u - target_u[indices]),
            camera.fy * (next_distorted_v - target_v[indices]),
        )
        newly_converged = (
            usable
            & (np.hypot(step_u, step_v) <= step_tolerance)
            & (residual_pixels <= maximum_reprojection_error)
        )
        converged[indices[newly_converged]] = True
        active[indices[~usable | newly_converged]] = False

    normal_u, normal_v, ray_valid = _normal_from_fisheye(
        fisheye_u,
        fisheye_v,
    )
    rays = np.stack(
        (normal_u, normal_v, np.ones(normal_u.shape, dtype=np.float64)), axis=-1
    )
    rays[~ray_valid] = np.nan
    reprojection = project_thin_prism_fisheye(camera, rays)
    reprojection_error = np.linalg.norm(
        reprojection.pixels.reshape(-1, 2) - flat_pixels,
        axis=1,
    )
    converged &= (
        ray_valid
        & reprojection.valid.reshape(-1)
        & np.isfinite(reprojection_error)
        & (reprojection_error <= maximum_reprojection_error)
    )
    rays[~converged] = np.nan

    if require_all and not np.all(converged):
        failed_count = int(converged.size - np.count_nonzero(converged))
        finite_errors = reprojection_error[np.isfinite(reprojection_error)]
        maximum_error = (
            float(np.max(finite_errors)) if finite_errors.size else float("inf")
        )
        raise ProjectionError(
            f"THIN_PRISM_FISHEYE inversion failed for {failed_count}/"
            f"{converged.size} samples; maximum reprojection error="
            f"{maximum_error:.9g} px"
        )

    return InverseProjectionResult(
        rays=rays.reshape(original_shape + (3,)),
        converged=converged.reshape(original_shape),
        reprojection_error_pixels=reprojection_error.reshape(original_shape),
        iterations=iterations.reshape(original_shape),
    )


def validate_camera_roundtrip(
    raw_camera: ColmapCamera,
    target_camera: ColmapCamera,
    *,
    grid_size: int = 9,
    maximum_reprojection_error: float = 1.0e-4,
) -> dict[str, float | int]:
    """Validate raw/undistorted model parity over the shared image domain."""

    if raw_camera.model != "THIN_PRISM_FISHEYE":
        raise ValueError("raw_camera must use THIN_PRISM_FISHEYE")
    if target_camera.model != "PINHOLE":
        raise ValueError("target_camera must use PINHOLE")
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    columns, rows = np.meshgrid(
        np.linspace(0.0, target_camera.width - 1.0, grid_size),
        np.linspace(0.0, target_camera.height - 1.0, grid_size),
    )
    target_pixels = raster_indices_to_colmap_pixels(columns, rows).reshape(-1, 2)
    target_rays = unproject_pinhole(target_camera, target_pixels)
    raw_projection = project_thin_prism_fisheye(raw_camera, target_rays)
    raw_raster = colmap_pixels_to_raster_coordinates(raw_projection.pixels)
    inside = raw_projection.valid & (
        (raw_raster[:, 0] >= -0.5)
        & (raw_raster[:, 0] < raw_camera.width - 0.5)
        & (raw_raster[:, 1] >= -0.5)
        & (raw_raster[:, 1] < raw_camera.height - 0.5)
    )
    if not np.any(inside):
        raise ProjectionError("Camera models have no shared round-trip samples")
    inverse = unproject_thin_prism_fisheye(
        raw_camera,
        raw_projection.pixels[inside],
        maximum_reprojection_error=maximum_reprojection_error,
        require_all=True,
    )
    target_reprojection = project_pinhole(target_camera, inverse.rays)
    target_error = np.linalg.norm(
        target_reprojection.pixels - target_pixels[inside], axis=1
    )
    maximum_target_error = float(np.max(target_error))
    if maximum_target_error > maximum_reprojection_error:
        raise ProjectionError(
            "Raw-to-undistorted camera round trip exceeds tolerance: "
            f"{maximum_target_error:.9g} px > {maximum_reprojection_error:.9g} px"
        )
    return {
        "sample_count": int(np.count_nonzero(inside)),
        "raw_reprojection_error_max_pixels": float(
            np.max(inverse.reprojection_error_pixels)
        ),
        "target_reprojection_error_max_pixels": maximum_target_error,
    }


def _point_array(values: np.ndarray, dimensions: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != dimensions:
        raise ValueError(f"{label} must have shape (..., {dimensions})")
    return array


def _fisheye_from_normal(
    normal_u: np.ndarray,
    normal_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    radius = np.hypot(normal_u, normal_v)
    scale = np.ones(radius.shape, dtype=np.float64)
    nonzero = radius > np.finfo(np.float64).eps
    scale[nonzero] = np.arctan(radius[nonzero]) / radius[nonzero]
    return normal_u * scale, normal_v * scale


def _normal_from_fisheye(
    fisheye_u: np.ndarray,
    fisheye_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.hypot(fisheye_u, fisheye_v)
    valid = np.isfinite(theta) & (theta < np.pi / 2.0)
    scale = np.ones(theta.shape, dtype=np.float64)
    nonzero = valid & (theta > np.finfo(np.float64).eps)
    scale[nonzero] = np.tan(theta[nonzero]) / theta[nonzero]
    normal_u = fisheye_u * scale
    normal_v = fisheye_v * scale
    valid &= np.isfinite(normal_u) & np.isfinite(normal_v)
    return normal_u, normal_v, valid


def _distort_fisheye(
    camera: ColmapCamera,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distorted_u, distorted_v, _ = _distort_fisheye_with_jacobian(camera, u, v)
    return distorted_u, distorted_v


def _distort_fisheye_with_jacobian(
    camera: ColmapCamera,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k1, k2, p1, p2, k3, k4, sx1, sy1 = camera.params[4:]
    u2 = u * u
    uv = u * v
    v2 = v * v
    radius2 = u2 + v2
    radius4 = radius2 * radius2
    radius6 = radius4 * radius2
    radius8 = radius6 * radius2
    radial = k1 * radius2 + k2 * radius4 + k3 * radius6 + k4 * radius8
    delta_u = (
        u * radial
        + 2.0 * p1 * uv
        + p2 * (radius2 + 2.0 * u2)
        + sx1 * radius2
    )
    delta_v = (
        v * radial
        + 2.0 * p2 * uv
        + p1 * (radius2 + 2.0 * v2)
        + sy1 * radius2
    )

    radial_derivative = (
        k1 + 2.0 * k2 * radius2 + 3.0 * k3 * radius4 + 4.0 * k4 * radius6
    )
    radial_u = 2.0 * u * radial_derivative
    radial_v = 2.0 * v * radial_derivative
    jacobian = np.empty(u.shape + (2, 2), dtype=np.float64)
    jacobian[..., 0, 0] = (
        1.0 + radial + u * radial_u + 2.0 * p1 * v + 6.0 * p2 * u + 2.0 * sx1 * u
    )
    jacobian[..., 0, 1] = (
        u * radial_v + 2.0 * p1 * u + 2.0 * p2 * v + 2.0 * sx1 * v
    )
    jacobian[..., 1, 0] = (
        v * radial_u + 2.0 * p2 * v + 2.0 * p1 * u + 2.0 * sy1 * u
    )
    jacobian[..., 1, 1] = (
        1.0 + radial + v * radial_v + 2.0 * p2 * u + 6.0 * p1 * v + 2.0 * sy1 * v
    )
    return u + delta_u, v + delta_v, jacobian
