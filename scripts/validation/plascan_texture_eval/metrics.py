"""Appearance, real-edge seam, and UV-layout metrics for textured meshes."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, sobel

from .model import TexturedMesh
from .sampling import sample_texture_bilinear


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    ).astype(np.float32, copy=False)


def _masked_ssim(predicted: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    numeric_mask = mask.astype(np.float64)
    support = gaussian_filter(numeric_mask, sigma=1.5, truncate=3.5, mode="constant")
    core = mask & (support >= 0.999)
    if not np.any(core):
        core = mask & (support >= 0.5)
    support = np.maximum(support, 1.0e-12)
    channel_scores: list[float] = []
    c1 = 0.01**2
    c2 = 0.03**2
    for channel in range(3):
        x = predicted[:, :, channel].astype(np.float64)
        y = reference[:, :, channel].astype(np.float64)
        mean_x = gaussian_filter(x * numeric_mask, 1.5, truncate=3.5, mode="constant") / support
        mean_y = gaussian_filter(y * numeric_mask, 1.5, truncate=3.5, mode="constant") / support
        variance_x = (
            gaussian_filter(x * x * numeric_mask, 1.5, truncate=3.5, mode="constant") / support
            - mean_x * mean_x
        )
        variance_y = (
            gaussian_filter(y * y * numeric_mask, 1.5, truncate=3.5, mode="constant") / support
            - mean_y * mean_y
        )
        covariance = (
            gaussian_filter(x * y * numeric_mask, 1.5, truncate=3.5, mode="constant") / support
            - mean_x * mean_y
        )
        numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
        denominator = (mean_x * mean_x + mean_y * mean_y + c1) * (variance_x + variance_y + c2)
        score = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0.0)
        channel_scores.append(float(np.mean(np.clip(score[core], -1.0, 1.0))))
    return float(np.mean(channel_scores))


def appearance_metrics(predicted: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    predicted = np.asarray(predicted, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if predicted.shape != reference.shape or predicted.ndim != 3 or predicted.shape[2] != 3:
        raise ValueError("Predicted and reference images must have matching HxWx3 shapes")
    if mask.shape != predicted.shape[:2] or not np.any(mask):
        raise ValueError("Appearance mask must contain at least one valid pixel")

    differences = predicted[mask].astype(np.float64) - reference[mask].astype(np.float64)
    squared_error_sum = float(np.sum(differences * differences))
    sample_count = int(differences.size)
    mse = squared_error_sum / sample_count
    psnr = 120.0 if mse <= 1.0e-12 else float(10.0 * np.log10(1.0 / mse))
    per_pixel_error = np.sqrt(np.mean(differences * differences, axis=1))

    luminance_weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    predicted_luma = predicted.astype(np.float64) @ luminance_weights
    reference_luma = reference.astype(np.float64) @ luminance_weights
    gradient_mask = binary_erosion(mask, iterations=2, border_value=0)
    if not np.any(gradient_mask):
        gradient_mask = mask
    predicted_gradient = np.hypot(sobel(predicted_luma, axis=0), sobel(predicted_luma, axis=1))
    reference_gradient = np.hypot(sobel(reference_luma, axis=0), sobel(reference_luma, axis=1))
    predicted_sharpness = float(np.mean(predicted_gradient[gradient_mask]))
    reference_sharpness = float(np.mean(reference_gradient[gradient_mask]))

    return {
        "valid_pixel_count": int(np.count_nonzero(mask)),
        "linear_rgb_sample_count": sample_count,
        "linear_rgb_squared_error_sum": squared_error_sum,
        "linear_rgb_mse": mse,
        "linear_rgb_psnr_db": psnr,
        "linear_rgb_mae": float(np.mean(np.abs(differences))),
        "linear_rgb_rmse_p95": float(np.quantile(per_pixel_error, 0.95)),
        "masked_ssim": _masked_ssim(predicted, reference, mask),
        "linear_rgb_bias": [float(value) for value in np.mean(differences, axis=0)],
        "predicted_luma_gradient_mean": predicted_sharpness,
        "reference_luma_gradient_mean": reference_sharpness,
        "sharpness_ratio": predicted_sharpness / max(reference_sharpness, 1.0e-12),
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return 0.0
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    target = np.clip(quantile, 0.0, 1.0) * cumulative[-1]
    return float(ordered_values[min(int(np.searchsorted(cumulative, target, side="left")), len(values) - 1)])


def _geometric_vertex_ids(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return conservative, scale-aware IDs for numerically coincident OBJ vertices."""
    minimum = np.min(vertices, axis=0)
    diagonal = float(np.linalg.norm(np.max(vertices, axis=0) - minimum))
    maximum_magnitude = float(np.max(np.abs(vertices)))
    tolerance = max(
        diagonal * 1.0e-9,
        np.finfo(np.float64).eps * max(maximum_magnitude, 1.0) * 8.0,
        1.0e-15,
    )
    quantized = np.rint((vertices - minimum) / tolerance).astype(np.int64)
    ids = np.empty(len(vertices), dtype=np.int64)
    positions: list[np.ndarray] = []
    id_by_key: dict[tuple[int, int, int], int] = {}
    for vertex_index, key_values in enumerate(quantized):
        key = (int(key_values[0]), int(key_values[1]), int(key_values[2]))
        geometric_id = id_by_key.get(key)
        if geometric_id is None:
            geometric_id = len(positions)
            id_by_key[key] = geometric_id
            positions.append(vertices[vertex_index])
        ids[vertex_index] = geometric_id
    return ids, np.asarray(positions, dtype=np.float64), tolerance


def seam_metrics(mesh: TexturedMesh, samples_per_edge: int = 9) -> dict[str, object]:
    if samples_per_edge < 2:
        raise ValueError("samples_per_edge must be at least two")
    geometric_ids, geometric_positions, geometric_tolerance = _geometric_vertex_ids(mesh.vertices)
    adjacency: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for face_index, face in enumerate(mesh.faces):
        texture_face = mesh.face_texcoords[face_index]
        for first, second in ((0, 1), (1, 2), (2, 0)):
            geometric_a = int(geometric_ids[int(face[first])])
            geometric_b = int(geometric_ids[int(face[second])])
            if geometric_a == geometric_b:
                continue
            if geometric_a < geometric_b:
                key = (geometric_a, geometric_b)
                texture_edge = (int(texture_face[first]), int(texture_face[second]))
            else:
                key = (geometric_b, geometric_a)
                texture_edge = (int(texture_face[second]), int(texture_face[first]))
            adjacency[key].append((face_index, texture_edge))

    differences: list[float] = []
    lengths: list[float] = []
    nonmanifold_edges = 0
    shared_edges = 0
    seam_edges = 0
    parameters = np.linspace(0.0, 1.0, samples_per_edge, dtype=np.float64)[:, None]
    for edge, uses in adjacency.items():
        if len(uses) != 2:
            nonmanifold_edges += int(len(uses) > 2)
            continue
        shared_edges += 1
        (face_a, uv_indices_a), (face_b, uv_indices_b) = uses
        material_a = int(mesh.face_materials[face_a])
        material_b = int(mesh.face_materials[face_b])
        uv_a = mesh.texcoords[list(uv_indices_a)]
        uv_b = mesh.texcoords[list(uv_indices_b)]
        if material_a == material_b and np.allclose(uv_a, uv_b, rtol=0.0, atol=1.0e-12):
            continue
        seam_edges += 1
        samples_a = uv_a[0] * (1.0 - parameters) + uv_a[1] * parameters
        samples_b = uv_b[0] * (1.0 - parameters) + uv_b[1] * parameters
        colors_a = sample_texture_bilinear(
            mesh.materials[material_a].texture_linear_rgb, samples_a
        )
        colors_b = sample_texture_bilinear(
            mesh.materials[material_b].texture_linear_rgb, samples_b
        )
        edge_difference = float(np.mean(np.linalg.norm(colors_a - colors_b, axis=1) / np.sqrt(3.0)))
        edge_length = float(np.linalg.norm(geometric_positions[edge[1]] - geometric_positions[edge[0]]))
        if np.isfinite(edge_length) and edge_length > 0.0:
            differences.append(edge_difference)
            lengths.append(edge_length)

    difference_array = np.asarray(differences, dtype=np.float64)
    length_array = np.asarray(lengths, dtype=np.float64)
    total_length = float(np.sum(length_array))
    return {
        "geometric_vertex_count": int(len(geometric_positions)),
        "merged_duplicate_vertex_count": int(len(mesh.vertices) - len(geometric_positions)),
        "geometric_vertex_tolerance": geometric_tolerance,
        "shared_edge_count": shared_edges,
        "seam_edge_count": seam_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "seam_world_length": total_length,
        "linear_rgb_seam_difference_mean": (
            float(np.sum(difference_array * length_array) / total_length) if total_length > 0.0 else 0.0
        ),
        "linear_rgb_seam_difference_p95": _weighted_quantile(difference_array, length_array, 0.95),
    }


def _rasterize_uv_occupancy(mesh: TexturedMesh, resolution: int) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for material_index, material in enumerate(mesh.materials):
        occupancy = np.zeros((resolution, resolution), dtype=np.uint16)
        face_indices = np.flatnonzero(mesh.face_materials == material_index)
        for face_index in face_indices:
            uv = mesh.texcoords[mesh.face_texcoords[face_index]]
            pixels = np.column_stack(
                [uv[:, 0] * resolution - 0.5, (1.0 - uv[:, 1]) * resolution - 0.5]
            )
            minimum_x = max(0, int(np.ceil(np.min(pixels[:, 0]))))
            maximum_x = min(resolution - 1, int(np.floor(np.max(pixels[:, 0]))))
            minimum_y = max(0, int(np.ceil(np.min(pixels[:, 1]))))
            maximum_y = min(resolution - 1, int(np.floor(np.max(pixels[:, 1]))))
            if minimum_x > maximum_x or minimum_y > maximum_y:
                continue
            x0, y0 = pixels[0]
            x1, y1 = pixels[1]
            x2, y2 = pixels[2]
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denominator) <= 1.0e-12:
                continue
            grid_y, grid_x = np.mgrid[minimum_y : maximum_y + 1, minimum_x : maximum_x + 1]
            lambda0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
            lambda1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
            inside = (lambda0 >= -1.0e-9) & (lambda1 >= -1.0e-9) & (lambda0 + lambda1 <= 1.0 + 1.0e-9)
            region = occupancy[minimum_y : maximum_y + 1, minimum_x : maximum_x + 1]
            region[inside] = np.minimum(region[inside] + 1, np.iinfo(np.uint16).max)
        reports.append(
            {
                "material": material.name,
                "sample_resolution": resolution,
                "occupied_ratio": float(np.count_nonzero(occupancy) / occupancy.size),
                "overlap_ratio": float(np.count_nonzero(occupancy > 1) / occupancy.size),
                "maximum_overlap": int(np.max(occupancy)),
            }
        )
    return reports


def uv_metrics(mesh: TexturedMesh, occupancy_resolution: int = 512) -> dict[str, object]:
    if occupancy_resolution < 32:
        raise ValueError("occupancy_resolution must be at least 32")
    triangles = mesh.vertices[mesh.faces]
    uv_triangles = mesh.texcoords[mesh.face_texcoords]
    world_areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    uv_edge_a = uv_triangles[:, 1] - uv_triangles[:, 0]
    uv_edge_b = uv_triangles[:, 2] - uv_triangles[:, 0]
    uv_areas = 0.5 * np.abs(uv_edge_a[:, 0] * uv_edge_b[:, 1] - uv_edge_a[:, 1] * uv_edge_b[:, 0])
    valid = (world_areas > 1.0e-15) & (uv_areas > 1.0e-15)
    densities = uv_areas[valid] / world_areas[valid]
    scale = float(np.median(densities)) if densities.size else 1.0
    log_stretch = np.abs(np.log(np.maximum(densities / max(scale, 1.0e-30), 1.0e-30)))
    weights = world_areas[valid]
    weight_sum = float(np.sum(weights))

    # Build a local orthonormal basis on every valid world triangle.  If X maps
    # the triangle's two local world-plane edge coordinates and U maps its UV
    # edges, J = U X^-1 is the differential from world tangent space to UV.
    world_edge_a = triangles[valid, 1] - triangles[valid, 0]
    world_edge_b = triangles[valid, 2] - triangles[valid, 0]
    tangent_x_length = np.linalg.norm(world_edge_a, axis=1)
    tangent_x = world_edge_a / tangent_x_length[:, None]
    world_b_x = np.sum(world_edge_b * tangent_x, axis=1)
    world_b_y = np.linalg.norm(world_edge_b - world_b_x[:, None] * tangent_x, axis=1)
    inverse_world_edges = np.zeros((len(world_edge_a), 2, 2), dtype=np.float64)
    inverse_world_edges[:, 0, 0] = 1.0 / tangent_x_length
    inverse_world_edges[:, 0, 1] = -world_b_x / (tangent_x_length * world_b_y)
    inverse_world_edges[:, 1, 1] = 1.0 / world_b_y
    uv_edges = np.stack((uv_edge_a[valid], uv_edge_b[valid]), axis=2)
    jacobians = uv_edges @ inverse_world_edges
    singular_values = np.linalg.svd(jacobians, compute_uv=False)
    sigma_max = singular_values[:, 0]
    sigma_min = singular_values[:, 1]
    anisotropy = sigma_max / sigma_min
    # This symmetric conformal distortion is one for a similarity transform and
    # grows for angle-distorting anisotropy, while remaining independent of scale.
    conformal_distortion = 0.5 * (anisotropy + 1.0 / anisotropy)

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.sum(values * weights) / weight_sum) if weight_sum > 0.0 else 0.0

    outside_uv = np.any((mesh.texcoords < -1.0e-9) | (mesh.texcoords > 1.0 + 1.0e-9), axis=1)
    return {
        "face_count": int(len(mesh.faces)),
        "valid_uv_jacobian_face_count": int(np.count_nonzero(valid)),
        "degenerate_world_face_count": int(np.count_nonzero(world_areas <= 1.0e-15)),
        "degenerate_uv_face_count": int(np.count_nonzero(uv_areas <= 1.0e-15)),
        "out_of_range_uv_count": int(np.count_nonzero(outside_uv)),
        "median_uv_area_per_world_area": scale,
        "area_weighted_absolute_log_stretch_mean": (
            float(np.sum(log_stretch * weights) / weight_sum) if weight_sum > 0.0 else 0.0
        ),
        "area_weighted_absolute_log_stretch_p95": _weighted_quantile(log_stretch, weights, 0.95),
        "area_weighted_uv_jacobian_sigma_max_mean": weighted_mean(sigma_max),
        "area_weighted_uv_jacobian_sigma_max_p95": _weighted_quantile(sigma_max, weights, 0.95),
        "area_weighted_uv_jacobian_sigma_min_mean": weighted_mean(sigma_min),
        "area_weighted_uv_jacobian_sigma_min_p95": _weighted_quantile(sigma_min, weights, 0.95),
        "area_weighted_uv_anisotropy_mean": weighted_mean(anisotropy),
        "area_weighted_uv_anisotropy_p95": _weighted_quantile(anisotropy, weights, 0.95),
        "area_weighted_uv_conformal_distortion_mean": weighted_mean(conformal_distortion),
        "area_weighted_uv_conformal_distortion_p95": _weighted_quantile(
            conformal_distortion, weights, 0.95
        ),
        "atlas_occupancy": _rasterize_uv_occupancy(mesh, occupancy_resolution),
    }
