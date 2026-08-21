"""Exact ETH3D raw-to-undistorted depth evaluation utilities."""

from .camera import (
    InverseProjectionResult,
    ProjectionError,
    ProjectionResult,
    project_pinhole,
    project_thin_prism_fisheye,
    unproject_pinhole,
    unproject_thin_prism_fisheye,
    validate_camera_roundtrip,
)
from .camera_io import (
    ColmapCamera,
    colmap_pixels_to_raster_coordinates,
    raster_indices_to_colmap_pixels,
    read_colmap_cameras,
    scale_pinhole_camera,
    select_colmap_camera,
    validate_scaled_pinhole_camera,
)
from .depth_io import (
    CV_8UC1,
    CV_32FC1,
    FAST_MATRIX_HEADER,
    FAST_MATRIX_MAGIC,
    LoadedDepth,
    depth_validity_summary,
    read_eth3d_raw_depth,
    read_prediction_depth,
    read_plascan_fast_matrix,
)
from .evaluation import evaluate_depth_prediction
from .image_io import (
    ColmapImagePose,
    read_colmap_image_poses,
    select_colmap_image_pose,
    validate_manifest_pose,
)
from .manifest_io import (
    MVS_WORKSPACE_SCHEMA,
    PredictionManifestFrame,
    read_prediction_manifest_frame,
    validate_manifest_pixel_domain_against_official_camera,
)
from .remap import (
    RemapResult,
    depth_boundary_safe_mask,
    remap_eth3d_depth_to_undistorted,
)

__all__ = [
    "CV_8UC1",
    "CV_32FC1",
    "ColmapCamera",
    "ColmapImagePose",
    "FAST_MATRIX_HEADER",
    "FAST_MATRIX_MAGIC",
    "InverseProjectionResult",
    "LoadedDepth",
    "MVS_WORKSPACE_SCHEMA",
    "PredictionManifestFrame",
    "ProjectionError",
    "ProjectionResult",
    "RemapResult",
    "colmap_pixels_to_raster_coordinates",
    "depth_validity_summary",
    "depth_boundary_safe_mask",
    "evaluate_depth_prediction",
    "project_pinhole",
    "project_thin_prism_fisheye",
    "raster_indices_to_colmap_pixels",
    "read_colmap_cameras",
    "read_colmap_image_poses",
    "read_eth3d_raw_depth",
    "read_prediction_depth",
    "read_plascan_fast_matrix",
    "read_prediction_manifest_frame",
    "remap_eth3d_depth_to_undistorted",
    "scale_pinhole_camera",
    "select_colmap_camera",
    "select_colmap_image_pose",
    "unproject_pinhole",
    "unproject_thin_prism_fisheye",
    "validate_camera_roundtrip",
    "validate_manifest_pose",
    "validate_manifest_pixel_domain_against_official_camera",
    "validate_scaled_pinhole_camera",
]
