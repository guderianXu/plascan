"""Supported COLMAP camera records and pixel-coordinate conventions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_MODEL_PARAMETER_COUNTS = {
    "PINHOLE": 4,
    "THIN_PRISM_FISHEYE": 12,
}


@dataclass(frozen=True)
class ColmapCamera:
    """A supported camera record from COLMAP's ``cameras.txt`` format."""

    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    def __post_init__(self) -> None:
        expected_count = _MODEL_PARAMETER_COUNTS.get(self.model)
        if expected_count is None:
            raise ValueError(f"Unsupported COLMAP camera model: {self.model}")
        if len(self.params) != expected_count:
            raise ValueError(
                f"{self.model} expects {expected_count} parameters, "
                f"got {len(self.params)}"
            )
        if self.camera_id < 0:
            raise ValueError("COLMAP camera id must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("COLMAP camera dimensions must be positive")
        if not np.all(np.isfinite(np.asarray(self.params, dtype=np.float64))):
            raise ValueError("COLMAP camera parameters must all be finite")
        if self.params[0] <= 0.0 or self.params[1] <= 0.0:
            raise ValueError("COLMAP focal lengths must be positive")

    @property
    def fx(self) -> float:
        return self.params[0]

    @property
    def fy(self) -> float:
        return self.params[1]

    @property
    def cx(self) -> float:
        return self.params[2]

    @property
    def cy(self) -> float:
        return self.params[3]

    def as_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "params": list(self.params),
        }


def read_colmap_cameras(path: Path) -> dict[int, ColmapCamera]:
    """Read the supported records in a COLMAP text ``cameras.txt`` file."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"COLMAP cameras file not found: {path}")
    cameras: dict[int, ColmapCamera] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Malformed COLMAP camera at {path}:{line_number}")
        try:
            camera_id = int(fields[0])
            model = fields[1]
            width = int(fields[2])
            height = int(fields[3])
            params = tuple(float(value) for value in fields[4:])
        except ValueError as error:
            raise ValueError(
                f"Malformed COLMAP camera at {path}:{line_number}: {error}"
            ) from error
        if camera_id in cameras:
            raise ValueError(f"Duplicate COLMAP camera id {camera_id}: {path}")
        cameras[camera_id] = ColmapCamera(
            camera_id=camera_id,
            model=model,
            width=width,
            height=height,
            params=params,
        )
    if not cameras:
        raise ValueError(f"No camera records found: {path}")
    return cameras


def select_colmap_camera(
    path: Path,
    expected_model: str,
    camera_id: int | None = None,
) -> ColmapCamera:
    cameras = read_colmap_cameras(path)
    if camera_id is None:
        if len(cameras) != 1:
            raise ValueError(
                f"{path} contains {len(cameras)} cameras; specify a camera id"
            )
        camera = next(iter(cameras.values()))
    else:
        try:
            camera = cameras[camera_id]
        except KeyError as error:
            raise ValueError(f"Camera id {camera_id} is not present in {path}") from error
    if camera.model != expected_model:
        raise ValueError(
            f"Expected {expected_model} camera, got {camera.model}: {path}"
        )
    return camera


def scale_pinhole_camera(
    camera: ColmapCamera,
    width: int,
    height: int,
) -> ColmapCamera:
    """Scale a COLMAP corner-origin PINHOLE camera."""

    if camera.model != "PINHOLE":
        raise ValueError(f"Expected PINHOLE camera, got {camera.model}")
    if width <= 0 or height <= 0:
        raise ValueError("Scaled camera dimensions must be positive")
    scale_x = width / camera.width
    scale_y = height / camera.height
    return ColmapCamera(
        camera_id=camera.camera_id,
        model="PINHOLE",
        width=width,
        height=height,
        params=(
            camera.fx * scale_x,
            camera.fy * scale_y,
            camera.cx * scale_x,
            camera.cy * scale_y,
        ),
    )


def validate_scaled_pinhole_camera(
    official_camera: ColmapCamera,
    prediction_camera: ColmapCamera,
    *,
    maximum_absolute_residual_pixels: float = 1.0e-6,
) -> dict[str, object]:
    """Fail unless two COLMAP cameras describe the same resized raster."""

    if official_camera.model != "PINHOLE":
        raise ValueError(
            f"Expected official PINHOLE camera, got {official_camera.model}"
        )
    if prediction_camera.model != "PINHOLE":
        raise ValueError(
            f"Expected prediction PINHOLE camera, got {prediction_camera.model}"
        )
    if (
        not np.isfinite(maximum_absolute_residual_pixels)
        or maximum_absolute_residual_pixels <= 0.0
    ):
        raise ValueError(
            "maximum_absolute_residual_pixels must be finite and positive"
        )

    expected = scale_pinhole_camera(
        official_camera,
        prediction_camera.width,
        prediction_camera.height,
    )
    names = ("fx", "fy", "cx", "cy")
    residuals = {
        name: float(actual - expected_value)
        for name, actual, expected_value in zip(
            names,
            prediction_camera.params,
            expected.params,
            strict=True,
        )
    }
    maximum_residual = max(abs(value) for value in residuals.values())
    diagnostics = {
        "official_camera_coordinate_convention": "COLMAP corner origin",
        "prediction_camera_coordinate_convention": "COLMAP corner origin",
        "colmap_resize_formula": "f'=f*scale; c'=c*scale",
        "plascan_manifest_principal_formula": "c_manifest'=c_colmap*scale-0.5",
        "scale_x": prediction_camera.width / official_camera.width,
        "scale_y": prediction_camera.height / official_camera.height,
        "expected_camera": expected.as_dict(),
        "intrinsic_residual_pixels": residuals,
        "maximum_absolute_intrinsic_residual_pixels": maximum_residual,
        "maximum_allowed_absolute_residual_pixels": (
            maximum_absolute_residual_pixels
        ),
    }
    if maximum_residual > maximum_absolute_residual_pixels:
        raise ValueError(
            "Prediction camera is not the corner-origin scaling of the "
            "official undistorted camera: maximum intrinsic residual="
            f"{maximum_residual:.9g} px, allowed="
            f"{maximum_absolute_residual_pixels:.9g} px; residuals={residuals}"
        )
    return diagnostics


def raster_indices_to_colmap_pixels(
    columns: np.ndarray | float,
    rows: np.ndarray | float,
) -> np.ndarray:
    """Convert zero-based array indices to COLMAP's corner-origin pixels."""

    columns_array, rows_array = np.broadcast_arrays(
        np.asarray(columns, dtype=np.float64),
        np.asarray(rows, dtype=np.float64),
    )
    return np.stack((columns_array + 0.5, rows_array + 0.5), axis=-1)


def colmap_pixels_to_raster_coordinates(pixels: np.ndarray) -> np.ndarray:
    """Convert COLMAP pixels to zero-based continuous array coordinates."""

    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim == 0 or pixels.shape[-1] != 2:
        raise ValueError("pixels must have shape (..., 2)")
    return pixels - 0.5
