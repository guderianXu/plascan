"""Texture sampling that follows normalized hardware texel-center coordinates."""

from __future__ import annotations

import numpy as np


def sample_texture_bilinear(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample clamp-to-edge UVs with texel centers at ``(index + 0.5) / size``."""
    if texture.ndim != 3 or texture.shape[2] < 1:
        raise ValueError("Texture must have shape (height, width, channels)")
    coordinates = np.asarray(uv, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("UV coordinates must have shape (N, 2)")

    height, width, _ = texture.shape
    x = np.clip(coordinates[:, 0] * width - 0.5, 0.0, width - 1.0)
    y = np.clip(
        (1.0 - coordinates[:, 1]) * height - 0.5,
        0.0,
        height - 1.0,
    )
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = texture[y0, x0] * (1.0 - wx) + texture[y0, x1] * wx
    bottom = texture[y1, x0] * (1.0 - wx) + texture[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy
