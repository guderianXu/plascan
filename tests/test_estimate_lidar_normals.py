import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from testData.estimate_lidar_normals import estimate_normals_for_ply


def write_plane_ply(path: Path) -> None:
    points = [
        (-1.0, -1.0, 0.0, 10.0),
        (0.0, -1.0, 0.0, 11.0),
        (1.0, -1.0, 0.0, 12.0),
        (-1.0, 0.0, 0.0, 13.0),
        (0.0, 0.0, 0.0, 14.0),
        (1.0, 0.0, 0.0, 15.0),
        (-1.0, 1.0, 0.0, 16.0),
        (0.0, 1.0, 0.0, 17.0),
        (1.0, 1.0, 0.0, 18.0),
    ]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float intensity\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for row in points:
            handle.write(struct.pack("<ffff", *row))


def read_normals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    vertex_count = 0
    fields = []
    for line in header.splitlines():
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("property"):
            fields.append(line.split()[-1])
    stride = 4 * len(fields)
    idx = {name: i for i, name in enumerate(fields)}
    normals = []
    curvature = []
    for i in range(vertex_count):
        values = struct.unpack_from("<" + "f" * len(fields), data, header_end + i * stride)
        normals.append([values[idx["normal_x"]], values[idx["normal_y"]], values[idx["normal_z"]]])
        curvature.append(values[idx["curvature"]])
    return np.asarray(normals), np.asarray(curvature)


class EstimateLidarNormalsTest(unittest.TestCase):
    def test_estimates_unit_normals_and_low_curvature_for_plane(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "plane.ply"
            output = root / "plane_normals.ply"
            write_plane_ply(source)

            summary = estimate_normals_for_ply(source, output, k_neighbors=8)

            self.assertEqual(summary["vertex_count"], 9)
            self.assertEqual(summary["valid_normal_count"], 9)
            normals, curvature = read_normals(output)
            self.assertTrue(np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5))
            self.assertTrue(np.all(np.abs(normals[:, 2]) > 0.99))
            self.assertLess(float(np.max(curvature)), 1e-5)


if __name__ == "__main__":
    unittest.main()
