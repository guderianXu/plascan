from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.validation.plascan_eth3d_surface_eval.cli import main
from scripts.validation.plascan_eth3d_surface_eval.evaluation import evaluate_surface
from scripts.validation.plascan_eth3d_surface_eval.mesh_io import TriangleMesh, sample_surface


class Eth3dSurfaceEvalTest(unittest.TestCase):
    def _write_ply(
        self, path: Path, vertices: np.ndarray, faces: np.ndarray | None = None
    ) -> None:
        lines = [
            "ply", "format ascii 1.0", f"element vertex {len(vertices)}",
            "property float x", "property float y", "property float z",
        ]
        if faces is not None:
            lines.extend(
                [f"element face {len(faces)}", "property list uchar int vertex_indices"]
            )
        lines.append("end_header")
        lines.extend(" ".join(map(str, vertex)) for vertex in vertices)
        if faces is not None:
            lines.extend("3 " + " ".join(map(str, face)) for face in faces)
        path.write_text("\n".join(lines) + "\n", encoding="ascii")

    def _fixture(self, root: Path) -> tuple[Path, Path]:
        vertices = np.asarray(
            [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
             [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
             [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
             [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]], dtype=np.float64
        )
        faces = np.asarray(
            [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
             [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
             [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64
        )
        mesh_path = root / "mesh.ply"
        self._write_ply(mesh_path, vertices, faces)
        points = sample_surface(TriangleMesh(vertices, faces), 5000, 7)
        scan_path = root / "scan.ply"
        self._write_ply(scan_path, points)
        mlp_path = root / "scan_alignment.mlp"
        mlp_path.write_text(
            "<MeshLabProject><MeshGroup><MLMesh filename=\"scan.ply\">"
            "<MLMatrix44>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</MLMatrix44>"
            "</MLMesh></MeshGroup></MeshLabProject>",
            encoding="utf-8",
        )
        return mesh_path, mlp_path

    def test_evaluates_geometry_and_topology_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mesh, mlp = self._fixture(Path(directory))
            first = evaluate_surface(
                mesh, mlp, voxel_size_m=0.03, mesh_sample_count=10_000, seed=9
            )
            second = evaluate_surface(
                mesh, mlp, voxel_size_m=0.03, mesh_sample_count=10_000, seed=9
            )
            self.assertEqual(first, second)
            self.assertTrue(first["topology"]["watertight"])
            self.assertEqual(first["topology"]["boundary_edge_count"], 0)
            self.assertLess(first["accuracy"]["p95_m"], 0.08)
            self.assertLess(first["completeness"]["p95_m"], 0.08)

    def test_cli_is_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh, mlp = self._fixture(root)
            output = root / "report.json"
            args = [
                "--mesh", str(mesh),
                "--scan-alignment", str(mlp),
                "--output", str(output),
                "--mesh-sample-count", "1000",
            ]
            self.assertEqual(main(args), 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["schema"],
                "plascan.eth3d_surface_evaluation.v1",
            )
            with self.assertRaises(FileExistsError):
                main(args)


if __name__ == "__main__":
    unittest.main()
