from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.validation.plascan_texture_eval.cli import main
from scripts.validation.plascan_texture_eval.image_io import load_linear_reference
from scripts.validation.plascan_texture_eval.metrics import seam_metrics, uv_metrics
from scripts.validation.plascan_texture_eval.model import load_textured_obj
from scripts.validation.plascan_texture_eval.rasterizer import PinholeCamera, render_textured_mesh
from scripts.validation.plascan_texture_eval.sampling import sample_texture_bilinear


class TexturedModelEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_plane(self, *, split_uv: bool = False) -> Path:
        texture = np.empty((4, 4, 3), dtype=np.uint8)
        if split_uv:
            texture[:, :2] = (255, 0, 0)
            texture[:, 2:] = (0, 0, 255)
        else:
            texture[:] = (64, 128, 192)
        Image.fromarray(texture, mode="RGB").save(self.root / "texture.png")
        (self.root / "plane.mtl").write_text(
            "newmtl atlas\nmap_Kd texture.png\n", encoding="utf-8"
        )
        texture_coordinates = (
            "vt 0.0 0.0\nvt 0.4 0.0\nvt 0.4 1.0\n"
            "vt 0.6 0.0\nvt 1.0 1.0\nvt 0.6 1.0\n"
            if split_uv
            else "vt 0.0 0.0\nvt 1.0 0.0\nvt 1.0 1.0\nvt 0.0 1.0\n"
        )
        faces = "f 1/1 2/2 3/3\nf 1/4 3/5 4/6\n" if split_uv else "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"
        obj = self.root / "plane.obj"
        obj.write_text(
            "mtllib plane.mtl\n"
            "v -2 -2 1\n"
            "v 2 -2 1\n"
            "v 2 2 1\n"
            "v -2 2 1\n"
            + texture_coordinates
            + "usemtl atlas\n"
            + faces,
            encoding="utf-8",
        )
        return obj

    def _write_duplicate_edge_pair(self, *, edge_offset: float = 0.0) -> Path:
        texture = np.empty((4, 4, 3), dtype=np.uint8)
        texture[:, :2] = (255, 0, 0)
        texture[:, 2:] = (0, 0, 255)
        Image.fromarray(texture, mode="RGB").save(self.root / "duplicate_texture.png")
        (self.root / "duplicate.mtl").write_text(
            "newmtl atlas\nmap_Kd duplicate_texture.png\n", encoding="utf-8"
        )
        obj = self.root / "duplicate.obj"
        obj.write_text(
            "mtllib duplicate.mtl\n"
            "v -2 -2 1\n"
            "v 2 -2 1\n"
            "v 2 2 1\n"
            "v -2 2 1\n"
            f"v {-2.0 + edge_offset} -2 1\n"
            f"v {2.0 + edge_offset} 2 1\n"
            "vt 0.0 0.0\n"
            "vt 0.4 0.0\n"
            "vt 0.4 1.0\n"
            "vt 0.6 0.0\n"
            "vt 1.0 1.0\n"
            "vt 0.6 1.0\n"
            "usemtl atlas\n"
            "f 1/1 2/2 3/3\n"
            "f 5/4 6/5 4/6\n",
            encoding="utf-8",
        )
        return obj

    def _write_anisotropic_triangle(self) -> Path:
        Image.new("RGB", (4, 4), (128, 128, 128)).save(self.root / "anisotropic.png")
        (self.root / "anisotropic.mtl").write_text(
            "newmtl atlas\nmap_Kd anisotropic.png\n", encoding="utf-8"
        )
        obj = self.root / "anisotropic.obj"
        obj.write_text(
            "mtllib anisotropic.mtl\n"
            "v 0 0 1\n"
            "v 2 0 1\n"
            "v 0 1 1\n"
            "vt 0 0\n"
            "vt 1 0\n"
            "vt 0 1\n"
            "usemtl atlas\n"
            "f 1/1 2/2 3/3\n",
            encoding="utf-8",
        )
        return obj

    def _write_triangle(self, name: str, vertices: list[tuple[float, float, float]]) -> Path:
        Image.new("RGB", (2, 2), (80, 120, 160)).save(self.root / f"{name}.png")
        (self.root / f"{name}.mtl").write_text(
            f"newmtl atlas\nmap_Kd {name}.png\n",
            encoding="utf-8",
        )
        vertex_lines = "".join(f"v {x} {y} {z}\n" for x, y, z in vertices)
        obj = self.root / f"{name}.obj"
        obj.write_text(
            f"mtllib {name}.mtl\n"
            + vertex_lines
            + "vt 0.0 0.0\nvt 1.0 0.0\nvt 0.5 1.0\n"
            + "usemtl atlas\nf 1/1 2/2 3/3\n",
            encoding="utf-8",
        )
        return obj

    @staticmethod
    def _camera_json() -> dict[str, object]:
        return {
            "fx": 1.0,
            "fy": 1.0,
            "cx": 1.5,
            "cy": 1.5,
            "rotation_world_to_camera": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "camera_center": [0.0, 0.0, 0.0],
        }

    def test_rasterizes_constant_plane_with_perspective_depth(self) -> None:
        mesh = load_textured_obj(self._write_plane())
        camera = PinholeCamera.from_json(self._camera_json())

        result = render_textured_mesh(mesh, camera, 4, 4)

        self.assertTrue(np.all(np.isfinite(result.depth)))
        np.testing.assert_allclose(result.depth, 1.0, atol=1.0e-12)
        self.assertEqual(result.rendered_face_count, 2)
        self.assertTrue(np.all(result.face_ids >= 0))

    def test_clips_triangle_crossing_near_plane_and_preserves_original_face_id(self) -> None:
        mesh = load_textured_obj(
            self._write_triangle(
                "near_crossing",
                [(-1.0, -1.0, 0.5), (1.0, -1.0, 2.0), (0.0, 1.0, 2.0)],
            )
        )
        camera_value = self._camera_json()
        camera_value.update({"fx": 4.0, "fy": 4.0, "cx": 3.5, "cy": 3.5})

        result = render_textured_mesh(
            mesh,
            PinholeCamera.from_json(camera_value),
            8,
            8,
            near_depth=1.0,
        )

        rendered = result.face_ids == 0
        self.assertEqual(result.near_clipped_face_count, 1)
        self.assertEqual(result.rendered_face_count, 1)
        self.assertTrue(np.any(rendered))
        self.assertTrue(np.all(result.face_ids[rendered] == 0))
        self.assertGreaterEqual(float(np.min(result.depth[rendered])), 1.0)
        self.assertLessEqual(float(np.max(result.depth[rendered])), 2.0)
        self.assertTrue(np.all(result.linear_rgb[rendered] > 0.0))

    def test_discards_triangle_fully_behind_near_plane(self) -> None:
        mesh = load_textured_obj(
            self._write_triangle(
                "behind_near",
                [(-1.0, -1.0, 0.25), (1.0, -1.0, 0.5), (0.0, 1.0, 0.75)],
            )
        )

        result = render_textured_mesh(
            mesh,
            PinholeCamera.from_json(self._camera_json()),
            4,
            4,
            near_depth=1.0,
        )

        self.assertEqual(result.near_clipped_face_count, 1)
        self.assertEqual(result.rendered_face_count, 0)
        self.assertTrue(np.all(result.face_ids == -1))
        self.assertTrue(np.all(np.isinf(result.depth)))
        self.assertTrue(np.all(result.linear_rgb == 0.0))

    def test_reports_real_shared_edge_texture_discontinuity(self) -> None:
        mesh = load_textured_obj(self._write_plane(split_uv=True))

        report = seam_metrics(mesh)

        self.assertEqual(report["shared_edge_count"], 1)
        self.assertEqual(report["seam_edge_count"], 1)
        self.assertGreater(report["linear_rgb_seam_difference_mean"], 0.5)

    def test_seam_adjacency_merges_duplicate_geometric_vertex_indices(self) -> None:
        mesh = load_textured_obj(self._write_duplicate_edge_pair())

        report = seam_metrics(mesh)

        self.assertEqual(report["geometric_vertex_count"], 4)
        self.assertEqual(report["merged_duplicate_vertex_count"], 2)
        self.assertEqual(report["shared_edge_count"], 1)
        self.assertEqual(report["seam_edge_count"], 1)
        self.assertGreater(report["linear_rgb_seam_difference_mean"], 0.5)

    def test_seam_adjacency_keeps_clearly_separated_edges_distinct(self) -> None:
        mesh = load_textured_obj(self._write_duplicate_edge_pair(edge_offset=1.0e-5))

        report = seam_metrics(mesh)

        self.assertEqual(report["merged_duplicate_vertex_count"], 0)
        self.assertEqual(report["shared_edge_count"], 0)
        self.assertEqual(report["seam_edge_count"], 0)

    def test_uv_jacobian_reports_anisotropy_and_conformal_distortion(self) -> None:
        mesh = load_textured_obj(self._write_anisotropic_triangle())

        report = uv_metrics(mesh, occupancy_resolution=32)

        self.assertEqual(report["valid_uv_jacobian_face_count"], 1)
        self.assertAlmostEqual(report["area_weighted_uv_jacobian_sigma_max_mean"], 1.0)
        self.assertAlmostEqual(report["area_weighted_uv_jacobian_sigma_min_mean"], 0.5)
        self.assertAlmostEqual(report["area_weighted_uv_anisotropy_mean"], 2.0)
        self.assertAlmostEqual(report["area_weighted_uv_anisotropy_p95"], 2.0)
        self.assertAlmostEqual(report["area_weighted_uv_conformal_distortion_mean"], 1.25)
        self.assertAlmostEqual(report["area_weighted_uv_conformal_distortion_p95"], 1.25)

    def test_end_to_end_cli_reports_perfect_held_out_render(self) -> None:
        obj = self._write_plane()
        Image.new("RGB", (4, 4), (1, 2, 3)).save(self.root / "training.png")
        Image.new("RGB", (4, 4), (64, 128, 192)).save(self.root / "held_out.png")
        Image.new("L", (4, 4), 255).save(self.root / "held_out_mask.png")
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "plascan.texture_eval.v1",
                    "training_images": ["training.png"],
                    "views": [
                        {
                            "id": "held-out",
                            "image": "held_out.png",
                            "mask": "held_out_mask.png",
                            "camera_model": self._camera_json(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        output = self.root / "evaluation"

        exit_code = main(
            [
                "--obj",
                str(obj),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--occupancy-resolution",
                "32",
            ]
        )

        self.assertEqual(exit_code, 0)
        report = json.loads((output / "texture_evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["aggregate"]["linear_rgb_psnr_db"], 120.0)
        self.assertAlmostEqual(report["aggregate"]["masked_ssim_weighted_mean"], 1.0, places=6)
        self.assertEqual(report["aggregate"]["silhouette_iou"], 1.0)
        self.assertEqual(report["aggregate"]["silhouette_precision"], 1.0)
        self.assertEqual(report["aggregate"]["silhouette_recall"], 1.0)
        self.assertEqual(report["settings"]["evaluation_mode"], "formal_masked")

    def test_rejects_same_image_content_in_training_and_held_out_sets(self) -> None:
        obj = self._write_plane()
        Image.new("RGB", (4, 4), (8, 9, 10)).save(self.root / "training.png")
        (self.root / "held_out.png").write_bytes((self.root / "training.png").read_bytes())
        manifest = self.root / "duplicate.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "plascan.texture_eval.v1",
                    "training_images": ["training.png"],
                    "views": [
                        {
                            "image": "held_out.png",
                            "camera_model": self._camera_json(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        exit_code = main(
            [
                "--obj",
                str(obj),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(self.root / "duplicate_output"),
            ]
        )

        self.assertEqual(exit_code, 2)

    def test_rejects_same_decoded_pixels_with_different_encoding(self) -> None:
        obj = self._write_plane()
        pixels = np.full((4, 4, 3), (8, 9, 10), dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(self.root / "training.png")
        Image.fromarray(pixels, mode="RGB").save(self.root / "held_out.bmp")
        manifest = self.root / "decoded_duplicate.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "plascan.texture_eval.v1",
                    "training_images": ["training.png"],
                    "views": [
                        {
                            "image": "held_out.bmp",
                            "mask": "mask.png",
                            "camera_model": self._camera_json(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        Image.new("L", (4, 4), 255).save(self.root / "mask.png")

        exit_code = main(
            [
                "--obj",
                str(obj),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(self.root / "decoded_duplicate_output"),
            ]
        )

        self.assertEqual(exit_code, 2)

    def test_formal_evaluation_requires_a_matching_size_mask(self) -> None:
        obj = self._write_plane()
        Image.new("RGB", (4, 4), (1, 2, 3)).save(self.root / "training.png")
        Image.new("RGB", (4, 4), (64, 128, 192)).save(self.root / "held_out.png")
        manifest_value = {
            "schema": "plascan.texture_eval.v1",
            "training_images": ["training.png"],
            "views": [
                {
                    "image": "held_out.png",
                    "camera_model": self._camera_json(),
                }
            ],
        }
        manifest = self.root / "missing_mask.json"
        manifest.write_text(json.dumps(manifest_value), encoding="utf-8")

        missing_exit = main(
            [
                "--obj",
                str(obj),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(self.root / "missing_mask_output"),
            ]
        )

        self.assertEqual(missing_exit, 2)
        Image.new("L", (3, 4), 255).save(self.root / "wrong_mask.png")
        manifest_value["views"][0]["mask"] = "wrong_mask.png"
        manifest = self.root / "wrong_mask.json"
        manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
        wrong_size_exit = main(
            [
                "--obj",
                str(obj),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(self.root / "wrong_mask_output"),
            ]
        )

        self.assertEqual(wrong_size_exit, 2)

    def test_camera_scaling_preserves_pixel_center_convention(self) -> None:
        camera = PinholeCamera.from_json(self._camera_json()).scaled(0.5, 0.25)

        self.assertEqual(camera.fx, 0.5)
        self.assertEqual(camera.fy, 0.25)
        self.assertEqual(camera.cx, 0.5)
        self.assertEqual(camera.cy, 0.0)

    def test_texture_sampling_uses_hardware_texel_centers(self) -> None:
        texture = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
        texel_centers = np.asarray(
            [[(index + 0.5) / 4.0, 0.5] for index in range(4)],
            dtype=np.float64,
        )

        sampled = sample_texture_bilinear(texture, texel_centers)

        np.testing.assert_array_equal(sampled, texture[0])

    def test_reference_downscale_happens_in_linear_light(self) -> None:
        pixels = np.asarray([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        path = self.root / "black_white.png"
        Image.fromarray(pixels, mode="RGB").save(path)

        reference = load_linear_reference(path, 1)

        np.testing.assert_allclose(reference.linear_rgb, 0.5, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
