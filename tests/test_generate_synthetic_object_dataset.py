import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "validation" / "generate_synthetic_object_dataset.py"
SPEC = importlib.util.spec_from_file_location("generate_synthetic_object_dataset", SCRIPT_PATH)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


class SyntheticObjectDatasetTest(unittest.TestCase):
    def small_config(self, output_dir: Path, seed: int = 12345):
        return generator.DatasetConfig(
            output_dir=output_dir,
            view_count=4,
            image_width=128,
            image_height=96,
            focal_length_px=110.0,
            reference_radius_m=3.0,
            camera_distance_m=10.0,
            latitude_segments=10,
            longitude_segments=20,
            radial_map_height=32,
            radial_map_width=64,
            texture_height=64,
            texture_width=128,
            seed=seed,
        )

    def test_generated_dataset_is_closed_object_with_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"
            manifest = generator.generate_dataset(self.small_config(output_dir))

            self.assertEqual(manifest["dataset_kind"], "plascan_synthetic_3d_object")
            self.assertEqual(manifest["summary"]["view_count"], 4)
            self.assertGreater(manifest["summary"]["minimum_silhouette_coverage"], 0.12)
            self.assertLess(manifest["summary"]["maximum_silhouette_coverage"], 0.65)
            self.assertGreater(manifest["object"]["mesh_vertices"], 100)
            self.assertGreater(manifest["object"]["mesh_faces"], 200)

            list_lines = (output_dir / "image_camera.lis").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(list_lines), 4)
            self.assertTrue((output_dir / "ground_truth" / "object.ply").is_file())
            self.assertTrue((output_dir / "ground_truth" / "radial_dom.png").is_file())

            radius = np.load(
                output_dir / "ground_truth" / "radial_radius.npy",
                allow_pickle=False,
            )
            elevation = np.load(
                output_dir / "ground_truth" / "radial_elevation.npy",
                allow_pickle=False,
            )
            self.assertEqual(radius.shape, (32, 64))
            self.assertTrue(np.isfinite(radius).all())
            self.assertGreater(float(np.ptp(radius)), 0.5)
            np.testing.assert_allclose(elevation, radius - 3.0, atol=1.0e-6)

            image = cv2.imread(str(output_dir / "images" / "view_00.png"))
            depth = np.load(
                output_dir / "ground_truth" / "depth" / "view_00.npy",
                allow_pickle=False,
            )
            mask = cv2.imread(
                str(output_dir / "ground_truth" / "masks" / "view_00.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            self.assertEqual(image.shape, (96, 128, 3))
            self.assertEqual(depth.shape, (96, 128))
            self.assertEqual(mask.shape, (96, 128))
            self.assertTrue(np.isfinite(depth[mask > 0]).all())
            self.assertTrue(np.isnan(depth[mask == 0]).all())
            self.assertGreater(float(np.ptp(depth[mask > 0])), 0.5)
            mvs_mask = cv2.imread(
                str(output_dir / "mvs_masks" / "view_00_mask.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            np.testing.assert_array_equal(mvs_mask, 255 - mask)

            stored_manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            file_records = {item["path"]: item for item in stored_manifest["files"]}
            self.assertIn("images/view_00.png", file_records)
            self.assertIn("ground_truth/object.ply", file_records)
            self.assertIn("mvs_masks/view_00_mask.png", file_records)
            for relative_path, record in file_records.items():
                path = output_dir / relative_path
                self.assertEqual(path.stat().st_size, record["size_bytes"])
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"]
                )

    def test_cameras_orbit_object_with_valid_rotations(self):
        config = self.small_config(Path("unused"))
        cameras = generator.camera_orbit(config)
        centers = np.asarray([camera.center for camera in cameras])
        self.assertGreater(float(np.ptp(centers[:, 0])), 10.0)
        self.assertGreater(float(np.ptp(centers[:, 1])), 10.0)
        self.assertGreater(float(np.ptp(centers[:, 2])), 3.0)
        for camera in cameras:
            np.testing.assert_allclose(
                camera.camera_to_world.T @ camera.camera_to_world,
                np.eye(3),
                atol=1.0e-12,
            )
            self.assertAlmostEqual(float(np.linalg.det(camera.camera_to_world)), 1.0)

    def test_same_seed_and_config_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            generator.generate_dataset(self.small_config(first))
            generator.generate_dataset(self.small_config(second))
            self.assertEqual(tree_hashes(first), tree_hashes(second))

    def test_different_seed_changes_texture_not_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            generator.generate_dataset(self.small_config(first, seed=1))
            generator.generate_dataset(self.small_config(second, seed=2))
            self.assertNotEqual(
                (first / "images" / "view_00.png").read_bytes(),
                (second / "images" / "view_00.png").read_bytes(),
            )
            np.testing.assert_array_equal(
                np.load(
                    first / "ground_truth" / "radial_radius.npy",
                    allow_pickle=False,
                ),
                np.load(
                    second / "ground_truth" / "radial_radius.npy",
                    allow_pickle=False,
                ),
            )

    def test_nonempty_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"
            output_dir.mkdir()
            marker = output_dir / "user-file.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not empty"):
                generator.generate_dataset(self.small_config(output_dir))
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_quality_presets_increase_simulation_fidelity(self):
        coarse = generator.QUALITY_PRESETS["coarse"]
        medium = generator.QUALITY_PRESETS["medium"]
        fine = generator.QUALITY_PRESETS["fine"]
        for key in (
            "view_count",
            "image_width",
            "image_height",
            "latitude_segments",
            "longitude_segments",
            "radial_map_height",
            "radial_map_width",
            "texture_detail_layers",
        ):
            self.assertLess(coarse[key], medium[key], key)
            self.assertLess(medium[key], fine[key], key)

        args = generator.parse_args(
            ["--output-dir", "unused", "--quality", "fine", "--views", "7"]
        )
        config = generator.config_from_args(args)
        self.assertEqual(config.quality_level, "fine")
        self.assertEqual(config.view_count, 7)
        self.assertEqual(config.image_width, fine["image_width"])


if __name__ == "__main__":
    unittest.main()
