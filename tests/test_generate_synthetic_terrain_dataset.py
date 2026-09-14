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
SCRIPT_PATH = ROOT / "scripts" / "validation" / "generate_synthetic_terrain_dataset.py"
SPEC = importlib.util.spec_from_file_location("generate_synthetic_terrain_dataset", SCRIPT_PATH)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


class SyntheticTerrainDatasetTest(unittest.TestCase):
    def small_config(self, output_dir: Path, seed: int = 12345):
        return generator.DatasetConfig(
            output_dir=output_dir,
            quality_level="custom",
            view_count=4,
            image_width=96,
            image_height=72,
            focal_length_px=104.0,
            terrain_width_m=24.0,
            terrain_height_m=18.0,
            camera_height_m=19.0,
            dem_height=48,
            dem_width=64,
            mesh_height=18,
            mesh_width=24,
            texture_height=72,
            texture_width=96,
            detail_layers=2,
            sensor_noise_std=0.0,
            seed=seed,
        )

    def test_dataset_contains_metric_terrain_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "terrain"
            manifest = generator.generate_dataset(self.small_config(output_dir))

            self.assertEqual(manifest["dataset_kind"], "plascan_synthetic_terrain")
            self.assertEqual(manifest["quality_level"], "custom")
            self.assertEqual(manifest["summary"]["view_count"], 4)
            self.assertGreater(manifest["summary"]["minimum_terrain_coverage"], 0.25)
            self.assertGreater(
                manifest["terrain"]["maximum_elevation_m"]
                - manifest["terrain"]["minimum_elevation_m"],
                2.0,
            )

            dem = np.load(output_dir / "ground_truth" / "dem.npy", allow_pickle=False)
            dem_tif = cv2.imread(
                str(output_dir / "ground_truth" / "dem.tif"),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(dem.shape, (48, 64))
            self.assertEqual(dem_tif.shape, (48, 64))
            np.testing.assert_allclose(dem_tif, dem, atol=1.0e-6)
            self.assertTrue((output_dir / "ground_truth" / "terrain.ply").is_file())
            self.assertTrue((output_dir / "ground_truth" / "dom.tfw").is_file())
            self.assertTrue((output_dir / "ground_truth" / "dem_color.png").is_file())

            image = cv2.imread(str(output_dir / "images" / "view_00.png"))
            depth = np.load(
                output_dir / "ground_truth" / "depth" / "view_00.npy",
                allow_pickle=False,
            )
            mask = cv2.imread(
                str(output_dir / "ground_truth" / "masks" / "view_00.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            self.assertEqual(image.shape, (72, 96, 3))
            self.assertTrue(np.isfinite(depth[mask > 0]).all())
            self.assertTrue(np.isnan(depth[mask == 0]).all())
            self.assertGreater(float(np.ptp(depth[mask > 0])), 1.0)
            mvs_mask = cv2.imread(
                str(output_dir / "mvs_masks" / "view_00_mask.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            np.testing.assert_array_equal(mvs_mask, 255 - mask)

            stored = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            for record in stored["files"]:
                path = output_dir / record["path"]
                self.assertEqual(path.stat().st_size, record["size_bytes"])
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"]
                )

    def test_camera_grid_has_metric_baseline_and_valid_rotations(self):
        cameras = generator.camera_grid(self.small_config(Path("unused")))
        centers = np.asarray([camera.center for camera in cameras])
        self.assertGreater(float(np.ptp(centers[:, 0])), 5.0)
        self.assertGreater(float(np.ptp(centers[:, 1])), 5.0)
        for camera in cameras:
            np.testing.assert_allclose(
                camera.camera_to_world.T @ camera.camera_to_world,
                np.eye(3),
                atol=1.0e-12,
            )
            self.assertAlmostEqual(float(np.linalg.det(camera.camera_to_world)), 1.0)

    def test_quality_presets_increase_simulation_fidelity(self):
        coarse = generator.QUALITY_PRESETS["coarse"]
        medium = generator.QUALITY_PRESETS["medium"]
        fine = generator.QUALITY_PRESETS["fine"]
        for key in (
            "view_count",
            "image_width",
            "image_height",
            "dem_height",
            "dem_width",
            "mesh_height",
            "mesh_width",
            "texture_height",
            "texture_width",
            "detail_layers",
        ):
            self.assertLess(coarse[key], medium[key], key)
            self.assertLess(medium[key], fine[key], key)

    def test_same_seed_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            generator.generate_dataset(self.small_config(first))
            generator.generate_dataset(self.small_config(second))
            self.assertEqual(tree_hashes(first), tree_hashes(second))

    def test_seed_changes_dom_but_not_dem(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            generator.generate_dataset(self.small_config(first, seed=1))
            generator.generate_dataset(self.small_config(second, seed=2))
            self.assertNotEqual(
                (first / "ground_truth" / "dom.png").read_bytes(),
                (second / "ground_truth" / "dom.png").read_bytes(),
            )
            np.testing.assert_array_equal(
                np.load(first / "ground_truth" / "dem.npy", allow_pickle=False),
                np.load(second / "ground_truth" / "dem.npy", allow_pickle=False),
            )

    def test_nonempty_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "terrain"
            output_dir.mkdir()
            marker = output_dir / "user-file.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not empty"):
                generator.generate_dataset(self.small_config(output_dir))
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
