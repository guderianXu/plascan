import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import rasterio
import trimesh
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "scripts" / "validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, VALIDATION_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


metrics = load_module("synthetic_e2e_metrics", "synthetic_e2e_metrics.py")
runner = load_module("run_synthetic_e2e", "run_synthetic_e2e.py")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_grid_tif(path: Path, data: np.ndarray, transform=None) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="float32",
        transform=transform or rasterio.Affine.identity(),
    ) as destination:
        destination.write(data.astype(np.float32), 1)


def write_xyza_tif(path: Path, dem: np.ndarray, transform) -> None:
    rows, columns = np.indices(dem.shape)
    x = transform.c + (columns + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=dem.shape[1],
        height=dem.shape[0],
        count=4,
        dtype="float32",
        transform=transform,
    ) as destination:
        destination.write(x.astype(np.float32), 1)
        destination.write(y.astype(np.float32), 2)
        destination.write(dem.astype(np.float32), 3)
        destination.write(np.ones(dem.shape, dtype=np.float32), 4)


def write_object_fixture(root: Path) -> tuple[Path, Path, Path]:
    dataset = root / "object"
    truth = dataset / "ground_truth"
    truth.mkdir(parents=True)
    latitude_count, longitude_count = 36, 72
    latitudes = np.linspace(np.pi * 0.5, -np.pi * 0.5, latitude_count)
    longitudes = np.linspace(-np.pi, np.pi, longitude_count, endpoint=False)
    longitude, latitude = np.meshgrid(longitudes, latitudes)
    radius = 3.0 + 0.15 * np.sin(3.0 * longitude) * np.cos(latitude) ** 2
    directions = np.stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ),
        axis=-1,
    )
    points = (directions * radius[..., None]).reshape(-1, 3)
    np.save(truth / "radial_radius.npy", radius.astype(np.float32))
    np.save(truth / "radial_elevation.npy", (radius - 3.0).astype(np.float32))
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=3.0)
    mesh.export(truth / "object.ply")
    dense = root / "object_dense.ply"
    trimesh.points.PointCloud(points).export(dense)
    pipeline = root / "object_pipeline.json"
    write_json(
        pipeline,
        {
            "status": "ok",
            "sfm": {"registered_images": 12, "points": 100, "mean_reprojection_error": 0.2},
            "dense": {"refined_point_cloud": str(dense)},
            "model": {"model_ply": str(truth / "object.ply")},
            "timings": {"total_elapsed_ms": 100.0},
        },
    )
    radial = root / "radial.tif"
    write_grid_tif(
        radial,
        radius - 3.0,
        from_origin(-180.0, 90.0, 360.0 / longitude_count, 180.0 / latitude_count),
    )
    small_body = root / "small_body.json"
    write_json(small_body, {"elevation_dem_tif": str(radial), "dom_tif": "dom.tif"})
    write_json(
        dataset / "manifest.json",
        {
            "dataset_kind": "plascan_synthetic_3d_object",
            "quality_level": "coarse",
            "seed": 1,
            "summary": {"view_count": 12},
            "object": {"reference_radius_m": 3.0},
        },
    )
    return dataset, pipeline, small_body


def write_terrain_fixture(root: Path) -> tuple[Path, Path]:
    dataset = root / "terrain"
    truth = dataset / "ground_truth"
    truth.mkdir(parents=True)
    height, width = 64, 96
    terrain_width, terrain_height = 24.0, 18.0
    x = np.linspace(-terrain_width * 0.5, terrain_width * 0.5, width)
    y = np.linspace(terrain_height * 0.5, -terrain_height * 0.5, height)
    xx, yy = np.meshgrid(x, y)
    dem = 0.3 * np.sin(0.3 * xx) + 0.2 * np.cos(0.4 * yy)
    np.save(truth / "dem.npy", dem.astype(np.float32))
    vertices = np.column_stack((xx.ravel(), yy.ravel(), dem.ravel()))
    faces = []
    for row in range(height - 1):
        for column in range(width - 1):
            first = row * width + column
            faces.extend([[first, first + width, first + 1], [first + 1, first + width, first + width + 1]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    mesh.export(truth / "terrain.ply")
    dense = root / "terrain_dense.ply"
    trimesh.points.PointCloud(vertices).export(dense)
    candidate_dem = root / "candidate_dem.tif"
    transform = from_origin(
        -terrain_width * 0.5,
        terrain_height * 0.5,
        terrain_width / width,
        terrain_height / height,
    )
    write_grid_tif(candidate_dem, dem, transform)
    dom = root / "dom.png"
    cv2.imwrite(str(dom), np.full((height, width, 3), 128, dtype=np.uint8))
    pipeline = root / "terrain_pipeline.json"
    write_json(
        pipeline,
        {
            "status": "ok",
            "sfm": {"registered_images": 12, "points": 500, "mean_reprojection_error": 0.2},
            "dense": {"refined_point_cloud": str(dense)},
            "model": {"model_ply": str(truth / "terrain.ply")},
            "terrain": {
                "dem": {"dem_path": str(candidate_dem)},
                "dom": {"output_path": str(dom)},
            },
            "timings": {"total_elapsed_ms": 100.0},
        },
    )
    write_json(
        dataset / "manifest.json",
        {
            "dataset_kind": "plascan_synthetic_terrain",
            "quality_level": "coarse",
            "seed": 1,
            "summary": {"view_count": 12},
            "terrain": {"width_m": terrain_width, "height_m": terrain_height},
        },
    )
    return dataset, pipeline


class SyntheticE2ETest(unittest.TestCase):
    def test_perfect_object_fixture_passes_and_writes_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, pipeline, small_body = write_object_fixture(root)
            report = metrics.evaluate_run(dataset, pipeline, small_body, sample_count=4000)
            self.assertEqual(report["status"], "pass")
            self.assertTrue(all(item["passed"] for item in report["checks"]))
            html_path = root / "report.html"
            metrics.write_html_report(report, html_path)
            contents = html_path.read_text(encoding="utf-8")
            self.assertIn("PlaScan 合成数据全流程测试报告", contents)
            self.assertIn("PASS", contents)

    def test_perfect_terrain_fixture_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, pipeline = write_terrain_fixture(Path(tmp))
            report = metrics.evaluate_run(dataset, pipeline, sample_count=5000)
            self.assertEqual(report["status"], "pass")
            self.assertLess(report["metrics"]["products"]["dem"]["error"]["rmse"], 0.03)

    def test_planar_dem_reads_plascan_xyza_z_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            height, width = 24, 32
            terrain_width, terrain_height = 24.0, 18.0
            x = np.linspace(-terrain_width * 0.5, terrain_width * 0.5, width)
            y = np.linspace(terrain_height * 0.5, -terrain_height * 0.5, height)
            xx, yy = np.meshgrid(x, y)
            dem = 0.25 * np.sin(xx * 0.2) + 0.1 * np.cos(yy * 0.3)
            transform = from_origin(
                -terrain_width * 0.5,
                terrain_height * 0.5,
                terrain_width / width,
                terrain_height / height,
            )
            candidate = root / "xyza_dem.tif"
            write_xyza_tif(candidate, dem, transform)
            result = metrics.compare_planar_dem(
                candidate,
                dem,
                terrain_width,
                terrain_height,
            )
            self.assertLess(result["error"]["rmse"], 0.03)

    def test_runner_selects_scene_specific_pipeline_options(self):
        base_args = Namespace(
            device="cpu",
            sfm_quality=None,
            threads=4,
            mvs_quality=None,
            mvs_backend="auto",
            point_cloud_backend="auto",
            mesh_resolution=None,
            dem_resolution_m=None,
        )
        object_command = runner.build_pipeline_command(
            Path("reconstruct_pipeline_cli"),
            Path("object"),
            Path("out"),
            {"dataset_kind": "plascan_synthetic_3d_object"},
            base_args,
            runner.QUALITY_RUN_CONFIG["coarse"],
        )
        self.assertIn("orbital_object", object_command)
        self.assertIn("--sfm-guided-rematching", object_command)
        self.assertIn("--skip-terrain", object_command)
        self.assertIn("object\\mvs_masks", object_command)

        terrain_command = runner.build_pipeline_command(
            Path("reconstruct_pipeline_cli"),
            Path("terrain"),
            Path("out"),
            {"dataset_kind": "plascan_synthetic_terrain"},
            base_args,
            runner.QUALITY_RUN_CONFIG["fine"],
        )
        self.assertIn("aerial_terrain", terrain_command)
        self.assertIn("--dem-resolution", terrain_command)
        self.assertNotIn("--skip-terrain", terrain_command)

    def test_quality_defaults_increase_work(self):
        coarse = runner.QUALITY_RUN_CONFIG["coarse"]
        medium = runner.QUALITY_RUN_CONFIG["medium"]
        fine = runner.QUALITY_RUN_CONFIG["fine"]
        self.assertLess(coarse["mesh_resolution"], medium["mesh_resolution"])
        self.assertLess(medium["mesh_resolution"], fine["mesh_resolution"])
        self.assertLess(coarse["metric_samples"], medium["metric_samples"])
        self.assertLess(medium["metric_samples"], fine["metric_samples"])
        self.assertGreater(coarse["dem_resolution_m"], medium["dem_resolution_m"])
        self.assertGreater(medium["dem_resolution_m"], fine["dem_resolution_m"])
        self.assertLessEqual(coarse["sfm_quality"], medium["sfm_quality"])
        self.assertLess(medium["sfm_quality"], fine["sfm_quality"])
        self.assertEqual(coarse["mvs_quality"], "low")
        self.assertEqual(medium["mvs_quality"], "medium")
        self.assertEqual(fine["mvs_quality"], "high")

    def test_nonempty_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not empty"):
                runner.ensure_empty_output(output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
