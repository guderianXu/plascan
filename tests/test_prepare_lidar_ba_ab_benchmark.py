import contextlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "testData" / "prepare_lidar_ba_ab_benchmark.py"
SPEC = importlib.util.spec_from_file_location("prepare_lidar_ba_ab_benchmark", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def write_ba_ready_ply(path: Path, vertex_count: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {vertex_count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float normal_x\n"
            "property float normal_y\n"
            "property float normal_z\n"
            "property float curvature\n"
            "end_header\n"
        ).encode("ascii")
        + b"\x00" * 128
    )


def write_binary_ba_ready_ply(path: Path, vertices: list[tuple[float, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float normal_x\n"
        "property float normal_y\n"
        "property float normal_z\n"
        "property float curvature\n"
        "end_header\n"
    )
    body = b"".join(struct.pack("<7f", *vertex) for vertex in vertices)
    path.write_bytes(header.encode("ascii") + body)


def read_merged_ply(path: Path) -> tuple[list[str], bytes]:
    data = path.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii").splitlines()
    return header, data[header_end:]


def write_fixture_dataset(root: Path) -> None:
    association_dir = root / "associations"
    association_dir.mkdir(parents=True)
    rows = [
        "image_index,image_stamp_ns,image_path,lidar_index,lidar_stamp_ns,lidar_path,dt_ms,lidar_points",
        "4,1000,images/color/frame_000004.jpg,0,1100,lidar/cloud_registered/frame_000000.ply,120.0,12",
        "5,1050,images/color/frame_000005.jpg,0,1100,lidar/cloud_registered/frame_000000.ply,50.0,12",
        "6,1100,images/color/frame_000006.jpg,0,1100,lidar/cloud_registered/frame_000000.ply,0.0,12",
        "7,1150,images/color/frame_000007.jpg,1,1200,lidar/cloud_registered/frame_000001.ply,50.0,16",
        "8,1200,images/color/frame_000008.jpg,1,1200,lidar/cloud_registered/frame_000001.ply,0.0,16",
        "9,1250,images/color/frame_000009.jpg,2,1300,lidar/cloud_registered/frame_000002.ply,50.0,18",
    ]
    (association_dir / "color_to_cloud_registered_nearest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    (root / "camera").mkdir()
    (root / "camera" / "camera_info_first.yaml").write_text(
        "width: 1440\nheight: 1080\nK: [853.17, 0.0, 780.32, 0.0, 852.06, 520.69, 0.0, 0.0, 1.0]\n",
        encoding="utf-8",
    )

    for image_index in range(4, 10):
        image_path = root / "images" / "color" / f"frame_{image_index:06d}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"jpeg")

    for lidar_index, vertex_count in [(0, 12), (1, 16), (2, 18)]:
        write_ba_ready_ply(
            root / "lidar" / "cloud_registered" / f"frame_{lidar_index:06d}.ply",
            vertex_count=vertex_count,
        )


class PrepareLidarBaAbBenchmarkTest(unittest.TestCase):
    def test_merge_lidar_clouds_concatenates_compatible_binary_ply_vertices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.ply"
            second = root / "second.ply"
            merged = root / "merged.ply"
            write_binary_ba_ready_ply(
                first,
                [
                    (1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.01),
                    (4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.02),
                ],
            )
            write_binary_ba_ready_ply(
                second,
                [
                    (7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0.03),
                ],
            )

            result = benchmark.merge_lidar_clouds([first, second], merged)

            self.assertEqual(result["input_cloud_count"], 2)
            self.assertEqual(result["merged_vertex_count"], 3)
            header, body = read_merged_ply(merged)
            self.assertIn("format binary_little_endian 1.0", header)
            self.assertIn("element vertex 3", header)
            self.assertEqual(len(body), 3 * struct.calcsize("<7f"))

    def test_write_outputs_can_merge_lidar_clouds_and_update_lidar_run_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "extracted"
            output_dir = Path(tmp) / "benchmark"
            write_fixture_dataset(root)
            write_binary_ba_ready_ply(
                root / "lidar" / "cloud_registered" / "frame_000000.ply",
                [(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.01)],
            )
            write_binary_ba_ready_ply(
                root / "lidar" / "cloud_registered" / "frame_000001.ply",
                [(4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.02)],
            )

            plan = benchmark.build_benchmark_plan(
                dataset_root=root,
                association_csv=root / "associations" / "color_to_cloud_registered_nearest.csv",
                start_index=5,
                window_size=4,
                max_abs_dt_ms=60.0,
            )
            benchmark.write_benchmark_outputs(plan, output_dir, merge_lidar=True)

            summary = json.loads((output_dir / "benchmark_plan.json").read_text(encoding="utf-8"))
            merged_path = output_dir / "merged_lidar_cloud.ply"
            self.assertTrue(merged_path.exists())
            self.assertEqual(summary["merged_lidar_cloud"]["merged_vertex_count"], 2)
            self.assertEqual(
                summary["runs"]["lidar_ba"]["options"]["laser_constraint_cloud_path"],
                str(merged_path),
            )

    def test_select_window_filters_by_time_and_keeps_unique_lidar_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "extracted"
            write_fixture_dataset(root)

            plan = benchmark.build_benchmark_plan(
                dataset_root=root,
                association_csv=root / "associations" / "color_to_cloud_registered_nearest.csv",
                start_index=4,
                window_size=4,
                max_abs_dt_ms=60.0,
            )

            self.assertEqual(plan["window"]["requested_start_index"], 4)
            self.assertEqual(plan["window"]["selected_image_count"], 3)
            self.assertEqual(plan["window"]["rejected_by_dt_count"], 1)
            self.assertEqual([item["image_index"] for item in plan["images"]], [5, 6, 7])
            self.assertEqual([item["lidar_index"] for item in plan["lidar_clouds"]], [0, 1])
            self.assertTrue(plan["readiness"]["has_camera_info_first"])
            self.assertTrue(plan["readiness"]["lidar_stream_ba_ready"])

    def test_write_benchmark_outputs_baseline_and_lidar_run_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "extracted"
            output_dir = Path(tmp) / "benchmark"
            write_fixture_dataset(root)

            plan = benchmark.build_benchmark_plan(
                dataset_root=root,
                association_csv=root / "associations" / "color_to_cloud_registered_nearest.csv",
                start_index=5,
                window_size=4,
                max_abs_dt_ms=60.0,
            )
            benchmark.write_benchmark_outputs(plan, output_dir)

            summary = json.loads((output_dir / "benchmark_plan.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["runs"]["baseline_ba"]["options"]["enable_laser_constraints"])
            self.assertTrue(summary["runs"]["lidar_ba"]["options"]["enable_laser_constraints"])
            self.assertEqual(
                summary["runs"]["lidar_ba"]["options"]["laser_constraint_cloud_path"],
                summary["lidar_clouds"][0]["path"],
            )
            self.assertTrue((output_dir / "images.lis").exists())
            self.assertTrue((output_dir / "lidar_clouds.lis").exists())
            self.assertTrue((output_dir / "README.md").exists())

    def test_main_writes_default_plan_for_fixture_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "extracted"
            output_dir = Path(tmp) / "benchmark"
            write_fixture_dataset(root)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = benchmark.main([
                    "--dataset-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--start-index",
                    "5",
                    "--window-size",
                    "4",
                    "--max-abs-dt-ms",
                    "60",
                ])

            self.assertEqual(exit_code, 0)
            plan = json.loads((output_dir / "benchmark_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["window"]["selected_image_count"], 4)
            self.assertEqual(plan["window"]["unique_lidar_cloud_count"], 2)


if __name__ == "__main__":
    unittest.main()
