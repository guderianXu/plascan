from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from testData.estimate_lidar_normals import parse_ply_layout
from testData.prepare_tanks_ignatius_lidar_ba import (
    align_camera_poses,
    portable_path_token,
    prepare_ignatius,
    reset_output_directory,
)


def write_scan(path: Path, z_offset: float = 0.0) -> None:
    points = [
        (float(x), float(y), z_offset, 100, 110, 120)
        for y in (-1, 0, 1)
        for x in (-1, 0, 1)
    ]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for point in points:
            handle.write(struct.pack("<dddBBB", *point))


def camera_pose(center: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = center
    return pose


def write_pose_log(path: Path, poses: list[np.ndarray]) -> None:
    lines: list[str] = []
    for index, pose in enumerate(poses):
        lines.append(f"{index} {index} 0")
        lines.extend(" ".join(f"{value:.12g}" for value in row) for row in pose)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.write_text(
        "\n".join(" ".join(f"{value:.12g}" for value in row) for row in matrix) + "\n",
        encoding="utf-8",
    )


def parse_tsai_vector(path: Path, field: str) -> np.ndarray:
    prefix = f"{field} = "
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return np.asarray([float(value) for value in line[len(prefix):].split()])
    raise AssertionError(f"missing {field} in {path}")


class PrepareTanksIgnatiusLidarBaTest(unittest.TestCase):
    def test_aligns_cameras_and_builds_oriented_plane_cloud(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ignatius"
            archives = root / "archives"
            images = root / "extracted" / "Ignatius"
            scans = root / "extracted" / "individual_scans" / "Ignatius"
            archives.mkdir(parents=True)
            images.mkdir(parents=True)
            scans.mkdir(parents=True)

            (images / "000001.jpg").write_bytes(b"fixture")
            (images / "000002.jpg").write_bytes(b"fixture")
            write_pose_log(
                archives / "Ignatius_COLMAP_SfM.log",
                [camera_pose((1.0, 0.0, 0.0)), camera_pose((0.0, 1.0, 0.0))],
            )
            rotation_z = np.asarray(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            )
            transform = np.eye(4)
            transform[:3, :3] = 2.0 * rotation_z
            transform[:3, 3] = (10.0, 20.0, 30.0)
            write_matrix(archives / "Ignatius_trans.txt", transform)

            write_scan(scans / "Ignatius01.ply")
            write_scan(scans / "Ignatius02.ply", z_offset=0.01)
            (scans / "scanner_pos.txt").write_text(
                "Ignatius01.ply 0 0 5\nIgnatius02.ply 0 0 5\n",
                encoding="utf-8",
            )

            summary = prepare_ignatius(
                root,
                voxel_size=0.0,
                k_neighbors=8,
                max_points_per_scan=50,
                max_output_points=50,
                chunk_size=4,
                workers=1,
            )

            output = root / "prepared" / "plascan_lidar_ba"
            self.assertEqual(summary["camera_preparation"]["image_count"], 2)
            self.assertAlmostEqual(summary["camera_preparation"]["similarity_scale"], 2.0)
            self.assertTrue(
                np.allclose(
                    parse_tsai_vector(output / "cameras" / "000001.tsai", "C"),
                    (10.0, 22.0, 30.0),
                )
            )
            self.assertEqual(len((output / "image_camera.lis").read_text().splitlines()), 2)

            laser_path = output / "lidar" / "Ignatius_lidar_planes.ply"
            layout = parse_ply_layout(laser_path)
            self.assertEqual(layout.vertex_count, 18)
            self.assertTrue(
                {"x", "y", "z", "normal_x", "normal_y", "normal_z", "curvature"}
                .issubset(layout.fields)
            )
            rows = np.memmap(
                laser_path,
                dtype=layout.dtype,
                mode="r",
                offset=layout.data_offset,
                shape=(layout.vertex_count,),
            )
            self.assertTrue(np.all(rows["normal_z"] > 0.99))
            del rows
            self.assertFalse(
                summary["recommended_bundle_adjust"][
                    "laser_missing_normals_as_height_planes"
                ]
            )
            self.assertEqual(
                summary["recommended_bundle_adjust"]["laser_effective_weight"],
                160000.0,
            )
            self.assertEqual(
                summary["recommended_bundle_adjust"]["laser_sigma_m"],
                0.0025,
            )
            self.assertEqual(
                summary["recommended_bundle_adjust"]["laser_huber_delta_m"],
                0.05,
            )

    def test_rejects_non_similarity_alignment(self):
        transform = np.eye(4)
        transform[0, 0] = 2.0
        transform[1, 1] = 3.0
        with self.assertRaisesRegex(ValueError, "similarity"):
            align_camera_poses([np.eye(4)], transform)

    def test_rejects_image_pose_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ignatius"
            archives = root / "archives"
            images = root / "extracted" / "Ignatius"
            archives.mkdir(parents=True)
            images.mkdir(parents=True)
            (images / "000001.jpg").write_bytes(b"fixture")
            (images / "000002.jpg").write_bytes(b"fixture")
            write_pose_log(archives / "Ignatius_COLMAP_SfM.log", [np.eye(4)])
            write_matrix(archives / "Ignatius_trans.txt", np.eye(4))

            with self.assertRaisesRegex(ValueError, "image/pose count mismatch"):
                prepare_ignatius(root, skip_laser=True)

    def test_overwrite_rejects_dataset_ancestor_and_source_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "ignatius"
            (root / "archives").mkdir(parents=True)
            (root / "extracted").mkdir()

            with self.assertRaisesRegex(ValueError, "contains the source dataset"):
                reset_output_directory(parent, root, overwrite=True)
            with self.assertRaisesRegex(ValueError, "overlaps protected source data"):
                reset_output_directory(root / "archives" / "prepared", root, overwrite=True)

    def test_overwrite_rejects_unmarked_custom_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "ignatius"
            root.mkdir()
            custom_output = parent / "custom-output"
            custom_output.mkdir()
            (custom_output / "keep.txt").write_text("user data", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unmarked custom output"):
                reset_output_directory(custom_output, root, overwrite=True)
            self.assertEqual(
                (custom_output / "keep.txt").read_text(encoding="utf-8"),
                "user data",
            )

    def test_marked_custom_directory_can_be_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "ignatius"
            root.mkdir()
            custom_output = parent / "custom-output"

            reset_output_directory(custom_output, root, overwrite=False)
            (custom_output / "generated.txt").write_text("old", encoding="utf-8")
            reset_output_directory(custom_output, root, overwrite=True)

            self.assertFalse((custom_output / "generated.txt").exists())
            self.assertTrue(
                (custom_output / ".plascan_ignatius_lidar_ba_output").is_file()
            )

    def test_cross_drive_path_token_falls_back_to_absolute_path(self):
        path = Path("D:/dataset/image.jpg")
        with mock.patch("os.path.relpath", side_effect=ValueError("different drives")):
            token = portable_path_token(path, Path("E:/output"))
        self.assertTrue(token.casefold().endswith("d:/dataset/image.jpg"))
        self.assertNotIn("\\", token)


if __name__ == "__main__":
    unittest.main()
