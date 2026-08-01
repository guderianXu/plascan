import json
import tempfile
import unittest
import zipfile
from pathlib import Path


from testData.prepare_mun_frl_lidar_ba_project import prepare_project


def write_match_report(matches_dir: Path, image0: Path, image1: Path) -> Path:
    """Create the CLI index used by the project-preparation unit fixture.

    The payload bytes are deliberately opaque here. Production validation is
    covered by the C++ ImageMatchFile tests; this test only verifies project
    metadata transport and must not duplicate the binary parser in Python.
    """
    shard0 = matches_dir / "frame_000001_test.pimatch"
    shard1 = matches_dir / "frame_000002_test.pimatch"
    shard0.write_bytes(b"opaque-pimatch-fixture-0")
    shard1.write_bytes(b"opaque-pimatch-fixture-1")
    report_path = matches_dir / "match_photos_report.json"
    report_path.write_text(
        json.dumps(
            {
                "success": True,
                "pair_match_count": 1,
                "image_match_files": [
                    {
                        "image": str(image0),
                        "output": str(shard0),
                        "neighbors": [str(image1)],
                        "settings": {"algorithm_id": "sift_lightglue"},
                    },
                    {
                        "image": str(image1),
                        "output": str(shard1),
                        "neighbors": [str(image0)],
                        "settings": {"algorithm_id": "sift_lightglue"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return report_path


class PrepareMunFrlLidarBaProjectTest(unittest.TestCase):
    def test_prepare_project_writes_plascan_with_cameras_and_match_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir()
            image0 = images_dir / "frame_000001.jpg"
            image1 = images_dir / "frame_000002.jpg"
            image0.write_bytes(b"jpg0")
            image1.write_bytes(b"jpg1")

            camera_info = root / "camera_info_first.yaml"
            camera_info.write_text(
                "\n".join(
                    [
                        "width: 1440",
                        "height: 1080",
                        "D: [-0.1, 0.01, 0.001, -0.002, 0.0]",
                        "K: [800.0, 0.0, 720.0, 0.0, 801.0, 540.0, 0.0, 0.0, 1.0]",
                    ]
                ),
                encoding="utf-8",
            )

            trajectory = root / "odometry.csv"
            trajectory.write_text(
                "\n".join(
                    [
                        "index,topic,bag_stamp_ns,header_stamp_ns,frame_id,child_frame_id,px,py,pz,qx,qy,qz,qw",
                        "0,/Odometry,1,1000,world,camera,1.0,2.0,3.0,0.0,0.0,0.0,1.0",
                        "1,/Odometry,2,2000,world,camera,2.0,3.0,4.0,0.0,0.0,0.0,1.0",
                    ]
                ),
                encoding="utf-8",
            )

            matches_dir = root / "matches"
            matches_dir.mkdir()
            match_report = write_match_report(matches_dir, image0, image1)

            plan = root / "benchmark_plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "images": [
                            {"path": str(image0), "relative_path": "images/frame_000001.jpg", "stamp_ns": 1000},
                            {"path": str(image1), "relative_path": "images/frame_000002.jpg", "stamp_ns": 2000},
                        ],
                        "laser_constraint_cloud_path": str(root / "merged_lidar_cloud.ply"),
                    }
                ),
                encoding="utf-8",
            )

            output = root / "project"
            result = prepare_project(
                benchmark_plan=plan,
                camera_info=camera_info,
                trajectory=trajectory,
                matches_dir=matches_dir,
                output_dir=output,
                project_name="mun_test",
            )

            self.assertTrue(result.project_path.exists())
            self.assertEqual(result.image_count, 2)
            self.assertEqual(result.match_count, 1)

            chunk_path = result.project_path.with_suffix(".files") / "1" / "chunk.zip"
            with zipfile.ZipFile(chunk_path) as archive:
                document = json.loads(archive.read("doc.json").decode("utf-8"))
                project_files = document["project_files"]
                project_results = document["project_results"]

            self.assertEqual(len(project_files["images"]), 2)
            first_camera = project_files["images"][0]["camera"]
            self.assertEqual(first_camera["fu"], 800.0)
            self.assertEqual(first_camera["fv"], 801.0)
            self.assertEqual(first_camera["C"], [1.0, 2.0, 3.0])
            self.assertEqual(first_camera["k1"], -0.1)
            self.assertEqual(len(project_results["image_match_results"]), 2)
            self.assertEqual(
                project_results["image_match_results"][0]["settings"]["source_report"],
                str(match_report.resolve()),
            )
            self.assertEqual(
                project_results["image_match_results"][0]["settings"]["storage_format"],
                "pimatch",
            )

    def test_prepare_project_applies_tf_static_body_to_camera_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir()
            image0 = images_dir / "frame_000001.jpg"
            image1 = images_dir / "frame_000002.jpg"
            image0.write_bytes(b"jpg0")
            image1.write_bytes(b"jpg1")

            camera_info = root / "camera_info_first.yaml"
            camera_info.write_text(
                "\n".join(
                    [
                        "width: 1440",
                        "height: 1080",
                        "D: [0.0, 0.0, 0.0, 0.0, 0.0]",
                        "K: [800.0, 0.0, 720.0, 0.0, 800.0, 540.0, 0.0, 0.0, 1.0]",
                    ]
                ),
                encoding="utf-8",
            )

            trajectory = root / "odometry.csv"
            trajectory.write_text(
                "\n".join(
                    [
                        "index,topic,bag_stamp_ns,header_stamp_ns,frame_id,child_frame_id,px,py,pz,qx,qy,qz,qw",
                        "0,/Odometry,1,1000,world,lio_body,10.0,20.0,30.0,0.0,0.0,0.0,1.0",
                        "1,/Odometry,2,2000,world,lio_body,11.0,20.0,30.0,0.0,0.0,0.0,1.0",
                    ]
                ),
                encoding="utf-8",
            )

            tf_static = root / "tf_static_unique.csv"
            tf_static.write_text(
                "\n".join(
                    [
                        "topic,bag_stamp_ns,header_stamp_ns,parent_frame_id,child_frame_id,tx,ty,tz,qx,qy,qz,qw",
                        "/tf_static,1,1,camera,imu_link,1.0,0.0,0.0,0.0,0.0,0.0,1.0",
                    ]
                ),
                encoding="utf-8",
            )

            matches_dir = root / "matches"
            matches_dir.mkdir()
            write_match_report(matches_dir, image0, image1)

            plan = root / "benchmark_plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "images": [
                            {"path": str(image0), "relative_path": "images/frame_000001.jpg", "stamp_ns": 1000},
                            {"path": str(image1), "relative_path": "images/frame_000002.jpg", "stamp_ns": 2000},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output = root / "project"
            result = prepare_project(
                benchmark_plan=plan,
                camera_info=camera_info,
                trajectory=trajectory,
                matches_dir=matches_dir,
                output_dir=output,
                project_name="mun_test",
                tf_static=tf_static,
                camera_frame="camera",
                body_frame="imu_link",
            )

            chunk_path = result.project_path.with_suffix(".files") / "1" / "chunk.zip"
            with zipfile.ZipFile(chunk_path) as archive:
                document = json.loads(archive.read("doc.json").decode("utf-8"))
                project_files = document["project_files"]
                project_results = document["project_results"]

            first_camera = project_files["images"][0]["camera"]
            second_camera = project_files["images"][1]["camera"]
            self.assertEqual(first_camera["C"], [9.0, 20.0, 30.0])
            self.assertEqual(second_camera["C"], [10.0, 20.0, 30.0])
            self.assertEqual(first_camera["pose_source"], "mun_frl_camera_info_odometry_tf_static")
            self.assertEqual(first_camera["odometry_child_frame"], "lio_body")
            self.assertEqual(first_camera["tf_static_body_frame"], "imu_link")
            self.assertEqual(project_results["lidar_ba_input"]["tf_static"], str(tf_static.resolve()))


if __name__ == "__main__":
    unittest.main()
