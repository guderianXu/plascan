import contextlib
import hashlib
import importlib.util
import io
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "testData" / "download_photogrammetry_testdata.py"
SPEC = importlib.util.spec_from_file_location("download_photogrammetry_testdata", SCRIPT_PATH)
downloader = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


class DownloadPhotogrammetryTestDataTest(unittest.TestCase):
    def test_default_target_root_is_project_testdata(self):
        expected = Path(__file__).resolve().parents[1] / "testData" / "photogrammetry_benchmarks"

        self.assertEqual(downloader.DEFAULT_TARGET_ROOT, expected)

    def test_catalog_covers_required_reconstruction_scenarios(self):
        categories = {dataset.category for dataset in downloader.DATASETS.values()}

        self.assertIn("classic_sfm_mvs", categories)
        self.assertIn("calibrated_object_mvs", categories)
        self.assertIn("benchmark_depth_gt", categories)
        self.assertIn("real_scene_video", categories)
        self.assertIn("aerial_mapping", categories)
        self.assertIn("planetary_stereo", categories)

    def test_catalog_entries_have_source_and_license_metadata(self):
        self.assertGreaterEqual(len(downloader.DATASETS), 8)

        for dataset_id, dataset in downloader.DATASETS.items():
            with self.subTest(dataset_id=dataset_id):
                self.assertEqual(dataset_id, dataset.dataset_id)
                self.assertTrue(dataset.title)
                self.assertTrue(dataset.source_url.startswith("http"))
                self.assertTrue(dataset.license_note)
                self.assertGreater(len(dataset.resources), 0)
                self.assertTrue(any(resource.url or resource.manual_url for resource in dataset.resources))

    def test_eth3d_office_highres_has_pinned_official_resources(self):
        dataset = downloader.DATASETS["eth3d_office_highres"]
        resources = {resource.filename: resource for resource in dataset.resources}

        self.assertEqual(dataset.category, "benchmark_depth_gt")
        self.assertEqual(
            set(resources),
            {
                "office_dslr_undistorted.7z",
                "office_dslr_scan_eval.7z",
                "office_dslr_depth.7z",
            },
        )
        self.assertEqual(sum(resource.expected_bytes for resource in resources.values()), 883139523)
        for filename, resource in resources.items():
            with self.subTest(filename=filename):
                self.assertEqual(resource.url, f"https://www.eth3d.net/data/{filename}")
                self.assertTrue(resource.large)
                self.assertEqual(len(resource.expected_sha256), 64)

    def test_eth3d_courtyard_highres_has_complete_pinned_scene(self):
        dataset = downloader.DATASETS["eth3d_courtyard_highres"]
        resources = {resource.filename: resource for resource in dataset.resources}

        self.assertEqual(
            set(resources),
            {
                "courtyard_dslr_undistorted.7z",
                "courtyard_dslr_jpg.7z",
                "courtyard_dslr_scan_eval.7z",
                "courtyard_dslr_depth.7z",
            },
        )
        self.assertEqual(sum(resource.expected_bytes for resource in resources.values()), 1536923897)
        for filename, resource in resources.items():
            with self.subTest(filename=filename):
                self.assertEqual(resource.url, f"https://www.eth3d.net/data/{filename}")
                self.assertTrue(resource.large)
                self.assertEqual(len(resource.expected_sha256), 64)

    def test_eth3d_facade_highres_has_complete_scene(self):
        dataset = downloader.DATASETS["eth3d_facade_highres"]
        resources = {resource.filename: resource for resource in dataset.resources}

        self.assertEqual(
            set(resources),
            {
                "facade_dslr_undistorted.7z",
                "facade_dslr_jpg.7z",
                "facade_dslr_scan_eval.7z",
                "facade_dslr_depth.7z",
            },
        )
        self.assertEqual(sum(resource.expected_bytes for resource in resources.values()), 3394284959)
        for filename, resource in resources.items():
            with self.subTest(filename=filename):
                self.assertEqual(resource.url, f"https://www.eth3d.net/data/{filename}")
                self.assertTrue(resource.large)
                self.assertEqual(len(resource.expected_sha256), 64)

    def test_lronac_planetary_example_has_pinned_release_contract(self):
        dataset = downloader.DATASETS["asp_lronac_csm_example"]
        resource = dataset.resources[0]

        self.assertEqual(dataset.category, "planetary_stereo")
        self.assertEqual(resource.filename, "LRONAC_example.tar")
        self.assertEqual(resource.expected_bytes, 204144640)
        self.assertEqual(
            resource.expected_sha256,
            "5307b5a42a829833339e6e3e2991f837f352c11232c1a54c99a20409ed282dfa",
        )

    def test_dawn_fc_planetary_example_is_pinned_frame_camera_data(self):
        dataset = downloader.DATASETS["asp_dawn_fc_vesta_frame"]
        resource = dataset.resources[0]

        self.assertEqual(dataset.category, "planetary_stereo")
        self.assertIn("CSM Frame", dataset.description)
        self.assertEqual(resource.filename, "DawnFramingCamera_example.tar")
        self.assertEqual(resource.expected_bytes, 13189120)
        self.assertEqual(
            resource.expected_sha256,
            "fdb4f595175fd04ec4ec3a063d601c987bbc678d6ece6c98dbffc06fb8a506f7",
        )

    def test_selection_supports_dataset_ids_categories_and_all(self):
        selected = downloader.select_datasets(
            dataset_ids=["colmap_south_building"],
            categories=["planetary_stereo"],
            workflow_tags=[],
            include_all=False,
        )
        selected_ids = {dataset.dataset_id for dataset in selected}

        self.assertIn("colmap_south_building", selected_ids)
        self.assertIn("asp_dawn_fc_vesta_frame", selected_ids)
        self.assertIn("asp_lronac_csm_example", selected_ids)
        self.assertNotIn("middlebury_temple_sparse_ring", selected_ids)

        all_selected = downloader.select_datasets([], [], [], include_all=True)
        self.assertEqual(len(all_selected), len(downloader.DATASETS))

    def test_selection_supports_lidar_workflow_tags(self):
        selected = downloader.select_datasets(
            dataset_ids=[],
            categories=[],
            workflow_tags=["ba_constraint_candidate"],
            include_all=False,
        )
        selected_ids = {dataset.dataset_id for dataset in selected}

        self.assertIn("mun_frl_vil", selected_ids)
        self.assertIn("h3d_hessigheim_uav_lidar", selected_ids)
        self.assertIn("kitti_raw_lidar_camera", selected_ids)
        self.assertNotIn("urbanscene3d", selected_ids)
        self.assertGreaterEqual(len(selected_ids), 4)

    def test_workflow_tag_selection_writes_manual_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            target_root = Path(tmp) / "downloads"
            exit_code = downloader.main([
                "--target-root",
                str(target_root),
                "--workflow-tag",
                "ba_constraint_candidate",
                "--dry-run",
            ])

            manifest = target_root / "mun_frl_vil" / "manifest.json"
            manual = target_root / "mun_frl_vil" / "MANUAL_DOWNLOAD.txt"
            self.assertEqual(exit_code, 0)
            self.assertTrue(manifest.exists())
            self.assertTrue(manual.exists())
            text = manifest.read_text(encoding="utf-8")
            self.assertIn("ba_constraint_candidate", text)

    def test_list_command_prints_catalog_without_creating_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            target_root = Path(tmp) / "downloads"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = downloader.main(["--target-root", str(target_root), "--list"])

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("colmap_south_building", output)
            self.assertIn("eth3d_office_highres", output)
            self.assertIn("eth3d_courtyard_highres", output)
            self.assertIn("eth3d_facade_highres", output)
            self.assertIn("asp_dawn_fc_vesta_frame", output)
            self.assertIn("aerial_mapping", output)
            self.assertFalse(target_root.exists())

    def test_dry_run_writes_manifest_but_does_not_download_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            target_root = Path(tmp) / "downloads"
            exit_code = downloader.main([
                "--target-root",
                str(target_root),
                "--dataset",
                "colmap_south_building",
                "--dry-run",
            ])

            manifest = target_root / "colmap_south_building" / "manifest.json"
            archive_dir = target_root / "colmap_south_building" / "archives"
            self.assertEqual(exit_code, 0)
            self.assertTrue(manifest.exists())
            self.assertFalse(archive_dir.exists())
            text = manifest.read_text(encoding="utf-8")
            self.assertIn("South Building", text)
            self.assertIn("dry_run", text)

    def test_download_url_reports_progress_while_streaming(self):
        class FakeResponse:
            def __init__(self):
                self.headers = {"Content-Length": "6"}
                self._chunks = [b"abc", b"def", b""]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self, _size):
                return self._chunks.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fake.zip"
            resource = downloader.Resource(filename="fake.zip", url="https://example.invalid/fake.zip")
            stdout = io.StringIO()

            with mock.patch.object(downloader.urllib.request, "urlopen", return_value=FakeResponse()):
                with contextlib.redirect_stdout(stdout):
                    status = downloader.download_url(
                        resource,
                        output,
                        timeout=60,
                        retries=0,
                        overwrite=False,
                    )

            self.assertEqual(status, "downloaded")
            self.assertEqual(output.read_bytes(), b"abcdef")
            log = stdout.getvalue()
            self.assertIn("fake.zip", log)
            self.assertIn("100.0%", log)

    def test_download_rejects_truncated_content_before_publish(self):
        class FakeResponse:
            headers = {"Content-Length": "7"}

            def __init__(self):
                self._chunks = [b"abcdef", b""]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self, _size):
                return self._chunks.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "truncated.zip"
            resource = downloader.Resource(
                filename="truncated.zip",
                url="https://example.invalid/truncated.zip",
            )
            with mock.patch.object(downloader.urllib.request, "urlopen", return_value=FakeResponse()):
                with self.assertRaisesRegex(RuntimeError, "Content-Length mismatch"):
                    downloader.download_url(resource, output, timeout=60, retries=0, overwrite=False)

            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".zip.part").exists())

    def test_download_validates_expected_hash_before_publish(self):
        class FakeResponse:
            headers = {"Content-Length": "6"}

            def __init__(self):
                self._chunks = [b"abcdef", b""]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self, _size):
                return self._chunks.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "verified.zip"
            resource = downloader.Resource(
                filename="verified.zip",
                url="https://example.invalid/verified.zip",
                expected_bytes=6,
                expected_sha256=hashlib.sha256(b"abcdef").hexdigest(),
            )
            with mock.patch.object(downloader.urllib.request, "urlopen", return_value=FakeResponse()):
                status = downloader.download_url(
                    resource, output, timeout=60, retries=0, overwrite=False
                )

            self.assertEqual(status, "downloaded")
            self.assertEqual(output.read_bytes(), b"abcdef")

    def test_dataset_reuse_validates_hash_recorded_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            target_root = Path(tmp)
            dataset = downloader.Dataset(
                dataset_id="recorded-contract",
                title="Recorded contract",
                category="test",
                source_url="https://example.invalid/",
                license_note="test",
                description="test",
                resources=(
                    downloader.Resource(
                        filename="cached.zip",
                        url="https://example.invalid/cached.zip",
                    ),
                ),
            )
            dataset_dir = target_root / dataset.dataset_id
            archive_path = dataset_dir / "archives" / "cached.zip"
            archive_path.parent.mkdir(parents=True)
            archive_path.write_bytes(b"damaged")
            downloader.write_manifest(
                dataset,
                dataset_dir,
                "download",
                [
                    {
                        "filename": "cached.zip",
                        "size_bytes": len(b"healthy"),
                        "sha256": hashlib.sha256(b"healthy").hexdigest(),
                    }
                ],
            )

            with self.assertRaisesRegex(
                RuntimeError, "Cached resource validation failed"
            ):
                downloader.download_dataset(
                    dataset=dataset,
                    target_root=target_root,
                    include_large=False,
                    dry_run=False,
                    extract=False,
                    overwrite=False,
                    timeout=60,
                    retries=0,
                )

    def test_safe_tar_extraction_rejects_links_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            link_archive = root / "link.tar"
            with tarfile.open(link_archive, "w") as archive:
                link = tarfile.TarInfo("link")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../outside"
                archive.addfile(link)

            with self.assertRaisesRegex(RuntimeError, "links are not allowed"):
                downloader.safe_extract_tar(link_archive, root / "link-output")

            escape_archive = root / "escape.tar"
            with tarfile.open(escape_archive, "w") as archive:
                member = tarfile.TarInfo("../outside.txt")
                payload = b"escape"
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

            with self.assertRaisesRegex(RuntimeError, "Unsafe archive member path"):
                downloader.safe_extract_tar(escape_archive, root / "escape-output")

    def test_main_reports_download_failure_without_traceback(self):
        stderr = io.StringIO()

        with mock.patch.object(downloader, "download_dataset", side_effect=RuntimeError("network timeout")):
            with contextlib.redirect_stderr(stderr):
                exit_code = downloader.main(["--dataset", "middlebury_temple_sparse_ring"])

        self.assertEqual(exit_code, 1)
        self.assertIn("middlebury_temple_sparse_ring", stderr.getvalue())
        self.assertIn("network timeout", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
