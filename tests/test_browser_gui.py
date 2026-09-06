import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "dev" / "browser_gui.py"
SPEC = importlib.util.spec_from_file_location("plascan_browser_gui", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BROWSER_GUI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BROWSER_GUI)


class BrowserGuiLauncherTest(unittest.TestCase):
    def test_screen_parser_accepts_safe_dimensions(self):
        self.assertEqual((1440, 900), BROWSER_GUI.parse_screen("1440x900"))

    def test_screen_parser_rejects_invalid_or_tiny_dimensions(self):
        for value in ("1440", "0x900", "799x600", "800x599"):
            with self.subTest(value=value), self.assertRaises(Exception):
                BROWSER_GUI.parse_screen(value)

    def test_gateway_host_must_be_loopback(self):
        self.assertEqual("127.0.0.1", BROWSER_GUI.loopback_host("127.0.0.1"))
        self.assertEqual("localhost", BROWSER_GUI.loopback_host("localhost"))
        for host in ("0.0.0.0", "::1", "example.com"):
            with self.subTest(host=host), self.assertRaises(Exception):
                BROWSER_GUI.loopback_host(host)

    def test_browser_url_autoconnects_with_scaled_editable_view(self):
        url = BROWSER_GUI.novnc_url("127.0.0.1", 6081)
        self.assertEqual(
            "http://127.0.0.1:6081/vnc.html?autoconnect=1&resize=scale&view_only=0",
            url,
        )
        self.assertEqual(
            "http://127.0.0.1:6080/?token=session-token",
            BROWSER_GUI.browser_url("127.0.0.1", 6080, "session-token"),
        )

    def test_process_match_rejects_wrong_start_time_and_marker(self):
        start_time = BROWSER_GUI.process_start_time(os.getpid())
        self.assertIsNotNone(start_time)
        command = BROWSER_GUI.process_command(os.getpid())
        marker = Path(sys.executable).name
        self.assertIn(marker, command)
        self.assertTrue(
            BROWSER_GUI.process_matches(
                {"pid": os.getpid(), "start_time": start_time, "marker": marker}
            )
        )
        self.assertFalse(
            BROWSER_GUI.process_matches(
                {"pid": os.getpid(), "start_time": "wrong", "marker": marker}
            )
        )
        self.assertFalse(
            BROWSER_GUI.process_matches(
                {"pid": os.getpid(), "start_time": start_time, "marker": "not-present"}
            )
        )

    def test_state_round_trip_is_atomic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_path = Path(temporary_directory) / "state.json"
            expected = {"version": 1, "processes": {}, "url": "http://localhost"}
            BROWSER_GUI.save_state(state_path, expected)
            self.assertEqual(expected, BROWSER_GUI.load_state(state_path))
            self.assertFalse(state_path.with_suffix(".tmp").exists())
            self.assertEqual(state_path.stat().st_mode & 0o777, 0o600)

    def test_copy_project_for_case_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.plascan"
            source.write_bytes(b"project archive")
            sidecar = root / "source.files"
            (sidecar / "shared/images/hash").mkdir(parents=True)
            (sidecar / "1/assets").mkdir(parents=True)
            (sidecar / "1/mvs_output").mkdir(parents=True)
            (sidecar / "project.zip").write_bytes(b"project metadata")
            (sidecar / "shared/images/hash/image.jpg").write_bytes(b"image")
            (sidecar / "1/assets/model.bin").write_bytes(b"derived asset")
            (sidecar / "1/mvs_output/depth.bin").write_bytes(b"derived depth")
            destination = BROWSER_GUI.copy_project_for_case(source, root / "run")
            self.assertEqual(destination.read_bytes(), b"project archive")
            self.assertEqual(source.read_bytes(), b"project archive")
            self.assertNotEqual(destination, source)
            copied_sidecar = destination.with_name("source.files")
            self.assertEqual((copied_sidecar / "project.zip").read_bytes(), b"project metadata")
            self.assertEqual(
                (copied_sidecar / "shared/images/hash/image.jpg").read_bytes(), b"image"
            )
            self.assertEqual(
                (copied_sidecar / "1/assets/model.bin").read_bytes(), b"derived asset"
            )
            self.assertEqual(
                (copied_sidecar / "1/mvs_output/depth.bin").read_bytes(), b"derived depth"
            )
            self.assertEqual(
                sum(
                    len(payload)
                    for payload in (
                        b"project archive",
                        b"project metadata",
                        b"image",
                        b"derived asset",
                        b"derived depth",
                    )
                ),
                BROWSER_GUI.project_copy_size(source),
            )

    def test_start_accepts_named_fixture_but_not_fixture_and_project_together(self):
        parser = BROWSER_GUI.build_parser()
        args = parser.parse_args(["start", "--fixture", "south_building"])
        self.assertEqual("south_building", args.fixture)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["start", "--fixture", "south_building", "--project", "/tmp/demo.plascan"]
            )

    def test_agent_fixture_copies_images_and_uses_sparse_derived_resources(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.plascan"
            sidecar = root / "source.files"
            source.write_bytes(b"entry")
            (sidecar / "shared/images").mkdir(parents=True)
            (sidecar / "1/assets").mkdir(parents=True)
            (sidecar / "1/mvs_output").mkdir(parents=True)
            (sidecar / "project.zip").write_bytes(b"project")
            (sidecar / "1/chunk.zip").write_bytes(b"chunk")
            (sidecar / "shared/images/photo.jpg").write_bytes(b"photo")
            (sidecar / "1/assets/cloud.ply").write_bytes(b"cloud")
            (sidecar / "1/mvs_output/depth.bin").write_bytes(b"depth")

            destination = BROWSER_GUI.copy_project_for_agent_fixture(source, root / "run")
            copied_sidecar = destination.with_name("source.files")
            self.assertFalse(destination.is_symlink())
            self.assertFalse((copied_sidecar / "project.zip").is_symlink())
            self.assertFalse((copied_sidecar / "1/chunk.zip").is_symlink())
            copied_photo = copied_sidecar / "shared/images/photo.jpg"
            copied_cloud = copied_sidecar / "1/assets/cloud.ply"
            copied_depth = copied_sidecar / "1/mvs_output/depth.bin"
            self.assertFalse(copied_photo.is_symlink())
            self.assertEqual(b"photo", copied_photo.read_bytes())
            self.assertEqual(len(b"cloud"), copied_cloud.stat().st_size)
            self.assertEqual(len(b"depth"), copied_depth.stat().st_size)
            self.assertEqual(b"\0" * len(b"cloud"), copied_cloud.read_bytes())
            self.assertEqual(b"\0" * len(b"depth"), copied_depth.read_bytes())
            (copied_sidecar / "project.zip").write_bytes(b"changed")
            self.assertEqual(b"project", (sidecar / "project.zip").read_bytes())

    def test_cleanup_project_lock_only_removes_matching_stale_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project = root / "source.plascan"
            sidecar = root / "source.files"
            sidecar.mkdir()
            project.touch()
            lock = sidecar / ".plascan.lock"
            state = {
                "project_path": str(project),
                "processes": {
                    "plascan": {
                        "pid": 987654321,
                        "start_time": "not-running",
                        "marker": "plascan",
                    }
                },
            }

            lock.write_text("111\n", encoding="utf-8")
            self.assertFalse(BROWSER_GUI.cleanup_project_lock(state))
            self.assertTrue(lock.exists())

            lock.write_text("987654321\n", encoding="utf-8")
            self.assertTrue(BROWSER_GUI.cleanup_project_lock(state))
            self.assertFalse(lock.exists())

    def test_geospatial_environment_discovers_source_dependency_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            proj = root / "build/source-deps/vcpkg_installed/x64-linux/share/proj"
            gdal = root / "build/source-deps/install/share/gdal"
            proj.mkdir(parents=True)
            gdal.mkdir(parents=True)
            (proj / "proj.db").touch()
            (gdal / "gdalvrt.xsd").touch()

            environment = BROWSER_GUI.geospatial_environment(root)

            self.assertEqual(str(proj), environment["PROJ_DATA"])
            self.assertEqual(str(proj), environment["PROJ_LIB"])
            self.assertEqual(str(gdal), environment["GDAL_DATA"])


if __name__ == "__main__":
    unittest.main()
