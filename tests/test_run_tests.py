import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "env" / "run_tests.py"
SPEC = importlib.util.spec_from_file_location("plascan_run_tests", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RUN_TESTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_TESTS)


class RunTestsTest(unittest.TestCase):
    def test_default_parallel_jobs_uses_all_logical_cpus(self):
        self.assertEqual(RUN_TESTS.default_parallel_jobs(32), 32)
        self.assertEqual(RUN_TESTS.default_parallel_jobs(7), 7)
        self.assertEqual(RUN_TESTS.default_parallel_jobs(1), 1)
        self.assertEqual(RUN_TESTS.default_parallel_jobs(0), 1)

    def test_command_adds_default_parallelism(self):
        command = RUN_TESTS.build_ctest_command(
            ["--test-dir", "build", "--output-on-failure"],
            environment={},
            logical_cpus=12,
        )

        self.assertEqual(
            command,
            ["ctest", "--parallel", "12", "--test-dir", "build", "--output-on-failure"],
        )

    def test_explicit_ctest_parallelism_is_preserved(self):
        command = RUN_TESTS.build_ctest_command(
            ["--preset", "linux-vcpkg-release", "-j4"],
            environment={},
            logical_cpus=12,
        )

        self.assertEqual(command, ["ctest", "--preset", "linux-vcpkg-release", "-j4"])

    def test_environment_parallelism_is_preserved(self):
        command = RUN_TESTS.build_ctest_command(
            ["--test-dir", "build"],
            environment={"CTEST_PARALLEL_LEVEL": "3"},
            logical_cpus=12,
        )

        self.assertEqual(command, ["ctest", "--test-dir", "build"])

    def test_jobs_rejects_conflicting_ctest_override(self):
        with self.assertRaisesRegex(ValueError, "either --jobs"):
            RUN_TESTS.build_ctest_command(["--parallel", "4"], jobs=2, environment={})

    def test_test_directory_supports_split_and_equals_forms(self):
        root = Path.cwd() / "workspace"
        self.assertEqual(
            RUN_TESTS.ctest_test_directory(
                ["--test-dir", "build/release"], working_directory=root
            ),
            (root / "build/release").resolve(),
        )
        self.assertEqual(
            RUN_TESTS.ctest_test_directory(
                ["--test-dir=build/release"], working_directory=root
            ),
            (root / "build/release").resolve(),
        )

    def test_windows_environment_binds_build_runtime_and_qt_plugins(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            test_directory = root / "build"
            (test_directory / "bin" / "platforms").mkdir(parents=True)
            (test_directory / "tests").mkdir()

            environment = RUN_TESTS.build_test_environment(
                ["--test-dir", str(test_directory)],
                environment={"PATH": "system-runtime"},
                platform_name="nt",
                working_directory=root,
            )

            path_entries = environment["PATH"].split(os.pathsep)
            self.assertEqual(path_entries[0], str(test_directory / "bin"))
            self.assertEqual(path_entries[1], str(test_directory / "tests"))
            self.assertEqual(path_entries[2], "system-runtime")
            self.assertEqual(environment["QT_PLUGIN_PATH"], str(test_directory / "bin"))

    def test_non_windows_environment_is_unchanged(self):
        original = {"PATH": "system-runtime", "QT_PLUGIN_PATH": "custom-plugins"}
        environment = RUN_TESTS.build_test_environment(
            ["--test-dir", "build"],
            environment=original,
            platform_name="posix",
        )
        self.assertEqual(environment, original)


if __name__ == "__main__":
    unittest.main()
