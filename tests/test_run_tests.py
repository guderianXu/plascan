import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "env" / "run_tests.py"
SPEC = importlib.util.spec_from_file_location("plascan_run_tests", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RUN_TESTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_TESTS)


class RunTestsTest(unittest.TestCase):
    def test_default_parallel_jobs_uses_half_the_logical_cpus(self):
        self.assertEqual(RUN_TESTS.default_parallel_jobs(32), 16)
        self.assertEqual(RUN_TESTS.default_parallel_jobs(7), 3)
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
            ["ctest", "--parallel", "6", "--test-dir", "build", "--output-on-failure"],
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


if __name__ == "__main__":
    unittest.main()
