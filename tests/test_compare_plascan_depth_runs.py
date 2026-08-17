import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validation"
    / "compare_plascan_depth_runs.py"
)
SPEC = importlib.util.spec_from_file_location("compare_plascan_depth_runs", SCRIPT_PATH)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)

from plascan_depth_compare.accumulators import DistributionAccumulator


def write_fast_matrix(path: Path, values: np.ndarray, cv_type: int) -> None:
    contiguous = np.ascontiguousarray(values)
    payload = contiguous.tobytes(order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        comparison.FAST_MATRIX_HEADER.pack(
            comparison.FAST_MATRIX_MAGIC,
            contiguous.shape[0],
            contiguous.shape[1],
            cv_type,
            len(payload),
        )
        + payload
    )


def write_run(
    root: Path,
    depth: np.ndarray,
    support: np.ndarray,
    spread: np.ndarray,
    acceptance: str,
    image_path: str = r"C:\images\scene.png",
) -> Path:
    write_fast_matrix(root / "depth.bin", depth.astype("<f4"), comparison.CV_32FC1)
    write_fast_matrix(
        root / "support.bin", support.astype("<u2"), comparison.CV_16UC1
    )
    write_fast_matrix(
        root / "spread.bin", spread.astype("<f4"), comparison.CV_32FC1
    )
    manifest = {
        "frames": [
            {
                "ref_index": 7,
                "ref_image": image_path,
                "status": "completed",
                "acceptance": acceptance,
                "raw_depth_path": "depth.bin",
                "raw_geometry_support_path": "support.bin",
                "raw_inverse_depth_spread_path": "spread.bin",
            }
        ]
    }
    manifest_path = root / "mvs_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class ComparePlascanDepthRunsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.temp_path = Path(self._temporary_directory.name)

    def assertApproxEqual(
        self,
        actual: float,
        expected: float,
        *,
        absolute_tolerance: float = 1.0e-12,
    ) -> None:
        self.assertTrue(
            np.isclose(
                actual,
                expected,
                rtol=1.0e-6,
                atol=absolute_tolerance,
            ),
            msg=f"{actual!r} != {expected!r}",
        )

    def test_reads_supported_fast_matrix_types_and_rejects_payload_mismatch(
        self,
    ) -> None:
        float_path = self.temp_path / "float.bin"
        support_path = self.temp_path / "support.bin"
        write_fast_matrix(
            float_path,
            np.asarray([[1.25, 2.5]], dtype="<f4"),
            comparison.CV_32FC1,
        )
        write_fast_matrix(
            support_path,
            np.asarray([[1, 65535]], dtype="<u2"),
            comparison.CV_16UC1,
        )

        np.testing.assert_allclose(
            comparison.read_fast_matrix(float_path, comparison.CV_32FC1),
            [[1.25, 2.5]],
        )
        np.testing.assert_array_equal(
            comparison.read_fast_matrix(support_path, comparison.CV_16UC1),
            [[1, 65535]],
        )
        with self.assertRaisesRegex(ValueError, "Unexpected PlaScan matrix type"):
            comparison.read_fast_matrix(support_path, comparison.CV_32FC1)

        broken_path = self.temp_path / "broken.bin"
        broken_path.write_bytes(float_path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "file size mismatch"):
            comparison.read_fast_matrix(broken_path, comparison.CV_32FC1)

    def test_compares_relative_manifest_artifacts_and_reports_evidence(
        self,
    ) -> None:
        baseline_depth = np.asarray(
            [[1.0, 2.0, 0.0], [4.0, np.nan, -1.0]], dtype=np.float32
        )
        candidate_depth = np.asarray(
            [[1.1, 0.0, 3.0], [4.4, np.inf, -2.0]], dtype=np.float32
        )
        baseline_manifest = write_run(
            self.temp_path / "baseline",
            baseline_depth,
            np.asarray([[1, 2, 0], [3, 0, 0]], dtype=np.uint16),
            np.asarray([[0.01, 0.02, 0], [0.03, 0, 0]], dtype=np.float32),
            "accepted",
        )
        candidate_manifest = write_run(
            self.temp_path / "candidate",
            candidate_depth,
            np.asarray([[2, 0, 3], [4, 0, 0]], dtype=np.uint16),
            np.asarray([[0.02, 0, 0.04], [0.06, 0, 0]], dtype=np.float32),
            "rejected",
            image_path="/different/root/SCENE.PNG",
        )

        report = comparison.compare_depth_runs(
            baseline_manifest, candidate_manifest, quantile_sample_limit=100
        )
        frame = report["frames"][0]
        aggregate = report["aggregate"]

        self.assertEqual(report["pairing"]["paired_frame_count"], 1)
        self.assertEqual(frame["image_basename"], "scene.png")
        self.assertEqual(frame["baseline"]["depth"]["valid_pixel_count"], 3)
        self.assertEqual(frame["candidate"]["depth"]["valid_pixel_count"], 3)
        self.assertEqual(frame["baseline"]["depth"]["nonfinite_depth_count"], 1)
        self.assertEqual(frame["candidate"]["depth"]["negative_depth_count"], 1)
        self.assertEqual(frame["comparison"]["common_valid_pixel_count"], 2)
        self.assertEqual(frame["comparison"]["baseline_only_pixel_count"], 1)
        self.assertEqual(frame["comparison"]["candidate_only_pixel_count"], 1)
        self.assertApproxEqual(frame["comparison"]["mask_iou"], 0.5)
        self.assertApproxEqual(
            frame["comparison"]["relative_depth_difference"]["p95"],
            0.1,
            absolute_tolerance=1.0e-6,
        )
        self.assertApproxEqual(
            frame["baseline"]["geometry_support"]["ratio_ge_2"],
            2.0 / 3.0,
        )
        self.assertApproxEqual(
            frame["candidate"]["geometry_support"]["ratio_ge_3"],
            2.0 / 3.0,
        )
        self.assertApproxEqual(
            frame["candidate"]["inverse_depth_spread"]["distribution"]["p50"],
            0.04,
        )
        self.assertEqual(
            aggregate["baseline"]["acceptance_counts"], {"accepted": 1}
        )
        self.assertEqual(
            aggregate["candidate"]["fusion_eligible_counts"], {"false": 1}
        )
        self.assertEqual(
            aggregate["comparison"]["acceptance_transitions"],
            {"accepted->rejected": 1},
        )

    def test_latest_duplicate_manifest_record_wins(self) -> None:
        depth = np.asarray([[1.0, 2.0]], dtype=np.float32)
        support = np.asarray([[2, 3]], dtype=np.uint16)
        spread = np.asarray([[0.01, 0.02]], dtype=np.float32)
        baseline_manifest = write_run(
            self.temp_path / "baseline", depth, support, spread, "accepted"
        )
        candidate_manifest = write_run(
            self.temp_path / "candidate", depth, support, spread, "accepted"
        )
        baseline_document = json.loads(
            baseline_manifest.read_text(encoding="utf-8")
        )
        stale = dict(baseline_document["frames"][0])
        stale["status"] = "running"
        stale["raw_depth_path"] = "missing-stale-depth.bin"
        baseline_document["frames"].insert(0, stale)
        baseline_manifest.write_text(
            json.dumps(baseline_document), encoding="utf-8"
        )

        report = comparison.compare_depth_runs(
            baseline_manifest, candidate_manifest
        )

        self.assertEqual(
            report["pairing"]["duplicate_semantics"], "last record wins"
        )
        self.assertEqual(report["pairing"]["baseline_input_record_count"], 2)
        self.assertEqual(report["pairing"]["baseline_duplicate_record_count"], 1)
        self.assertEqual(report["pairing"]["baseline_frame_count"], 1)
        self.assertEqual(report["frames"][0]["baseline"]["status"], "completed")

    def test_distribution_reservoir_is_bounded_deterministic_and_exact_for_totals(
        self,
    ) -> None:
        first = DistributionAccumulator(sample_limit=10, seed=123)
        second = DistributionAccumulator(sample_limit=10, seed=123)
        for accumulator in (first, second):
            accumulator.add(np.arange(7, dtype=np.float64))
            accumulator.add(np.arange(7, 100, dtype=np.float64))

        summary = first.summary()
        self.assertEqual(summary["count"], 100)
        self.assertEqual(summary["sample_count"], 10)
        self.assertTrue(summary["quantiles_approximate"])
        self.assertApproxEqual(summary["mean"], 49.5)
        self.assertEqual(summary["max"], 99.0)
        np.testing.assert_array_equal(first.sample, second.sample)

    def test_relative_depth_uses_float64_without_dropping_extreme_common_pixels(
        self,
    ) -> None:
        baseline_depth = np.asarray([[1.0e-20]], dtype=np.float32)
        candidate_depth = np.asarray([[3.0e38]], dtype=np.float32)
        support = np.asarray([[2]], dtype=np.uint16)
        spread = np.asarray([[0.01]], dtype=np.float32)
        baseline_manifest = write_run(
            self.temp_path / "baseline",
            baseline_depth,
            support,
            spread,
            "accepted",
        )
        candidate_manifest = write_run(
            self.temp_path / "candidate",
            candidate_depth,
            support,
            spread,
            "accepted",
        )

        report = comparison.compare_depth_runs(
            baseline_manifest, candidate_manifest
        )
        relative = report["aggregate"]["comparison"][
            "relative_depth_difference"
        ]

        self.assertEqual(relative["count"], 1)
        self.assertTrue(np.isfinite(relative["p95"]))
        self.assertEqual(relative["max"], relative["p95"])

    def test_optional_gates_write_report_and_return_nonzero(self) -> None:
        baseline_manifest = write_run(
            self.temp_path / "baseline",
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            np.asarray([[2, 2]], dtype=np.uint16),
            np.asarray([[0.01, 0.01]], dtype=np.float32),
            "accepted",
        )
        candidate_manifest = write_run(
            self.temp_path / "candidate",
            np.asarray([[1.2, np.nan]], dtype=np.float32),
            np.asarray([[2, 0]], dtype=np.uint16),
            np.asarray([[0.02, 0.0]], dtype=np.float32),
            "accepted",
        )
        output_path = self.temp_path / "reports" / "comparison.json"
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--baseline-manifest",
            str(baseline_manifest),
            "--candidate-manifest",
            str(candidate_manifest),
            "--output",
            str(output_path),
            "--max-coverage-decline-pp",
            "2",
            "--min-aggregate-mask-iou",
            "0.9",
            "--max-relative-depth-p95",
            "0.05",
            "--require-zero-nonfinite-negative",
        ]
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True
        )

        self.assertEqual(completed.returncode, 2)
        report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["quality_gate"]["mode"], "enforced")
        self.assertFalse(report["quality_gate"]["passed"])
        failure_codes = {
            failure["code"] for failure in report["quality_gate"]["failures"]
        }
        self.assertSetEqual(
            failure_codes,
            {
                "coverage_decline_pp_exceeded",
                "aggregate_mask_iou_below_minimum",
                "relative_depth_p95_exceeded",
                "candidate_nonfinite_or_negative_depth",
            },
        )

        report_only = comparison.compare_depth_runs(
            baseline_manifest, candidate_manifest, quantile_sample_limit=100
        )
        self.assertEqual(
            comparison.evaluate_gates(report_only),
            {
                "mode": "report_only",
                "configured": {
                    "max_coverage_decline_pp": None,
                    "min_aggregate_mask_iou": None,
                    "max_relative_depth_p95": None,
                    "require_zero_candidate_nonfinite_negative": False,
                },
                "passed": True,
                "failures": [],
            },
        )


if __name__ == "__main__":
    unittest.main()
