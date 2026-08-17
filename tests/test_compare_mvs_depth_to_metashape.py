from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validation"
    / "compare_mvs_depth_to_metashape.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_mvs_depth_to_metashape", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
PREVIOUS_CV2 = sys.modules.get("cv2")
sys.modules["cv2"] = types.ModuleType("cv2")
try:
    SPEC.loader.exec_module(MODULE)
finally:
    if PREVIOUS_CV2 is None:
        sys.modules.pop("cv2", None)
    else:
        sys.modules["cv2"] = PREVIOUS_CV2


class CompareMvsDepthToMetashapeTest(unittest.TestCase):
    def test_global_scale_alignment_removes_uniform_scale_bias(self) -> None:
        summary = MODULE.summarize_global_scale_alignment(
            np.asarray([0.98, 0.98, 0.98], dtype=np.float32)
        )

        self.assertEqual(summary["sample_count"], 3)
        self.assertTrue(
            np.isclose(summary["candidate_to_reference_scale"], 1.0 / 0.98)
        )
        self.assertLess(summary["relative_residual_p99"], 1.0e-12)

    def test_global_scale_alignment_ignores_invalid_ratios(self) -> None:
        summary = MODULE.summarize_global_scale_alignment(
            np.asarray([0.0, -1.0, np.nan, np.inf], dtype=np.float64)
        )

        self.assertEqual(summary["sample_count"], 0)
        self.assertIsNone(summary["candidate_to_reference_scale"])
        self.assertIsNone(summary["relative_residual_p50"])

    def test_camera_center_scale_uses_rigid_invariant_pair_distances(self) -> None:
        candidate = [
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([2.0, 0.0, 0.0]),
        ]
        reference = [
            np.asarray([10.0, 20.0, 30.0]),
            np.asarray([10.0, 22.0, 30.0]),
            np.asarray([10.0, 24.0, 30.0]),
        ]

        summary = MODULE.summarize_camera_center_scale(candidate, reference, 1.99)

        self.assertEqual(summary["matched_camera_count"], 3)
        self.assertEqual(summary["pair_count"], 3)
        self.assertTrue(
            np.isclose(summary["reference_to_candidate_scale_p50"], 2.0)
        )
        self.assertTrue(
            np.isclose(summary["depth_alignment_scale_relative_difference"], -0.005)
        )


if __name__ == "__main__":
    unittest.main()
